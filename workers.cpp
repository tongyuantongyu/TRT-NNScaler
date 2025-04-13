//
// Created by TYTY on 2023-02-12 012.
//

#include <cmath>
#include <array>
#include <utility>

#include "nn-scaler.h"
#include "infer_engine.h"
#include "reformat/reformat.h"
#include "image_io.h"

#include "absl/flags/flag.h"
#include "logging.h"
#include "libyuv/scale_argb.h"
#include "libyuv/scale_rgb.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"

//#include "reveal.h"

extern InferenceSession *session;

extern int using_io;

extern pixel_importer_gpu<float> *importer_gpu;
extern pixel_exporter_gpu<float> *exporter_gpu;

extern pixel_importer_gpu<half> *importer_gpu_fp16;
extern pixel_exporter_gpu<half> *exporter_gpu_fp16;

cudaEvent_t output_consumed{};
std::once_flag output_consumed_flag;

static cudaEvent_t get_output_consumed() {
  std::call_once(output_consumed_flag, [&] {
    cudaEventCreateWithFlags(&output_consumed, cudaEventBlockingSync | cudaEventDisableTiming);
  });

  return output_consumed;
}

struct WorkContextInternal {
  // filled by launcher
  Work::output_t output;
  std::string alpha_mode;
  double post_scale;

  // filled by image_load
  hr_clock::time_point image_start, tile_start;
  patch<const uint8_t> in_image;
  mem_owner in_memory;// hold by pixel_import

  // filled by pixel_import
  int32_t y, x, th, tw;
  bool h_beg, h_end, w_beg, w_end;
  bool has_alpha, is_alpha, is_begin, is_end;
  std::promise<void> input_consumed;// hold by inference

  // filled by inference

  // alloc by image_load, filled by pixel_export
  patch<uint8_t> out_image;
  mem_owner out_memory;// hold by image_save
  std::future<void> alpha_filtered;// TODO filter scale alpha
};

typedef channel<WorkContextInternal> ichan;

std::string input_repr(Work::input_t &input, bool incr = true) {
  static size_t counter = 0;

  if (input.index() == 0) {
    return u8s(std::get<0>(input));
  }
  else {
    auto idx = incr ? counter++ : counter;
    return "<input memory stream #" + std::to_string(idx) + ">";;
  }
}

std::string output_repr(Work::output_t &output, bool incr = true) {
  static size_t counter = 0;

  if (output.index() == 0) {
    return u8s(std::get<0>(output));
  }
  else {
    auto idx = incr ? counter++ : counter;
    return "<output memory stream #" + std::to_string(idx) + ">";;
  }
}

// Scale down at most 1/2 each time to ensure quality.
static std::pair<md_view<uint8_t, int32_t, 3>, mem_owner> scale_view(md_view<uint8_t, int32_t, 3> src, double scale,
                                                                     mem_owner::alloc_flag flags =
                                                                         mem_owner::alloc_default) {
  const auto [h0, w0, c] = src.shape;
  const int32_t hn = std::round(scale * h0);
  const int32_t wn = std::round(scale * w0);

  int32_t hi = h0, wi = w0;
  int32_t hj, wj;
  md_view<uint8_t, int32_t, 3> scaled_view{};
  mem_owner src_ptr, scaled_ptr;

  do {
    hj = std::max((hi + 1) / 2, hn);
    wj = std::max((wi + 1) / 2, wn);

    std::tie(scaled_view, scaled_ptr) = alloc_buffer<uint8_t>(flags, hj, wj, c);
    if (c == 3) {
      libyuv::RGBScale(src.data,
                       src.at(0).size(),
                       wi,
                       hi,
                       scaled_view.data,
                       scaled_view.at(0).size(),
                       wj,
                       hj,
                       libyuv::kFilterBox);
    }
    else {
      libyuv::ARGBScale(src.data,
                        src.at(0).size(),
                        wi,
                        hi,
                        scaled_view.data,
                        scaled_view.at(0).size(),
                        wj,
                        hj,
                        libyuv::kFilterBox);
    }

    hi = hj;
    wi = wj;
    src = scaled_view;
    src_ptr.swap(scaled_ptr);
  } while (hj != hn && wj != wn);

  return {src, std::move(src_ptr)};
}

static void image_load_worker(chan &in, ichan &out) {
  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto c = std::move(*i);
    auto start = hr_clock::now();

    std::string input = input_repr(c.input);
    std::string err;
    auto img_ret = load_image(std::move(c.input), c.alpha_mode == "ignore");

    if (std::get_if<1>(&img_ret)) {
      c.submitted.set_value("failed reading image: " + std::get<1>(img_ret));
      continue;
    }

    auto [in_shape, in_ptr] = std::move(std::get<0>(img_ret));
    md_view in_view{(in_ptr.get()), in_shape};
    if (c.pre_scale != 1.0) {
      std::tie(in_view, in_ptr) = scale_view(in_view, c.pre_scale, mem_owner::alloc_h2d);
    }

    auto [h, w, ch] = in_view.shape;
    if (h < MinDimension || w < MinDimension) {
      LOG(WARNING) << "Skip too small image " << input << " (" << w << "x" << h << ")";
      c.submitted.set_value("too small image");
      continue;
    }

    VLOG(1) << "Image " << input << " loaded in " << elapsed(start) << "ms, dimension: " << w << "x" << h;

    auto [out_view, out_ptr] = alloc_buffer<uint8_t>(session->scale_h * h, session->scale_w * w, ch);
    c.submitted.set_value("");

#ifndef NDEBUG
    memset(out_ptr.get(), 0, in_view.size() * session->scale_h * session->scale_w);
#endif

    std::string output = output_repr(c.output, false);
    out.put(WorkContextInternal{
        .output = std::move(c.output),
        .alpha_mode = c.alpha_mode,
        .post_scale = c.post_scale,

        .image_start = start,
        .in_image = in_view.as_wuview(),
        .in_memory = std::move(in_ptr),

        .out_image = out_view.as_wuview(),
        .out_memory = std::move(out_ptr),
    });
    VLOG(3) << "Image " << output << " sent.";
  }

  out.close();
}

// v loaded image

template<typename I, typename Task>
static bool split_range(I total, I step, I overlap, I grace, Task task) {
  I current = 0, tile;
  bool beg = true, end = false;
  while (true) {
    I remain = total - current;
    if (remain <= step + grace) {
      tile = remain;
      end = true;
    }
    else {
      tile = step;
    }

    if (!task(current, tile, beg, end)) {
      return false;
    }

    if (end) {
      return true;
    }

    beg = false;
    current += step - overlap;
  }
}

ABSL_DECLARE_FLAG(uint32_t, tile_width);
ABSL_DECLARE_FLAG(uint32_t, tile_height);
ABSL_DECLARE_FLAG(uint32_t, tile_pad);
ABSL_DECLARE_FLAG(uint32_t, extend_grace);
ABSL_DECLARE_FLAG(uint32_t, alignment);

static int32_t align(int32_t n, size_t alignment) {
  n += alignment - 1;
  return n - (n % alignment);
}

static void pixel_import_worker(ichan &in, ichan &out) {
  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto ctx = std::move(*i);

    auto [h, w, c] = ctx.in_image.shape;
    auto process_alpha = ctx.alpha_mode == "nn" && c == 4;
    int32_t h_split = align(h, absl::GetFlag(FLAGS_alignment)), w_split = align(w, absl::GetFlag(FLAGS_alignment));

    split_range<int32_t>(
        h_split, absl::GetFlag(FLAGS_tile_height), absl::GetFlag(FLAGS_tile_pad), absl::GetFlag(FLAGS_extend_grace),
        [&, h = h, w = w](int32_t y, int32_t th, bool h_beg, bool h_end) {
          return split_range<int32_t>(
              w_split, absl::GetFlag(FLAGS_tile_width), absl::GetFlag(FLAGS_tile_pad),
              absl::GetFlag(FLAGS_extend_grace),
              [&](int32_t x, int32_t tw, bool w_beg, bool w_end) -> bool {
                auto tile_start = hr_clock::now();

                auto input_tile = ctx.in_image.slice<0>(y, std::min(y + th, h)).slice<1>(x, std::min(x + tw, w));
                auto input_tensor = session->input<float>(th, tw);
                auto input_tensor_fp16 = session->input<half>(th, tw);

                bool first_tile = h_beg && w_beg;

                if (process_alpha) {
                  auto alpha_start = hr_clock::now();
                  std::string ret;
                  switch (using_io) {
                    case 0: ret = importer_gpu->import_pixel<uint8_t>(input_tensor, input_tile, true, session->stream);
                      break;
                    case 1: ret = importer_gpu_fp16->import_pixel<uint8_t>(
                                input_tensor_fp16, input_tile, true, session->stream);
                      break;
                    default:
                      LOG(QFATAL) << "Unknown IO mode.";
                  }

                  if (!ret.empty()) {
                    LOG(QFATAL) << "Unexpected error importing pixel: " << ret;
                  }

                  WorkContextInternal tile_ctx = {
                      .alpha_mode = ctx.alpha_mode, .post_scale = ctx.post_scale,
                      .tile_start = alpha_start,
                      .y = y, .x = x, .th = th, .tw = tw,
                      .h_beg = h_beg, .h_end = h_end, .w_beg = w_beg, .w_end = w_end,
                      .has_alpha = true, .is_alpha = true,
                      .out_image = ctx.out_image,
                  };

                  if (first_tile) {
                    tile_ctx.is_begin = true;
                    tile_ctx.output = std::move(ctx.output);
                    tile_ctx.image_start = ctx.image_start;
                    tile_ctx.out_memory = std::move(ctx.out_memory);
                    first_tile = false;
                  }

                  VLOG(3) << "Tile "
                          << std::setw(4) << tw << 'x'
                          << std::setw(4) << th << '+'
                          << std::setw(4) << x << '+'
                          << std::setw(4) << y << " alpha imported in "
                          << elapsed(alpha_start) << "ms";

                  auto input_done = tile_ctx.input_consumed.get_future();
                  out.put(std::move(tile_ctx));
                  input_done.get();
                  // cudaStreamWaitEvent(session->stream, session->input_consumed);

                  VLOG(3) << "Tile "
                          << std::setw(4) << tw << 'x'
                          << std::setw(4) << th << '+'
                          << std::setw(4) << x << '+'
                          << std::setw(4) << y << " alpha input consumed in "
                          << elapsed(alpha_start) << "ms";

                }

                std::string ret;
                switch (using_io) {
                  case 0: ret = importer_gpu->import_pixel<uint8_t>(input_tensor, input_tile, false, session->stream);
                    break;
                  case 1: ret = importer_gpu_fp16->import_pixel<uint8_t>(
                              input_tensor_fp16, input_tile, false, session->stream);
                    break;
                  default:
                    LOG(QFATAL) << "Unknown IO mode.";
                }

                if (!ret.empty()) {
                  LOG(QFATAL) << "Unexpected error importing pixel: " << ret;
                }

                WorkContextInternal tile_ctx{
                    .alpha_mode = ctx.alpha_mode, .post_scale = ctx.post_scale,
                    .tile_start = tile_start,
                    .y = y, .x = x, .th = th, .tw = tw,
                    .h_beg = h_beg, .h_end = h_end, .w_beg = w_beg, .w_end = w_end,
                    .has_alpha = process_alpha, .is_alpha = false,
                    .out_image = ctx.out_image,
                };

                VLOG(3) << "Tile "
                        << std::setw(4) << tw << 'x'
                        << std::setw(4) << th << '+'
                        << std::setw(4) << x << '+'
                        << std::setw(4) << y << " imported in "
                        << elapsed(tile_start) << "ms";

                if (first_tile) {
                  tile_ctx.is_begin = true;
                  tile_ctx.output = std::move(ctx.output);
                  tile_ctx.image_start = ctx.image_start;
                  tile_ctx.out_memory = std::move(ctx.out_memory);
                }

                if (h_end && w_end) {
                  tile_ctx.is_end = true;
                }

                auto input_done = tile_ctx.input_consumed.get_future();
                out.put(std::move(tile_ctx));
                input_done.get();

                VLOG(3) << "Tile "
                        << std::setw(4) << tw << 'x'
                        << std::setw(4) << th << '+'
                        << std::setw(4) << x << '+'
                        << std::setw(4) << y << " input consumed in "
                        << elapsed(tile_start) << "ms ";

                return true;
              }
              );
        });
  }

  out.close();
}

// v pixel loaded notice     ^ input consumed via cudaEvent

static void inference_worker(ichan &in, ichan &out) {
  // Init CUDA context for this thread
  // TensorRT may do some query that requires the thread to have an initialized CUDA context.
  cudaFree(nullptr);

  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto ctx = std::move(*i);

    session->config(1, ctx.th, ctx.tw);
    if (!session->inference()) {
      LOG(QFATAL) << "CUDA error during inference: " << cudaGetErrorName(cudaGetLastError());
    }
    ctx.input_consumed.set_value();
    out.put(std::move(ctx));
  }

  out.close();
}

// v pixel produced notice (with consumed promise)   ^ fulfill promise

static void pixel_export_worker(ichan &in, ichan &out) {
  Work::output_t output;
  hr_clock::time_point start;
  mem_owner in_memory, out_memory;

  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto ctx = std::move(*i);

    if (ctx.is_begin) {
      // begin of image
      output = std::move(ctx.output);
      start = ctx.image_start;
      out_memory = std::move(ctx.out_memory);
    }

    auto output_tensor = session->output<float>(ctx.th, ctx.tw);
    auto output_tensor_fp16 = session->output<half>(ctx.th, ctx.tw);
    pad_descriptor pad_desc{static_cast<int32_t>(absl::GetFlag(FLAGS_tile_pad) * session->scale_h), ctx.h_beg,
                            ctx.h_end, ctx.w_beg, ctx.w_end};
    auto [h, w, _] = ctx.out_image.shape;
    auto out_tile = ctx.out_image
                       .slice<0>(session->scale_h * ctx.y, std::min(session->scale_h * (ctx.y + ctx.th), h))
                       .slice<1>(session->scale_w * ctx.x, std::min(session->scale_w * (ctx.x + ctx.tw), w));

    std::string ret;
    if (ctx.is_alpha) {
      switch (using_io) {
        case 0: ret = exporter_gpu->export_pixel(output_tensor, out_tile, true, pad_desc, session->stream);
          break;
        case 1: ret = exporter_gpu_fp16->export_pixel(output_tensor_fp16, out_tile, true, pad_desc, session->stream);
          break;
        default:
          LOG(QFATAL) << "Unknown IO mode.";
      }
    }
    else {
      switch (using_io) {
        case 0: ret = exporter_gpu->export_pixel<uint8_t>(output_tensor, out_tile, false, pad_desc, session->stream);
          break;
        case 1: ret = exporter_gpu_fp16->export_pixel<uint8_t>(output_tensor_fp16, out_tile, false, pad_desc,
                                                               session->stream);
          break;
        default:
          LOG(QFATAL) << "Unknown IO mode.";
      }
    }
    if (!ret.empty()) {
      LOG(QFATAL) << "Unexpected error fetching result pixel: " << ret;
    }

    VLOG(2) << "Tile "
            << std::setw(4) << ctx.tw << 'x'
            << std::setw(4) << ctx.th << '+'
            << std::setw(4) << ctx.x << '+'
            << std::setw(4) << ctx.y << " scale done in "
            << elapsed(ctx.tile_start) << "ms";

    if (ctx.is_end) {
      ctx.output = std::move(output);
      ctx.image_start = start;
      ctx.out_memory = std::move(out_memory);
      auto result = cudaEventRecord(get_output_consumed(), session->stream);
      if (result != cudaSuccess) {
        LOG(QFATAL) << "cudaEventRecord Error: " << cudaGetErrorName(result);
      }
      out.put(std::move(ctx));
    }
  }

  out.close();
}

// v output image

static void image_save_worker(ichan &in, chan *out) {
  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto ctx = std::move(*i);
    auto start = hr_clock::now();
    // TODO: wait alpha finish when alpha = filter

    auto result = cudaEventSynchronize(get_output_consumed());
    if (result != cudaSuccess) {
      LOG(QFATAL) << "cudaEventSynchronize Error: " << cudaGetErrorName(result);
    }
    if (ctx.post_scale != 1.0) {
      md_view<uint8_t, int32_t, 3> tmp;
      std::tie(tmp, ctx.out_memory) = scale_view(ctx.out_image.as_view(), ctx.post_scale);
      ctx.out_image = tmp.as_wuview();
    }

    std::string output = output_repr(ctx.output);
    auto err = save_image_png(std::move(ctx.output), ctx.out_image.as_view());
    if (err.empty()) {
      VLOG(1) << "Image " << output << " saved in " << elapsed(start) << "ms";

      auto [h, w, _] = ctx.out_image.shape;
      LOG(INFO) << "Image " << output << " (" << w << "x" << h << ") finished in " << elapsed(ctx.image_start)
                << "ms";
    }
    else {
      LOG(ERROR) << "Image " << output << " save failed: " << err;
    }

  }
}

void launch_pipeline(chan &in, chan *out) {
  ichan load_import, import_inference, inference_export, export_save;
  std::array<std::thread, 5> threads{
      std::thread(image_load_worker, std::ref(in), std::ref(load_import)),
      std::thread(pixel_import_worker, std::ref(load_import), std::ref(import_inference)),
      std::thread(inference_worker, std::ref(import_inference), std::ref(inference_export)),
      std::thread(pixel_export_worker, std::ref(inference_export), std::ref(export_save)),
      std::thread(image_save_worker, std::ref(export_save), out),
  };

  for (auto &t: threads) {
    t.join();
  }
}