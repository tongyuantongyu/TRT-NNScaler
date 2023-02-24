//
// Created by TYTY on 2023-02-12 012.
//

#include <cmath>
#include <array>
#include <iostream>

#include "nn-scaler.h"
#include "reformat/reformat.h"
#include "image_io.h"

#include "gflags/gflags.h"
#include "logging.h"
#include "libyuv/scale_argb.h"
#include "libyuv/scale_rgb.h"
#include "cuda_fp16.h"

//#include "reveal.h"

extern InferenceSession *session;

extern int using_io;

extern pixel_importer_cpu *importer_cpu;
extern pixel_exporter_cpu *exporter_cpu;

extern pixel_importer_gpu<float> *importer_gpu;
extern pixel_exporter_gpu<float> *exporter_gpu;

extern pixel_importer_gpu<half> *importer_gpu_fp16;
extern pixel_exporter_gpu<half> *exporter_gpu_fp16;

extern int32_t h_scale, w_scale;

struct WorkContextInternal {
  // filled by launcher
  std::filesystem::path output;

  // filled by image_load
  hr_clock::time_point image_start, tile_start;
  md_uview<const uint8_t, 3> in_image;
  mem_owner in_memory;  // hold by pixel_import

  // filled by pixel_import
  offset_t y, x, th, tw;
  bool h_beg, h_end, w_beg, w_end;
  bool has_alpha, is_alpha, is_end;
  std::promise<void> input_consumed;  // hold by inference

  // filled by inference
  std::promise<void> output_consumed;  // hold by pixel_export

  // alloc by image_load, filled by pixel_export
  md_uview<uint8_t, 3> out_image;
  mem_owner out_memory;  // hold by image_save
  std::future<void> alpha_filtered;  // TODO filter scale alpha
};

typedef channel<WorkContextInternal> ichan;

DEFINE_string(alpha, "nn", "alpha process mode: nn, filter, ignore");
DEFINE_validator(alpha, [](const char *flagname, const std::string &value) {
  if (value == "nn" || value == "ignore") {
    return true;
  }
  if (value == "filter") {
    std::cerr << "filter process mode is unimplemented." << std::endl;
    return false;
  }
  std::cerr << "Invalid value for --" << flagname << ": " << value << std::endl;
  return false;
});

DEFINE_double(pre_scale, 1.0, "scale ratio before NN super resolution.");
DEFINE_double(post_scale, 1.0, "scale ratio before NN super resolution.");

static void image_load_worker(chan &in, ichan &out) {
  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto c = std::move(*i);
    auto start = hr_clock::now();

    std::string err;
    auto img_ret = load_image(c.input, FLAGS_alpha == "ignore");

    if (get_if<1>(&img_ret)) {
      c.submitted.set_value("failed reading image: " + get<1>(img_ret));
      continue;
    }

    auto [in_shape, in_ptr] = std::move(get<0>(img_ret));
    md_view<uint8_t, 3> in_view{reinterpret_cast<uint8_t *>(in_ptr.get()), in_shape};
    if (FLAGS_pre_scale != 1.0) {
      auto [ho, wo, co] = in_view.shape;
      offset_t hs = round(FLAGS_pre_scale * ho);
      offset_t ws = round(FLAGS_pre_scale * wo);
      auto [scaled_view, scaled_ptr] = alloc_buffer<uint8_t>(hs, ws, co);
      if (co == 3) {
        libyuv::RGBScale(in_view.data,
                         in_view.at(0).size(),
                         wo,
                         ho,
                         scaled_view.data,
                         scaled_view.at(0).size(),
                         ws,
                         hs,
                         libyuv::kFilterBox);
      }
      else {
        libyuv::ARGBScale(in_view.data,
                          in_view.at(0).size(),
                          wo,
                          ho,
                          scaled_view.data,
                          scaled_view.at(0).size(),
                          ws,
                          hs,
                          libyuv::kFilterBox);
      }
      in_view = scaled_view;
      in_ptr.swap(scaled_ptr);
    }

    auto [h, w, ch] = in_view.shape;
    if (h < MinDimension || w < MinDimension) {
      LOG(WARNING) << "Skip too small image " << o(c.input) << " (" << w << "x" << h << ")";
    }

    VLOG(1) << "Image " << o(c.input) << " loaded in " << elapsed(start) << "ms, dimension: " << w << "x" << h;

    auto [out_view, out_ptr] = alloc_buffer<uint8_t>(h_scale * h, w_scale * w, ch);
    c.submitted.set_value("");

#ifndef NDEBUG
    memset(out_ptr.get(), 0, in_view.size() * h_scale * w_scale);
#endif

    out.put(WorkContextInternal{
        .output = c.output,

        .image_start = start,
        .in_image = in_view.as_uview(),
        .in_memory = std::move(in_ptr),

        .out_image = out_view.as_uview(),
        .out_memory = std::move(out_ptr),
    });
    VLOG(2) << "Image " << o(c.output) << " sent.";
  }

  out.close();
}

// v loaded image

template<std::integral I, typename Task>
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

DECLARE_int32(tile_width);
DECLARE_int32(tile_height);
DECLARE_int32(tile_pad);
DECLARE_int32(extend_grace);
DEFINE_int32(alignment, 1, "model input alignment requirement");

static offset_t align(offset_t n, size_t alignment) {
  n += alignment - 1;
  return n - (n % alignment);
}

static void pixel_import_worker(ichan &in, ichan &out) {
  bool nn_alpha = FLAGS_alpha == "nn";

  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto ctx = std::move(*i);

    auto [h, w, c] = ctx.in_image.shape;
    auto process_alpha = nn_alpha && c == 4;
    offset_t h_split = align(h, FLAGS_alignment), w_split = align(w, FLAGS_alignment);

    split_range<offset_t>(
        h_split, FLAGS_tile_height, FLAGS_tile_pad, FLAGS_extend_grace,
        [&, h = h, w = w](offset_t y, offset_t th, bool h_beg, bool h_end) {
          return split_range<offset_t>(
              w_split, FLAGS_tile_width, FLAGS_tile_pad, FLAGS_extend_grace,
              [&](offset_t x, offset_t tw, bool w_beg, bool w_end) -> bool {
                auto tile_start = hr_clock::now();

                auto input_tile = ctx.in_image.slice<0>(y, std::min(y + th, h)).slice<1>(x, std::min(x + tw, w));
                md_view<float, 3> input_tensor = {reinterpret_cast<float *>(session->input), {3, th, tw}};
                md_view<half, 3> input_tensor_fp16 = {reinterpret_cast<half *>(session->input), {3, th, tw}};

                bool first_tile = h_beg && w_beg;

                if (process_alpha) {
                  auto alpha_start = hr_clock::now();
                  std::string ret;
                  switch (using_io) {
                    case 0:
                      ret = importer_cpu->import_alpha(input_tensor, input_tile, session->stream); break;
                    case 1:
                      ret = importer_gpu->import_alpha<uint8_t>(input_tensor, input_tile, session->stream); break;
                    case 2:
                      ret = importer_gpu_fp16->import_alpha<uint8_t>(input_tensor_fp16, input_tile, session->stream); break;
                    default:
                      LOG(FATAL) << "Unknown IO mode.";
                  }

                  if (!ret.empty()) {
                    LOG(FATAL) << "Unexpected error importing pixel: " << ret;
                  }

                  WorkContextInternal tile_ctx = {
                      .tile_start = alpha_start,
                      .y = y, .x = x, .th = th, .tw = tw,
                      .h_beg = h_beg, .h_end = h_end, .w_beg = w_beg, .w_end = w_end,
                      .has_alpha = true, .is_alpha = true,
                      .out_image = ctx.out_image,
                  };

                  if (first_tile) {
                    tile_ctx.output = ctx.output;
                    tile_ctx.image_start = ctx.image_start;
                    tile_ctx.out_memory = std::move(ctx.out_memory);
                    first_tile = false;
                  }

                  VLOG(2) << "Tile "
                          << std::setw(4) << tw << 'x'
                          << std::setw(4) << th << '+'
                          << std::setw(4) << x << '+'
                          << std::setw(4) << y << " alpha imported in "
                          << elapsed(alpha_start) << "ms";

                  auto input_done = tile_ctx.input_consumed.get_future();
                  out.put(std::move(tile_ctx));
                  input_done.get();

                  VLOG(2) << "Tile "
                          << std::setw(4) << tw << 'x'
                          << std::setw(4) << th << '+'
                          << std::setw(4) << x << '+'
                          << std::setw(4) << y << " alpha input consumed in "
                          << elapsed(alpha_start) << "ms";

                }

                std::string ret;
                switch (using_io) {
                  case 0:
                    ret = importer_cpu->import_color(input_tensor, input_tile, session->stream); break;
                  case 1:
                    ret = importer_gpu->import_color<uint8_t>(input_tensor, input_tile, session->stream); break;
                  case 2:
                    ret = importer_gpu_fp16->import_color<uint8_t>(input_tensor_fp16, input_tile, session->stream); break;
                  default:
                    LOG(FATAL) << "Unknown IO mode.";
                }

                if (!ret.empty()) {
                  LOG(FATAL) << "Unexpected error importing pixel: " << ret;
                }

                WorkContextInternal tile_ctx{
                    .tile_start = tile_start,
                    .y = y, .x = x, .th = th, .tw = tw,
                    .h_beg = h_beg, .h_end = h_end, .w_beg = w_beg, .w_end = w_end,
                    .has_alpha = process_alpha, .is_alpha = false,
                    .out_image = ctx.out_image,
                };

                VLOG(2) << "Tile "
                        << std::setw(4) << tw << 'x'
                        << std::setw(4) << th << '+'
                        << std::setw(4) << x << '+'
                        << std::setw(4) << y << " imported in "
                        << elapsed(tile_start) << "ms";

                if (first_tile) {
                  tile_ctx.output = ctx.output;
                  tile_ctx.image_start = ctx.image_start;
                  tile_ctx.out_memory = std::move(ctx.out_memory);
                }

                if (h_end && w_end) {
                  tile_ctx.is_end = true;
                }

                auto input_done = tile_ctx.input_consumed.get_future();
                out.put(std::move(tile_ctx));
                input_done.get();

                VLOG(2) << "Tile "
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
  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto ctx = std::move(*i);

    session->config(1, ctx.th, ctx.tw);
    if (!session->inference()) {
      LOG(FATAL) << "CUDA error during inference: " << cudaGetErrorName(cudaGetLastError());
    }
    auto err = cudaEventSynchronize(session->input_consumed);
    if (err != cudaSuccess) {
      LOG(FATAL) << "CUDA Error: " << cudaGetErrorName(err);
    }
    ctx.input_consumed.set_value();

    auto output_done = ctx.output_consumed.get_future();
    out.put(std::move(ctx));
    output_done.get();
  }

  out.close();
}

// v pixel produced notice (with consumed promise)   ^ fulfill promise

static void pixel_export_worker(ichan &in, ichan &out) {
  std::filesystem::path output;
  hr_clock::time_point start;
  std::unique_ptr<uint8_t[]> in_memory, out_memory;

  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto ctx = std::move(*i);

    if (!ctx.output.empty()) {  // begin of image
      output.swap(ctx.output);
      start = ctx.image_start;
      out_memory = std::move(ctx.out_memory);
    }

    md_view<float, 3> output_tensor =
        {reinterpret_cast<float *>(session->output), {3, ctx.th * h_scale, ctx.tw * w_scale}};
    md_view<half, 3> output_tensor_fp16 =
        {reinterpret_cast<half *>(session->output), {3, ctx.th * h_scale, ctx.tw * w_scale}};
    pad_descriptor pad_desc{FLAGS_tile_pad * h_scale, ctx.h_beg, ctx.h_end, ctx.w_beg, ctx.w_end};
    auto [h, w, _] = ctx.out_image.shape;
    auto out_tile = ctx.out_image
        .slice<0>(h_scale * ctx.y, std::min(h_scale * (ctx.y + ctx.th), h))
        .slice<1>(w_scale * ctx.x, std::min(w_scale * (ctx.x + ctx.tw), w));

    std::string ret;
    if (ctx.is_alpha) {
      switch (using_io) {
        case 0:
          ret = exporter_cpu->fetch_alpha(output_tensor, session->stream); break;
        case 1:
          ret = exporter_gpu->fetch_alpha(output_tensor, session->stream); break;
        case 2:
          ret = exporter_gpu_fp16->fetch_alpha(output_tensor_fp16, session->stream); break;
        default:
          LOG(FATAL) << "Unknown IO mode.";
      }
    }
    else {
      switch (using_io) {
        case 0:
          ret = exporter_cpu->fetch_color(output_tensor, out_tile, pad_desc, session->stream); break;
        case 1:
          ret = exporter_gpu->fetch_color<uint8_t>(output_tensor, out_tile, pad_desc, session->stream); break;
        case 2:
          ret = exporter_gpu_fp16->fetch_color<uint8_t>(output_tensor_fp16, out_tile, pad_desc, session->stream); break;
        default:
          LOG(FATAL) << "Unknown IO mode.";
      }
    }
    if (!ret.empty()) {
      LOG(FATAL) << "Unexpected error fetching result pixel: " << ret;
    }

    ctx.output_consumed.set_value();

    VLOG(1) << "Tile "
            << std::setw(4) << ctx.tw << 'x'
            << std::setw(4) << ctx.th << '+'
            << std::setw(4) << ctx.x << '+'
            << std::setw(4) << ctx.y << " scale done in "
            << elapsed(ctx.tile_start) << "ms";



    if (ctx.is_end) {
      ctx.output.swap(output);
      ctx.image_start = start;
      ctx.out_memory = std::move(out_memory);
      out.put(std::move(ctx));
    }
  }

  out.close();
}

// v output image

static void image_save_worker(ichan &in) {
  while (true) {
    auto i = in.get();
    if (!i) {
      break;
    }

    auto ctx = std::move(*i);
    auto start = hr_clock::now();
    // TODO: wait alpha finish when alpha = filter

    if (FLAGS_post_scale != 1.0) {
      auto [ho, wo, co] = ctx.out_image.shape;
      offset_t hs = round(FLAGS_post_scale * ho);
      offset_t ws = round(FLAGS_post_scale * wo);
      auto [scaled_view, scaled_ptr] = alloc_buffer<uint8_t, 3>({hs, ws, co});
      if (co == 3) {
        libyuv::RGBScale(ctx.out_image.data,
                         ctx.out_image.at(0).size(),
                         wo,
                         ho,
                         scaled_view.data,
                         scaled_view.at(0).size(),
                         ws,
                         hs,
                         libyuv::kFilterBox);
      }
      else {
        libyuv::ARGBScale(ctx.out_image.data,
                          ctx.out_image.at(0).size(),
                          wo,
                          ho,
                          scaled_view.data,
                          scaled_view.at(0).size(),
                          ws,
                          hs,
                          libyuv::kFilterBox);
      }
      ctx.out_image = scaled_view;
      ctx.out_memory.swap(scaled_ptr);
    }

    auto err = save_image_png(ctx.output, ctx.out_image.as_view());
    VLOG(2) << "Image encoded in " << elapsed(start) << "ms";

    auto [h, w, _] = ctx.out_image.shape;
    LOG(INFO) << "Image " << o(ctx.output) << " (" << w << "x" << h << ") finished in " << elapsed(ctx.image_start)
              << "ms";
  }
}

void launch_pipeline(chan &in) {
  ichan load_import, import_inference, inference_export, export_save;
  std::array<std::thread, 5> threads{
      std::thread(image_load_worker, std::ref(in), std::ref(load_import)),
      std::thread(pixel_import_worker, std::ref(load_import), std::ref(import_inference)),
      std::thread(inference_worker, std::ref(import_inference), std::ref(inference_export)),
      std::thread(pixel_export_worker, std::ref(inference_export), std::ref(export_save)),
      std::thread(image_save_worker, std::ref(export_save)),
  };

  for (auto &t: threads) {
    t.join();
  }
}
