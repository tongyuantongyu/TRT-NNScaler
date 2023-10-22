#pragma once

#include <cmath>
#include <string>
#include <memory>
#include <concepts>

#include "cuda_runtime_api.h"

#include "md_view.h"
#include "reformat_cuda.h"

//#include "reveal.h"

class pixel_importer_cpu {
  std::unique_ptr<float[]> buffer{};
  std::unique_ptr<float[]> buffer_alpha{};
  bool buffer_filled;
  size_t max_size;

  template<typename U>
  std::string import_pixel(shape_t<2> dst_shape, md_uview<const U, 3> src, cudaStream_t stream, float quant = 0.0);

 public:
  explicit pixel_importer_cpu(size_t max_size, bool handle_alpha = true)
      : max_size(max_size), buffer(std::make_unique<float[]>(max_size * 3)), buffer_filled(false) {
    if (handle_alpha) {
      buffer_alpha = std::make_unique<float[]>(max_size);
    }
  }

  template<typename U>
  std::string import_color(md_view<float, 3> dst, md_uview<const U, 3> src, cudaStream_t stream, float quant = 0.0);

  template<typename U>
  std::string import_alpha(md_view<float, 3> dst, md_uview<const U, 3> src, cudaStream_t stream, float quant = 0.0);
};

struct pad_descriptor {
  offset_t pad;
  bool top, bottom, left, right;
};

class pixel_exporter_cpu {
  std::unique_ptr<float[]> buffer{};
  std::unique_ptr<float[]> buffer_alpha{};
  bool alpha_filled{};
  shape_t<3> current_buffer_shape;
  size_t max_size;

 public:
  explicit pixel_exporter_cpu(size_t max_size, bool handle_alpha = true)
      : max_size(max_size), buffer(std::make_unique<float[]>(max_size * 3)) {
    if (handle_alpha) {
      buffer_alpha = std::make_unique<float[]>(max_size);
    }
  }

  template<typename U>
  std::string fetch_color(md_view<const float, 3> src,
                          md_uview<U, 3> dst,
                          pad_descriptor pad,
                          cudaStream_t stream,
                          float quant = 0.0);

  template<typename T = void>
  std::string fetch_alpha(md_view<const float, 3> src, cudaStream_t stream);
};

// ----------
// Implementation

template<typename U>
std::string pixel_importer_cpu::import_pixel(shape_t<2> dst_shape,
                                             md_uview<const U, 3> src,
                                             cudaStream_t stream,
                                             float quant) {
  auto [h, w, c] = src.shape;
  auto [dh, dw] = dst_shape;

  if (dh * dw > max_size) {
    return "dimension too big";
  }

  if (h > dh || w > dw) {
    return "incompatible dimension";
  }

  if (quant == 0.0) {
    quant = 1.0 / float(std::numeric_limits<U>::max());
  }

  md_view<float, 3> tmp{buffer.get(), {3, dh, dw}};
  md_view<float, 2> tmp_alpha{buffer_alpha.get(), {h, w}};

  if (c == 3) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t x = 0; x < w; ++x) {
        tmp.at(0, y, x) = static_cast<float>(src.at(y, x, 2)) * quant;
        tmp.at(1, y, x) = static_cast<float>(src.at(y, x, 1)) * quant;
        tmp.at(2, y, x) = static_cast<float>(src.at(y, x, 0)) * quant;
      }
    }
  }
  else if (c == 4) {
    if (!buffer_alpha) {
      for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
          tmp.at(0, y, x) = static_cast<float>(src.at(y, x, 2)) * quant;
          tmp.at(1, y, x) = static_cast<float>(src.at(y, x, 1)) * quant;
          tmp.at(2, y, x) = static_cast<float>(src.at(y, x, 0)) * quant;
        }
      }
    }
    else {
      for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
          tmp.at(0, y, x) = static_cast<float>(src.at(y, x, 2)) * quant;
          tmp.at(1, y, x) = static_cast<float>(src.at(y, x, 1)) * quant;
          tmp.at(2, y, x) = static_cast<float>(src.at(y, x, 0)) * quant;
          tmp_alpha.at(y, x) = static_cast<float>(src.at(y, x, 3)) * quant;
        }
      }
    }
  }
  else {
    assert(false);
  }

  for (size_t y = h; y < dh; ++y) {
    for (size_t x = 0; x < w; ++x) {
      tmp.at(0, y, x) = tmp.at(0, h - 1, x);
      tmp.at(1, y, x) = tmp.at(1, h - 1, x);
      tmp.at(2, y, x) = tmp.at(2, h - 1, x);
      if (c == 4 && buffer_alpha) {
        tmp_alpha.at(y, x) = tmp_alpha.at(h - 1, x);
      }
    }
  }

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = w; x < dw; ++x) {
      tmp.at(0, y, x) = tmp.at(0, y, w - 1);
      tmp.at(1, y, x) = tmp.at(1, y, w - 1);
      tmp.at(2, y, x) = tmp.at(2, y, w - 1);
      if (c == 4 && buffer_alpha) {
        tmp_alpha.at(y, x) = tmp_alpha.at(y, w - 1);
      }
    }
  }

  for (size_t y = h; y < dh; ++y) {
    for (size_t x = w; x < dw; ++x) {
      tmp.at(0, y, x) = tmp.at(0, h - 1, w - 1);
      tmp.at(1, y, x) = tmp.at(1, h - 1, w - 1);
      tmp.at(2, y, x) = tmp.at(2, h - 1, w - 1);
      if (c == 4 && buffer_alpha) {
        tmp_alpha.at(y, x) = tmp_alpha.at(h - 1, w - 1);
      }
    }
  }

  buffer_filled = true;
  return "";
}

template<typename U>
std::string pixel_importer_cpu::import_color(md_view<float, 3> dst,
                                             md_uview<const U, 3> src,
                                             cudaStream_t stream,
                                             float quant) {
  if (!buffer_filled) {
    auto err = import_pixel(dst.shape.slice<1, 2>(), src, stream, quant);
    if (!err.empty()) {
      return err;
    }
  }

  auto [dc, dh, dw] = dst.shape;

  md_view<float, 3> tmp{buffer.get(), {3, dh, dw}};

  auto err = cudaMemcpyAsync(dst.data, tmp.data, dst.size() * 4, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  buffer_filled = false;

  return "";
}

template<typename U>
std::string pixel_importer_cpu::import_alpha(md_view<float, 3> dst,
                                             md_uview<const U, 3> src,
                                             cudaStream_t stream,
                                             float quant) {
  if (!buffer_filled) {
    auto err = import_pixel(dst.shape.slice<1, 2>(), src, stream, quant);
    if (!err.empty()) {
      return err;
    }
  }

  auto [_, h, w] = dst.shape;

  if (h * w > max_size) {
    return "dimension too big";
  }

  auto err = cudaMemcpyAsync(dst.data, buffer_alpha.get(), h * w * 4, cudaMemcpyHostToDevice, stream);
  err = err ? err : cudaMemcpyAsync(dst.at(1).data, dst.at(0).data, h * w * 4, cudaMemcpyDeviceToDevice, stream);
  err = err ? err : cudaMemcpyAsync(dst.at(2).data, dst.at(0).data, h * w * 4, cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  return "";
}


template<typename T>
std::string pixel_exporter_cpu::fetch_alpha(md_view<const float, 3> src, cudaStream_t stream) {
  auto [c, h, w] = src.shape;

  if (h * w > max_size) {
    return "dimension too big";
  }

  auto err = cudaMemcpyAsync(buffer_alpha.get(), src.data, h * w * 4, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  current_buffer_shape = src.shape;
  alpha_filled = true;
  return "";
}

template<typename U>
std::string pixel_exporter_cpu::fetch_color(md_view<const float, 3> src,
                                            md_uview<U, 3> dst,
                                            pad_descriptor pad,
                                            cudaStream_t stream,
                                            float quant) {
  auto [_1, hs, ws] = src.shape;

  if (hs * ws > max_size) {
    return "dimension too big";
  }

  if (alpha_filled && current_buffer_shape != src.shape) {
    return "incompatible color buffer shape";
  }
  alpha_filled = false;

  auto err = cudaMemcpyAsync(buffer.get(), src.data, src.size() * 4, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  if (quant == 0.0) {
    quant = float(std::numeric_limits<U>::max());
  }

  auto [he, we, c] = dst.shape;
  if (he > hs || we > ws) {
    return "incompatible dimension";
  }

  md_uview<float, 3> tmp = md_view<float, 3>{buffer.get(), src.shape};
  md_uview<float, 2> tmp_alpha = md_view<float, 2>{buffer_alpha.get(), src.shape.slice<1, 2>()};

  offset_t shrink = pad.pad / 2;
  offset_t hb = pad.top ? 0 : shrink;
  offset_t wb = pad.left ? 0 : shrink;
  he -= pad.bottom ? 0 : shrink;
  we -= pad.right ? 0 : shrink;

  dst = dst.template slice<0>(hb, he).
      template slice<1>(wb, we);
  tmp = tmp.template slice<1>(hb, he).
      template slice<2>(wb, we);
  tmp_alpha = tmp_alpha.template slice<0>(hb, he).
      template slice<1>(wb, we);

  auto h = he - hb, w = we - wb;

  auto clamp = [](float v) {
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    return v;
  };

  if (c == 3) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t x = 0; x < w; ++x) {
        assert(dst.at(y, x, 0) == 0);
        assert(dst.at(y, x, 1) == 0);
        assert(dst.at(y, x, 2) == 0);
        dst.at(y, x, 0) = static_cast<U>(roundf(clamp(tmp.at(0, y, x)) * quant));
        dst.at(y, x, 1) = static_cast<U>(roundf(clamp(tmp.at(1, y, x)) * quant));
        dst.at(y, x, 2) = static_cast<U>(roundf(clamp(tmp.at(2, y, x)) * quant));
      }
    }
  }
  else if (c == 4) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t x = 0; x < w; ++x) {
        assert(dst.at(y, x, 0) == 0);
        assert(dst.at(y, x, 1) == 0);
        assert(dst.at(y, x, 2) == 0);
        assert(dst.at(y, x, 3) == 0);
        auto alpha = clamp(tmp_alpha.at(y, x));
        if (alpha * quant < 0.5) {
          dst.at(y, x, 0) = 0;
          dst.at(y, x, 1) = 0;
          dst.at(y, x, 2) = 0;
          dst.at(y, x, 3) = 0;
        }
        else {
          dst.at(y, x, 0) = static_cast<U>(roundf(clamp(tmp.at(0, y, x) / alpha) * quant));
          dst.at(y, x, 1) = static_cast<U>(roundf(clamp(tmp.at(1, y, x) / alpha) * quant));
          dst.at(y, x, 2) = static_cast<U>(roundf(clamp(tmp.at(2, y, x) / alpha) * quant));
          dst.at(y, x, 3) = static_cast<U>(roundf(alpha * quant));
        }
      }
    }
  }
  else {
    assert(false);
  }

  return "";
}

// -----------------------------------------------------------------------------
// GPU part

template<typename F, size_t eSize = 2>
class pixel_importer_gpu {
  std::unique_ptr<uint8_t[]> buffer{};
  void *gpu_buffer;
  void *gpu_buffer_color;
  bool buffer_filled;
  size_t max_size;

  template<typename U, std::enable_if_t<sizeof(U) <= eSize, bool> = true>
  std::string import_pixel(md_view<F, 3> dst,
                           md_view<F, 2> dst_alpha,
                           md_uview<const U, 3> src,
                           cudaStream_t stream,
                           float quant);

 public:
  explicit pixel_importer_gpu(size_t max_size, bool handle_alpha = true)
      : max_size(max_size),
        buffer(std::make_unique<uint8_t[]>(max_size * eSize * (handle_alpha ? 4 : 3))),
        gpu_buffer{},
        gpu_buffer_color{},
        buffer_filled{false} {
    cudaMalloc(&gpu_buffer, max_size * 3 * sizeof(F));
    if (handle_alpha) {
      cudaMalloc(&gpu_buffer_color, max_size * 3 * sizeof(F));
    }
  }

  ~pixel_importer_gpu() {
    if (gpu_buffer) {
      cudaFree(gpu_buffer);
    }
    if (gpu_buffer_color) {
      cudaFree(gpu_buffer_color);
    }
  }

  explicit operator bool() {
    return gpu_buffer != nullptr;
  }

  template<typename U>
  std::string import_color(md_view<F, 3> dst, md_uview<const U, 3> src, cudaStream_t stream, float quant = 0.0);

  template<typename U>
  std::string import_alpha(md_view<F, 3> dst, md_uview<const U, 3> src, cudaStream_t stream, float quant = 0.0);
};

template<typename F, size_t eSize = 1>
class pixel_exporter_gpu {
  std::unique_ptr<uint8_t[]> buffer{};
  void *gpu_buffer;
  void *gpu_buffer_alpha;
  bool alpha_filled{};
  shape_t<3> current_buffer_shape;
  size_t max_size;

 public:
  explicit pixel_exporter_gpu(size_t max_size, bool handle_alpha = true)
      : max_size(max_size), buffer(std::make_unique<uint8_t[]>(max_size * eSize * (handle_alpha ? 4 : 3))) {
    cudaMalloc(&gpu_buffer, max_size * (handle_alpha ? 4 : 3) * sizeof(F));
    if (handle_alpha) {
      cudaMalloc(&gpu_buffer_alpha, max_size * sizeof(F));
    }
  }

  ~pixel_exporter_gpu() {
    if (gpu_buffer) {
      cudaFree(gpu_buffer);
    }
  }

  template<typename U, std::enable_if_t<sizeof(U) <= eSize, bool> = true>
  std::string fetch_color(md_view<const F, 3> src,
                          md_uview<U, 3> dst,
                          pad_descriptor pad,
                          cudaStream_t stream,
                          float quant = 0.0);
  std::string fetch_alpha(md_view<const F, 3> src, cudaStream_t stream);
};

// ----------
// Implementation

template<typename F, size_t eSize>
template<typename U, std::enable_if_t<sizeof(U) <= eSize, bool>>
std::string pixel_importer_gpu<F, eSize>::import_pixel(md_view<F, 3> dst,
                                                       md_view<F, 2> dst_alpha,
                                                       md_uview<const U, 3> src,
                                                       cudaStream_t stream,
                                                       float quant) {
  auto [h, w, c] = src.shape;
  auto [dc, dh, dw] = dst.shape;

  if (dh * dw > max_size) {
    return "dimension too big";
  }

  if (h > dh || w > dw) {
    return "incompatible dimension";
  }

  if (quant == 0.0) {
    quant = 1.0 / float(std::numeric_limits<U>::max());
  }

  md_view<U, 3> tmp{reinterpret_cast<U *>(buffer.get()), src.shape};
  copy(to_uview(tmp), src);

  md_view<U, 3> gpu_tmp{reinterpret_cast<U *>(gpu_buffer), src.shape};
  auto err = cudaMemcpyAsync(gpu_tmp.data, tmp.data, tmp.size() * sizeof(U), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  import_pixel_cuda(dst, dst_alpha, md_view<const U, 3>(gpu_tmp), quant, 0.0, stream);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  buffer_filled = true;
  return "";
}

template<typename F, size_t eSize>
template<typename U>
std::string pixel_importer_gpu<F, eSize>::import_alpha(md_view<F, 3> dst,
                                                       md_uview<const U, 3> src,
                                                       cudaStream_t stream,
                                                       float quant) {
  md_view<F, 3> dst_color{reinterpret_cast<F *>(gpu_buffer_color), dst.shape};
  auto err = import_pixel(dst_color, dst.at(0), src, stream, quant);
  if (!err.empty()) {
    return err;
  }

  auto [_, h, w] = dst.shape;
  auto err2 = cudaMemcpyAsync(dst.at(1).data, dst.at(0).data, h * w * sizeof(F), cudaMemcpyDeviceToDevice, stream);
  err2 = err2 ? err2 : cudaMemcpyAsync(dst.at(2).data, dst.at(0).data, h * w * sizeof(F), cudaMemcpyDeviceToDevice, stream);
  if (err2 != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err2);
  }

  return "";
}

template<typename F, size_t eSize>
template<typename U>
std::string pixel_importer_gpu<F, eSize>::import_color(md_view<F, 3> dst,
                                                       md_uview<const U, 3> src,
                                                       cudaStream_t stream,
                                                       float quant) {
  if (buffer_filled) {
    auto err = cudaMemcpyAsync(dst.data, gpu_buffer_color, dst.size() * sizeof(F), cudaMemcpyDeviceToDevice, stream);;
    if (err != cudaSuccess) {
      return std::string("CUDA error: ") + cudaGetErrorName(err);
    }

    buffer_filled = false;
    return "";
  }
  else {
    auto err = import_pixel(dst, {nullptr}, src, stream, quant);
    buffer_filled = false;
    return err;
  }
}

template<typename F, size_t eSize>
std::string pixel_exporter_gpu<F, eSize>::fetch_alpha(md_view<const F, 3> src, cudaStream_t stream) {
  auto [c, h, w] = src.shape;

  if (h * w > max_size) {
    return "dimension too big";
  }

  auto err = cudaMemcpyAsync(gpu_buffer_alpha, src.data, h * w * sizeof(F), cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  current_buffer_shape = src.shape;
  alpha_filled = true;
  return "";
}

template<typename F, size_t eSize>
template<typename U, std::enable_if_t<sizeof(U) <= eSize, bool>>
std::string pixel_exporter_gpu<F, eSize>::fetch_color(md_view<const F, 3> src,
                                                      md_uview<U, 3> dst,
                                                      pad_descriptor pad,
                                                      cudaStream_t stream,
                                                      float quant) {
  auto [_1, hs, ws] = src.shape;

  if (hs * ws > max_size) {
    return "dimension too big";
  }

  if (alpha_filled && current_buffer_shape != src.shape) {
    return "incompatible color buffer shape";
  }

  if (quant == 0.0) {
    quant = float(std::numeric_limits<U>::max());
  }

  auto [he, we, c] = dst.shape;
  if (he > hs || we > ws) {
    return "incompatible dimension";
  }

  md_uview<const F, 3> usrc = src;
  md_uview<const F, 2> src_alpha = md_view<const F, 2>{
      static_cast<const F *>(gpu_buffer_alpha),
      src.shape.template slice<1, 2>()
  };

  offset_t shrink = pad.pad / 2;
  offset_t hb = pad.top ? 0 : shrink;
  offset_t wb = pad.left ? 0 : shrink;
  he -= pad.bottom ? 0 : shrink;
  we -= pad.right ? 0 : shrink;

  dst = dst.template slice<0>(hb, he).
      template slice<1>(wb, we);
  usrc = usrc.template slice<1>(hb, he).
      template slice<2>(wb, we);
  src_alpha = src_alpha.template slice<0>(hb, he).
      template slice<1>(wb, we);

  if (!alpha_filled) {
    src_alpha.data = nullptr;
  }
  alpha_filled = false;

  md_view<U, 3> gpu_tmp{static_cast<U *>(gpu_buffer), dst.shape};
  md_view<U, 3> tmp{reinterpret_cast<U *>(buffer.get()), dst.shape};

  export_pixel_cuda(gpu_tmp, usrc, src_alpha, quant, 0, stream);

  auto err = cudaMemcpyAsync(tmp.data, gpu_tmp.data, gpu_tmp.size() * sizeof(U), cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  copy(dst, tmp.as_uview());

  return "";
}
