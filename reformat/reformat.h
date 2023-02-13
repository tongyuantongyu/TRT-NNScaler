#pragma once

#include <string>
#include <memory>
#include <concepts>

#include "cuda_runtime_api.h"

#include "md_view.h"
#include "reformat_cuda.h"

class pixel_importer_cpu {
  std::unique_ptr<float[]> buffer{};
  std::unique_ptr<float[]> buffer_alpha{};
  size_t max_size;

 public:
  explicit pixel_importer_cpu(size_t max_size, bool handle_alpha = true)
      : max_size(max_size), buffer(std::make_unique<float[]>(max_size * 3)) {
    if (handle_alpha) {
      buffer_alpha = std::make_unique<float[]>(max_size);
    }
  }

  template<std::unsigned_integral U>
  std::string import_color(md_view<float, 3> dst, md_uview<const U, 3> src, cudaStream_t stream, float quant = 0.0);

  std::string import_alpha(md_view<float, 3> dst, cudaStream_t stream);

};

struct pad_descriptor {
  offset_t pad;
  bool top, bottom, left, right;
};

class pixel_exporter_cpu_crop {
  std::unique_ptr<float[]> buffer{};
  std::unique_ptr<float[]> buffer_alpha{};
  size_t max_size;

 public:
  explicit pixel_exporter_cpu_crop(size_t max_size, bool handle_alpha = true)
      : max_size(max_size), buffer(std::make_unique<float[]>(max_size * 3)) {
    if (handle_alpha) {
      buffer_alpha = std::make_unique<float[]>(max_size);
    }
  }
  std::string fetch_color(md_view<const float, 3> src, cudaStream_t stream);
  std::string fetch_alpha(md_view<const float, 3> src, cudaStream_t stream);

  template<std::unsigned_integral U>
  std::string export_data(md_uview<U, 3> dst, pad_descriptor pad, float quant = 0.0);
};

template<typename F, size_t eSize = 2>
class pixel_importer_gpu {
  std::unique_ptr<uint8_t[]> buffer{};
  void *gpu_buffer;
  size_t max_size;

 public:
  explicit pixel_importer_gpu(size_t max_size);
  ~pixel_importer_gpu();

  explicit operator bool() {
    return gpu_buffer != nullptr;
  }

  template<std::unsigned_integral U, std::enable_if<sizeof(U) <= eSize, bool> = true>
  std::string import_pixel(md_view<F, 3> dst, md_uview<const U, 3> src, cudaStream_t stream, float quant = 0.0);

  template<std::unsigned_integral U, std::enable_if<sizeof(U) <= eSize, bool> = true>
  std::string import_pixel_gray(md_view<F, 3> dst, md_uview<const U, 3> src, cudaStream_t stream, float quant = 0.0);
};

// ----------
// Implementation

template<std::unsigned_integral U>
std::string pixel_importer_cpu::import_color(md_view<float, 3> dst,
                                             md_uview<const U, 3> src,
                                             cudaStream_t stream,
                                             float quant) {
  if (dst.shape.slice<1, 2>() != src.shape.template slice<0, 2>()) {
    return "dimension mismatch";
  }

  auto [h, w, c] = src.shape;

  if (h * w > max_size) {
    return "dimension too big";
  }

  if (quant == 0.0) {
    quant = 1.0 / float(std::numeric_limits<U>::max());
  }

  md_view<float, 3> tmp{buffer.get(), dst.shape};

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
      md_view<float, 2> tmp_alpha{buffer_alpha.get(), {h, w}};
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

  auto err = cudaMemcpyAsync(dst.data, tmp.data, dst.size() * 4, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  return "";
}

template<std::unsigned_integral U>
std::string pixel_exporter_cpu_crop::export_data(md_uview<U, 3> dst, pad_descriptor pad, float quant) {
  if (quant == 0.0) {
    quant = float(std::numeric_limits<U>::max());
  }

  auto [he, we, c] = dst.shape;
  md_uview<float, 3> tmp = md_view<float, 3>{buffer.get(), {c, he, we}};
  md_uview<float, 2> tmp_alpha = md_view<float, 2>{buffer_alpha.get(), {he, we}};

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

template<typename F, size_t eSize>
pixel_importer_gpu<F, eSize>::pixel_importer_gpu(size_t max_size)
    : max_size(max_size), buffer(std::make_unique<uint8_t[]>(max_size * 4 * eSize)), gpu_buffer{} {
  cudaMalloc(&gpu_buffer, max_size * 4 * sizeof(F));
}

template<typename F, size_t eSize>
pixel_importer_gpu<F, eSize>::~pixel_importer_gpu() {
  if (gpu_buffer) {
    cudaFree(gpu_buffer);
  }
}

template<typename F, size_t eSize>
template<std::unsigned_integral U, std::enable_if<sizeof(U) <= eSize, bool>>
std::string pixel_importer_gpu<F, eSize>::import_pixel(md_view<F, 3> dst,
                                                       md_uview<const U, 3> src,
                                                       cudaStream_t stream,
                                                       float quant) {
  return "unimplemented";
}

template<typename F, size_t eSize>
template<std::unsigned_integral U, std::enable_if<sizeof(U) <= eSize, bool>>
std::string pixel_importer_gpu<F, eSize>::import_pixel_gray(md_view<F, 3> dst,
                                                            md_uview<const U, 3> src,
                                                            cudaStream_t stream,
                                                            float quant) {
  return "unimplemented";
}
