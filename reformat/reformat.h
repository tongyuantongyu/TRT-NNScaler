#pragma once

#include <cmath>
#include <string>
#include <memory>
#include <concepts>

#include "cuda_runtime_api.h"

#include "logging.h"
#include "md_view.h"
#include "reformat_cuda.h"

//#include "reveal.h"

struct pad_descriptor {
  int32_t pad;
  bool top, bottom, left, right;
};

struct pixel_io_base {
  void *cuda_buffer{};
  size_t max_size;
  size_t el_size;

  explicit pixel_io_base(size_t max_size, size_t el_size) : max_size(max_size), el_size(el_size) {
    auto err = cudaMalloc(&cuda_buffer, max_size * el_size);
    if (err != cudaSuccess) {
      LOG(QFATAL) << "Failed allocating import buffer: " << cudaGetErrorString(err);
    }
  }

  ~pixel_io_base() {
    if (cuda_buffer) {
      cudaFree(cuda_buffer);
    }
  }
};

template<typename F>
struct pixel_importer_gpu : pixel_io_base {
  explicit pixel_importer_gpu(size_t max_size, size_t el_size) : pixel_io_base(max_size, el_size) {}

  template<typename U>
  std::string import_pixel(patch<F> dst,
                           patch<const U> src,
                           bool is_alpha,
                           cudaStream_t stream,
                           float quant = 0.0);


};

template<typename F>
struct pixel_exporter_gpu : pixel_io_base {
  explicit pixel_exporter_gpu(size_t max_size, size_t el_size) : pixel_io_base(max_size, el_size) {}

  template<typename U>
  std::string export_pixel(patch<const F> src,
                           patch<U> dst,
                           bool is_alpha,
                           pad_descriptor pad,
                           cudaStream_t stream,
                           float quant = 0.0);
};

// ----------
// Implementation

template<typename F>
template<typename U>
std::string pixel_importer_gpu<F>::import_pixel(patch<F> dst,
                                                patch<const U> src,
                                                bool is_alpha,
                                                cudaStream_t stream,
                                                float quant) {
  auto [h, w, c] = src.shape;
  auto [dc, dh, dw] = dst.shape;

  if (dh * dw > max_size) {
    return "dimension too big";
  }

  if (sizeof(U) > el_size) {
    return "element size too big";
  }

  if (h > dh || w > dw) {
    return "incompatible dimension";
  }

  auto tmp_stride = src.shape.template stride<int64_t>();
  auto result = cudaMemcpy2DAsync(cuda_buffer, tmp_stride[0] * sizeof(U),//
                                  src.data, src.stride[0] * sizeof(U),//
                                  tmp_stride[0] * sizeof(U), h, cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    return std::string("Failed copying into buffer: ") + cudaGetErrorString(result);
  }
  auto tmp = md_view{static_cast<const U *>(cuda_buffer), src.shape};

  if (quant == 0.0) {
    quant = 1.0 / float(std::numeric_limits<U>::max());
  }

  import_pixel_cuda(dst, tmp.as_wuview(), quant, 0.0, is_alpha, stream);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  return "";
}

template<typename F>
template<typename U>
std::string pixel_exporter_gpu<F>::export_pixel(patch<const F> src,
                                                patch<U> dst,
                                                bool is_alpha,
                                                pad_descriptor pad,
                                                cudaStream_t stream,
                                                float quant) {
  auto [_1, hs, ws] = src.shape;

  if (src.shape.count() > max_size) {
    return "dimension too big";
  }

  if (sizeof(U) > el_size) {
    return "element size too big";
  }

  if (quant == 0.0) {
    quant = static_cast<float>(std::numeric_limits<U>::max());
  }

  auto [he, we, c] = dst.shape;
  if (he > hs || we > ws) {
    return "incompatible dimension";
  }

  int32_t shrink = pad.pad / 2;
  int32_t hb = pad.top ? 0 : shrink;
  int32_t wb = pad.left ? 0 : shrink;
  he -= pad.bottom ? 0 : shrink;
  we -= pad.right ? 0 : shrink;

  dst = dst.template slice<0>(hb, he).
            template slice<1>(wb, we);
  src = src.template slice<1>(hb, he).
            template slice<2>(wb, we);

  auto tmp = md_view{static_cast<U *>(cuda_buffer), dst.shape};
  export_pixel_cuda(tmp.as_wuview(), src, quant, 0, 0, quant, is_alpha, stream);

  auto tmp_stride = dst.shape.template stride<int64_t>();
  auto result = cudaMemcpy2DAsync(dst.data, dst.stride[0] * sizeof(U),//
                                  tmp.data, tmp_stride[0] * sizeof(U),//
                                  tmp_stride[0] * sizeof(U), tmp.shape[0], cudaMemcpyDeviceToHost, stream);
  if (result != cudaSuccess) {
    return std::string("Failed copying into buffer: ") + cudaGetErrorString(result);
  }

  cudaStreamSynchronize(stream);

  return "";
}