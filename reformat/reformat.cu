#include "reformat_cuda.h"

#include <cuda_fp16.h>

static half __device__ round(half f) {
  const half v0_5{0.5f};
  return hfloor(f + v0_5);
}

template<class F, class U>
static void __global__ import_color_kernel(patch<F> dst, patch<const U> src, F a, F b) {
  uint32_t dst_x = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t dst_y = threadIdx.y + blockDim.y * blockIdx.y;

  auto [_1, dst_h, dst_w] = dst.shape;
  if (dst_x >= dst_w || dst_y >= dst_h) {
    return;
  }

  auto [src_h, src_w, _2] = src.shape;
  uint32_t src_x = dst_x >= src_w ? src_w - 1 : dst_x;
  uint32_t src_y = dst_y >= src_h ? src_h - 1 : dst_y;

  dst.at(0, dst_y, dst_x) = a * static_cast<F>(src.at(src_y, src_x, 2)) + b;
  dst.at(1, dst_y, dst_x) = a * static_cast<F>(src.at(src_y, src_x, 1)) + b;
  dst.at(2, dst_y, dst_x) = a * static_cast<F>(src.at(src_y, src_x, 0)) + b;
}

template<class F, class U>
static void __global__ import_alpha_kernel(patch<F> dst, patch<const U> src, F a, F b) {
  uint32_t dst_x = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t dst_y = threadIdx.y + blockDim.y * blockIdx.y;

  auto [_1, dst_h, dst_w] = dst.shape;
  if (dst_x >= dst_w || dst_y >= dst_h) {
    return;
  }

  auto [src_h, src_w, _2] = src.shape;
  uint32_t src_x = dst_x >= src_w ? src_w - 1 : dst_x;
  uint32_t src_y = dst_y >= src_h ? src_h - 1 : dst_y;

  auto alpha = a * static_cast<F>(src.at(src_y, src_x, 3)) + b;
  dst.at(0, dst_y, dst_x) = alpha;
  dst.at(1, dst_y, dst_x) = alpha;
  dst.at(2, dst_y, dst_x) = alpha;
}

template<class U, class F>
static U __device__ cast(F v, F l, F h) {
  if (v < l)
    v = l;
  if (v > h)
    v = h;

  if constexpr (sizeof(U) == 1) {
    return static_cast<U>(static_cast<int16_t>(round(v)));
  }
  else {
    return static_cast<U>(round(v));
  }
}

template<class F, class U>
static void __global__ export_color_kernel(patch<U> dst, patch<const F> src, F a, F b, F l, F h) {
  uint32_t dst_x = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t dst_y = threadIdx.y + blockDim.y * blockIdx.y;

  auto [dst_h, dst_w, _1] = dst.shape;
  if (dst_x >= dst_w || dst_y >= dst_h) {
    return;
  }

  dst.at(dst_y, dst_x, 0) = cast<U>(a * src.at(0, dst_y, dst_x) + b, l, h);
  dst.at(dst_y, dst_x, 1) = cast<U>(a * src.at(1, dst_y, dst_x) + b, l, h);
  dst.at(dst_y, dst_x, 2) = cast<U>(a * src.at(2, dst_y, dst_x) + b, l, h);
}

template<class F, class U>
static void __global__ export_alpha_kernel(patch<U> dst,
                                           patch<const F, 2> src_alpha,
                                           F a,
                                           F b,
                                           F l,
                                           F h) {
  uint32_t dst_x = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t dst_y = threadIdx.y + blockDim.y * blockIdx.y;

  auto [dst_h, dst_w, _1] = dst.shape;
  if (dst_x >= dst_w || dst_y >= dst_h) {
    return;
  }

  dst.at(dst_y, dst_x, 3) = cast<U>(a * src_alpha.at(dst_y, dst_x) + b, l, h);
}

template<class F, class U>
void import_pixel_cuda(patch<F> dst,
                       patch<const U> src,
                       float a,
                       float b,
                       bool is_alpha,
                       cudaStream_t stream) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  auto [c, dst_h, dst_w] = dst.shape;
  dimGrid.x = (dst_w + 31) >> 5;
  dimGrid.y = (dst_h + 31) >> 5;

  if (is_alpha) {
    assert(src.shape[2] == 4);
    import_alpha_kernel<<<dimGrid, dimBlock, 0, stream>>>(dst, src, F(a), F(b));
  }
  else {
    import_color_kernel<<<dimGrid, dimBlock, 0, stream>>>(dst, src, F(a), F(b));
  }
}

template<class F, class U>
void export_pixel_cuda(patch<U> dst,
                       patch<const F> src,
                       float a,
                       float b,
                       float l,
                       float h,
                       bool is_alpha,
                       cudaStream_t stream) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  auto [dst_h, dst_w, _] = dst.shape;
  dimGrid.x = (dst_w + 31) >> 5;
  dimGrid.y = (dst_h + 31) >> 5;

  if (is_alpha) {
    assert(dst.shape[2] == 4);
    export_alpha_kernel<<<dimGrid, dimBlock, 0, stream>>>(dst, src.at(0), F(a), F(b), F(l), F(h));
  }
  else {
    export_color_kernel<<<dimGrid, dimBlock, 0, stream>>>(dst, src, F(a), F(b), F(l), F(h));
  }
}

template void import_pixel_cuda<float, uint8_t>(patch<float> dst,
                                                patch<const uint8_t> src,
                                                float a,
                                                float b,
                                                bool is_alpha,
                                                cudaStream_t stream);
template void import_pixel_cuda<half, uint8_t>(patch<half> dst,
                                               patch<const uint8_t> src,
                                               float a,
                                               float b,
                                                bool is_alpha,
                                               cudaStream_t stream);
template void export_pixel_cuda<float, uint8_t>(patch<uint8_t> dst,
                                                patch<const float> src,
                                                float a,
                                                float b,
                                                float l,
                                                float h,
                                                bool is_alpha,
                                                cudaStream_t stream);
template void export_pixel_cuda<half, uint8_t>(patch<uint8_t> dst,
                                               patch<const half> src,
                                               float a,
                                               float b,
                                                float l,
                                                float h,
                                                bool is_alpha,
                                               cudaStream_t stream);