#include "reformat_cuda.h"
#include <cuda_fp16.h>

static half __device__ round(half f) {
  const half v0_5{0.5f};
  return hfloor(f + v0_5);
}

template<class F, class U>
static void __global__ import_opaque_kernel(md_view<F, int32_t, 3> dst, md_view<const U, int32_t, 3> src, F a, F b) {
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
static void __global__ import_alpha_kernel(md_view<F, int32_t, 3> dst,
                                           md_view<F, int32_t, 2> dst_alpha,
                                           md_view<const U, int32_t, 3> src,
                                           F a,
                                           F b) {
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
  dst_alpha.at(dst_y, dst_x) = a * static_cast<F>(src.at(src_y, src_x, 3)) + b;
}

template<class F>
static F __device__ clamp(F v) {
  if (v < F{0.0f}) v = F{0.0f};
  if (v > F{1.0f}) v = F{1.0f};
  return v;
}

template<class U, class F>
static U __device__ cast(F v) {
  if constexpr (sizeof(U) == 1) {
    return static_cast<U>(static_cast<int16_t>(round(v)));
  }
  else {
    return static_cast<U>(round(v));
  }
}

template<class F, class U>
static void __global__ export_opaque_kernel(md_view<U, int32_t, 3> dst, md_uview<const F, int32_t, 3> src, F a, F b) {
  uint32_t dst_x = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t dst_y = threadIdx.y + blockDim.y * blockIdx.y;

  auto [dst_h, dst_w, _1] = dst.shape;
  if (dst_x >= dst_w || dst_y >= dst_h) {
    return;
  }

  dst.at(dst_y, dst_x, 0) = cast<U>(a * clamp(src.at(0, dst_y, dst_x)) + b);
  dst.at(dst_y, dst_x, 1) = cast<U>(a * clamp(src.at(1, dst_y, dst_x)) + b);
  dst.at(dst_y, dst_x, 2) = cast<U>(a * clamp(src.at(2, dst_y, dst_x)) + b);
}

template<class F, class U>
static void __global__ export_alpha_kernel(md_view<U, int32_t, 3> dst,
                                           md_uview<const F, int32_t, 3> src,
                                           md_uview<const F, int32_t, 2> src_alpha,
                                           F a,
                                           F b) {
  uint32_t dst_x = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t dst_y = threadIdx.y + blockDim.y * blockIdx.y;

  auto [dst_h, dst_w, _1] = dst.shape;
  if (dst_x >= dst_w || dst_y >= dst_h) {
    return;
  }

  F alpha = clamp(src_alpha.at(dst_y, dst_x));
  if (a * alpha < F{0.5f}) {
    dst.at(dst_y, dst_x, 0) = cast<U>(b);
    dst.at(dst_y, dst_x, 1) = cast<U>(b);
    dst.at(dst_y, dst_x, 2) = cast<U>(b);
    dst.at(dst_y, dst_x, 3) = cast<U>(b);
  }
  else {
    dst.at(dst_y, dst_x, 0) = cast<U>(a * clamp(src.at(0, dst_y, dst_x) / alpha) + b);
    dst.at(dst_y, dst_x, 1) = cast<U>(a * clamp(src.at(1, dst_y, dst_x) / alpha) + b);
    dst.at(dst_y, dst_x, 2) = cast<U>(a * clamp(src.at(2, dst_y, dst_x) / alpha) + b);
    dst.at(dst_y, dst_x, 3) = cast<U>(a * alpha + b);
  }
}

template<class F, class U>
void import_pixel_cuda(md_view<F, int32_t, 3> dst,
                       md_view<F, int32_t, 2> dst_alpha,
                       md_view<const U, int32_t, 3> src,
                       float a,
                       float b,
                       cudaStream_t stream) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  auto [c, dst_h, dst_w] = dst.shape;
  dimGrid.x = (dst_w + 31) & (~31);
  dimGrid.y = (dst_h + 31) & (~31);

  if (dst_alpha.data == nullptr) {
    import_opaque_kernel<<<dimGrid, dimBlock, 0, stream>>>(dst, src, F(a), F(b));
  }
  else {
    assert(src.shape[2] == 4);
    import_alpha_kernel<<<dimGrid, dimBlock, 0, stream>>>(dst, dst_alpha, src, F(a), F(b));
  }
}

template<class F, class U>
void export_pixel_cuda(md_view<U, int32_t, 3> dst,
                       md_uview<const F, int32_t, 3> src,
                       md_uview<const F, int32_t, 2> src_alpha,
                       float a,
                       float b,
                       cudaStream_t stream) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  auto [dst_h, dst_w, _] = dst.shape;
  dimGrid.x = (dst_w + 31) & (~31);
  dimGrid.y = (dst_h + 31) & (~31);

  if (src_alpha.data == nullptr) {
    export_opaque_kernel<<<dimGrid, dimBlock, 0, stream>>>(dst, src, F(a), F(b));
  }
  else {
    assert(dst.shape[2] == 4);
    export_alpha_kernel<<<dimGrid, dimBlock, 0, stream>>>(dst, src, src_alpha, F(a), F(b));
  }
}

template void import_pixel_cuda<float, uint8_t>(md_view<float, int32_t, 3> dst,
                                                md_view<float, int32_t, 2> dst_alpha,
                                                md_view<const uint8_t, int32_t, 3> src,
                                                float a,
                                                float b,
                                                cudaStream_t stream);
template void import_pixel_cuda<half, uint8_t>(md_view<half, int32_t, 3> dst,
                                               md_view<half, int32_t, 2> dst_alpha,
                                               md_view<const uint8_t, int32_t, 3> src,
                                               float a,
                                               float b,
                                               cudaStream_t stream);
template void export_pixel_cuda<float, uint8_t>(md_view<uint8_t, int32_t, 3> dst,
                                                md_uview<const float, int32_t, 3> src,
                                                md_uview<const float, int32_t, 2> src_alpha,
                                                float a,
                                                float b,
                                                cudaStream_t stream);
template void export_pixel_cuda<half, uint8_t>(md_view<uint8_t, int32_t, 3> dst,
                                               md_uview<const half, int32_t, 3> src,
                                               md_uview<const half, int32_t, 2> src_alpha,
                                               float a,
                                               float b,
                                               cudaStream_t stream);
