#include "reformat_cuda.h"
#include <cuda_fp16.h>

template<class F, class U>
static void __global__ fma_from(md_view<F, 3> dst, md_view<const U, 3> src, F a, F b) {
  uint32_t dst_x = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t dst_y = threadIdx.y + blockDim.y * blockIdx.y;
  uint32_t c = blockIdx.z;

  auto [_1, dst_h, dst_w] = dst.shape;
  if (dst_x >= dst_w || dst_y >= dst_h) {
    return;
  }

  auto [src_h, src_w, _2] = src.shape;
  uint32_t src_x = dst_x >= src_w ? src_w - 1 : dst_x;
  uint32_t src_y = dst_y >= src_h ? src_h - 1 : dst_y;

  F value = static_cast<F>(src.at(src_y, src_x, c));
  value = a * value + b;
  dst.at(c, dst_y, dst_x) = value;
}

template<class F, class U>
void import_pixel_cuda(md_view<F, 3> dst, md_view<const U, 3> src, float a, float b, cudaStream_t stream) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  auto [c, dst_h, dst_w] = dst.shape;
  assert(c == src.shape[2]);
  dimGrid.x = (dst_w + 31) & (~31);
  dimGrid.y = (dst_h + 31) & (~31);
  dimGrid.z = 3;

  fma_from<<<dimGrid, dimBlock, 0, stream>>>(dst, src, F(a), F(b));
}

template void import_pixel_cuda<float, uint8_t>(md_view<float, 3> dst, md_view<const uint8_t, 3> src, float a, float b,
                                                cudaStream_t stream);
template void import_pixel_cuda<half, uint8_t>(md_view<half, 3> dst, md_view<const uint8_t, 3> src, float a, float b,
                                               cudaStream_t stream);
