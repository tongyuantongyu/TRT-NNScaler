#include "reformat.h"


std::string pixel_importer_cpu::import_alpha(md_view<float, 3> dst, cudaStream_t stream) {
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

std::string pixel_exporter_cpu_crop::fetch_color(md_view<const float, 3> src, cudaStream_t stream) {
  auto [c, h, w] = src.shape;

  if (h * w > max_size) {
    return "dimension too big";
  }

  auto err = cudaMemcpyAsync(buffer.get(), src.data, src.size() * 4, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  current_buffer_shape = src.shape;
  return "";
}

std::string pixel_exporter_cpu_crop::fetch_alpha(md_view<const float, 3> src, cudaStream_t stream) {
  auto [c, h, w] = src.shape;

  if (h * w > max_size) {
    return "dimension too big";
  }

  if (current_buffer_shape != src.shape) {
    return "incompatible color buffer shape";
  }

  auto err = cudaMemcpyAsync(buffer_alpha.get(), src.data, h * w * 4, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return std::string("CUDA error: ") + cudaGetErrorName(err);
  }

  return "";
}
