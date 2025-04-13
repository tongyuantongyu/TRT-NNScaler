#pragma once

#include <cstdint>
#include <utility>
#include <string>
#include <variant>

#include <cuda_runtime_api.h>

#include "nn-scaler.h"
#include "md_view.h"

std::string init_image_io();

struct pinned_deleter {
  void operator()(void* p) const {
    cudaFreeHost(p);
  }
};

struct pinned_memory : std::unique_ptr<uint8_t[], pinned_deleter> {
  struct alloc_flag {
    uint32_t flags;
  };

  constexpr static alloc_flag alloc_default {cudaHostAllocDefault};
  constexpr static alloc_flag alloc_h2d {cudaHostAllocWriteCombined};

  pinned_memory() = default;

  explicit pinned_memory(size_t count, alloc_flag flags=alloc_default) {
    void* mem{};
    auto result = cudaHostAlloc(&mem, count, flags.flags);
    if (result != cudaSuccess) {
      throw std::bad_alloc();
    }
    this->reset(static_cast<uint8_t*>(mem));
  }
};

using mem_owner = pinned_memory;

template<typename U, size_t DIMS>
static std::pair<md_view<U, int32_t, DIMS>, mem_owner> alloc_buffer(mem_owner::alloc_flag flags, shape_t<int32_t, DIMS> s) {
  auto ptr = pinned_memory(s.count() * sizeof(U), flags);
  md_view<U, int32_t, DIMS> view = {reinterpret_cast<U *>(ptr.get()), s};
  return {view, std::move(ptr)};
}

template<typename U, typename ...D>
static std::pair<md_view<U, int32_t, sizeof...(D)>, mem_owner> alloc_buffer(mem_owner::alloc_flag flags, D... d) {
  shape_t<int32_t, sizeof...(D)> s{static_cast<int32_t>(d)...};
  auto ptr = pinned_memory(s.count() * sizeof(U), flags);
  md_view<U, int32_t, sizeof...(D)> view = {reinterpret_cast<U *>(ptr.get()), s};
  return {view, std::move(ptr)};
}

template<typename U, size_t DIMS>
static std::pair<md_view<U, int32_t, DIMS>, mem_owner> alloc_buffer(shape_t<int32_t, DIMS> s) {
  return alloc_buffer<U>(pinned_memory::alloc_default, s);
}

template<typename U, typename ...D>
static std::pair<md_view<U, int32_t, sizeof...(D)>, mem_owner> alloc_buffer(D... d) {
  return alloc_buffer<U>(pinned_memory::alloc_default, d...);
}

std::variant<std::pair<shape_t<int32_t, 3>, mem_owner>, std::string>
load_image(Work::input_t file, bool ignore_alpha);
std::string save_image(Work::output_t file, md_view<uint8_t, int32_t, 3> data);
std::string save_image_png(Work::output_t file, md_view<uint8_t, int32_t, 3> data);