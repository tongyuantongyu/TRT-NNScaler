#pragma once

#include <cstdint>
#include <utility>
#include <string>
#include <variant>
#include <filesystem>

#include "nn-scaler.h"
#include "md_view.h"

std::string init_image_io();

typedef std::unique_ptr<uint8_t[]> mem_owner;

template<typename U, size_t DIMS>
static std::pair<md_view<U, DIMS>, mem_owner> alloc_buffer(shape_t<DIMS> s) {
  auto ptr = std::make_unique<uint8_t[]>(s.count() * sizeof(U));
  md_view<U, DIMS> view = {reinterpret_cast<U*>(ptr.get()), s};
  return {view, std::move(ptr)};
}

template<typename U, typename ...D>
static std::pair<md_view<U, sizeof...(D)>, mem_owner> alloc_buffer(D... d) {
  shape_t<sizeof...(D)> s {d...};
  auto ptr = std::make_unique<uint8_t[]>(s.count() * sizeof(U));
  md_view<U, sizeof...(D)> view = {reinterpret_cast<U*>(ptr.get()), s};
  return {view, std::move(ptr)};
}

std::variant<std::pair<shape_t<3>, mem_owner>, std::string>
load_image(Work::input_t file, bool ignore_alpha);
std::string save_image(Work::output_t file, md_view<uint8_t, 3> data);
std::string save_image_png(Work::output_t file, md_view<uint8_t, 3> data);