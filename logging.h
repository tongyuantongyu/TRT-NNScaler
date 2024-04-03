#pragma once

#include <filesystem>

#include "absl/flags/flag.h"
#include "absl/flags/declare.h"
#include "absl/log/log.h"

static std::string u8s(const std::filesystem::path& p) {
#ifdef _WIN32
#ifdef __cpp_lib_char8_t
  std::u8string s = p.generic_u8string();
  std::string r(s.size(), '\0');
  memcpy(r.data(), s.data(), s.size());
  return std::move(r);
#else
  return p.generic_u8string();
#endif
#else
  return p.string();
#endif
}

struct path_output_wrapper {
  const std::filesystem::path& p;
};

static std::ostream &operator<<(std::ostream &os, const path_output_wrapper &o) {
#ifdef _WIN32
  os << std::quoted(u8s(o.p));
#else
  os << o.p;
#endif
  return os;
}

static path_output_wrapper o(const std::filesystem::path& p) {
  return {p};
}
