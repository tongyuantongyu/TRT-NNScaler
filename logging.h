#pragma once

#include <filesystem>

#include "NvInferRuntimeCommon.h"

#include "glog/logging.h"

using Severity = nvinfer1::ILogger::Severity;

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        LOG(FATAL) << "[TRT] " << msg;
        break;
      case Severity::kERROR:
        LOG(ERROR) << "[TRT] " << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << "[TRT] " << msg;
        break;
      case Severity::kINFO:
        VLOG(1) << "[TRT] [I] " << msg;
        break;
      case Severity::kVERBOSE:
        VLOG(3) << "[TRT] [V] " << msg;
        break;
    }
  }
};

static std::string u8s(const std::filesystem::path& p) {
#ifdef _WIN32
  std::u8string s = p.generic_u8string();
  return std::move(*reinterpret_cast<std::string*>(&s));
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
