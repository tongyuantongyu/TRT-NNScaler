#pragma once

#include "logging.h"

#include "NvInferRuntimeCommon.h"

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
        VLOG(2) << "[TRT] [I] " << msg;
        break;
      case Severity::kVERBOSE:
        VLOG(4) << "[TRT] [V] " << msg;
        break;
    }
  }
};
