#pragma once

#include <filesystem>
#include <sstream>
#include <atomic>

#include "NvInfer.h"

struct optimization_axis {
  optimization_axis(int32_t min, int32_t opt, int32_t max) : min(min), opt(opt), max(max) {}
  optimization_axis(int32_t same) : min(same), opt(same), max(same) {}
  optimization_axis() : min(0), opt(0), max(0) {}
  int32_t min, opt, max;
};

static std::ostream &operator<<(std::ostream &os, const optimization_axis &o) {
  os << o.min << ',' << o.opt << ',' << o.max;
  return os;
}

struct ScalerConfig {
  optimization_axis input_width;
  optimization_axis input_height;
  optimization_axis batch;

  int32_t aux_stream;
  bool use_fp16;
  bool use_int8;
  bool force_precision;
  bool external;
  bool low_mem;

  [[nodiscard]] std::string engine_name() const {
    std::stringstream ss;
    ss << "_w" << input_width << "_h" << input_height << "_b" << batch << "_a" << aux_stream;
    if (use_fp16) {
      ss << "_fp16";
    }
    if (use_int8) {
      ss << "_int8";
    }
    if (force_precision) {
      ss << "_force_prec";
    }
    if (external) {
      ss << "_ext";
    }
    if (low_mem) {
      ss << "_lm";
    }
    ss << ".engine";
    return ss.str();
  }
};

class OptimizationContext {
  ScalerConfig config;
  nvinfer1::ILogger &logger;
  std::filesystem::path path_prefix;
  std::filesystem::path path_engine;

  nvinfer1::IBuilder *builder;
  nvinfer1::ITimingCache *cache;

  cudaDeviceProp prop;
  size_t total_memory;

  [[nodiscard]] nvinfer1::IBuilderConfig *prepareConfig() const;
  [[nodiscard]] nvinfer1::INetworkDefinition *createNetwork() const;

 public:
  OptimizationContext(ScalerConfig config, nvinfer1::ILogger &logger, std::filesystem::path path_prefix);
  std::string optimize();
  ~OptimizationContext();
};

class InferenceSession;

class InferenceContext {
  nvinfer1::ILogger &logger;
  nvinfer1::IRuntime *runtime;
  std::filesystem::path path_engine;
  nvinfer1::ICudaEngine *engine;

  friend class InferenceSession;

 public:
  ScalerConfig config;
  InferenceContext(ScalerConfig config, nvinfer1::ILogger &logger, const std::filesystem::path& path_prefix);
  bool has_file();
  std::string load_engine();

  bool good() {
    return runtime != nullptr && engine != nullptr;
  }
};

class InferenceSession {
  InferenceContext ctx;

  nvinfer1::IExecutionContext *context;
  void *execution_memory;
  int32_t last_batch, last_height, last_width;
  std::atomic<bool> good_;

 public:
  cudaStream_t stream;
  cudaEvent_t input_consumed;
  void *input, *output;

  explicit InferenceSession(InferenceContext &ctx);
  ~InferenceSession();

  [[nodiscard]] bool good() const { return good_; }

  std::string init();
  std::string allocation();
  std::string deallocation();
  void config(int32_t batch, int32_t height, int32_t width);
  std::pair<int32_t, int32_t> detect_scale();

  bool inference();
};
