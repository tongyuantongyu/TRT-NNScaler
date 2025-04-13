#pragma once

#include <filesystem>
#include <sstream>
#include <atomic>

#include "NvInfer.h"

#include "md_view.h"

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
  bool use_strong_type;
  bool use_fp16;
  bool use_int8;
  bool force_precision;
  bool external;
  bool low_mem;

  [[nodiscard]] std::string engine_name() const {
    std::stringstream ss;
    ss << "_w" << input_width << "_h" << input_height << "_b" << batch << "_a" << aux_stream;
    if (use_strong_type) {
      ss << "_stype";
    } else {
      if (use_fp16) {
        ss << "_fp16";
      }
      if (use_int8) {
        ss << "_int8";
      }
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
  bool has_file() const;
  std::string load_engine();

  bool good() {
    return runtime != nullptr && engine != nullptr;
  }
};

class InferenceSession {
  InferenceContext ctx;

  nvinfer1::IExecutionContext *context;
  void *execution_memory{};
  int32_t last_batch=-1, last_height=-1, last_width=-1;
  std::atomic<bool> good_;
  void *input_ptr{}, *output_ptr{};
  bool input_interleaved{}, output_interleaved{};
  int32_t input_channel_stride{}, output_channel_stride{};

 public:
  cudaStream_t stream{};
  cudaEvent_t input_consumed{};
  int32_t scale_w=-1, scale_h=-1;

  explicit InferenceSession(InferenceContext &ctx);
  ~InferenceSession();

  [[nodiscard]] bool good() const { return good_; }

  std::string init();
  std::string allocation();
  std::string deallocation();
  void config(int32_t batch, int32_t height, int32_t width);
  void detect_scale();

  bool inference() const;

  template<typename F>
  md_uview<F, int32_t, 3, int64_t> input(int32_t height, int32_t width) const {
    shape_t shape {3, height, width};

    shape_t stride_shape {input_channel_stride, height, width};
    if (input_interleaved) {
      stride_shape = stride_shape.gather<1, 2, 0>();
    }
    stride_t stride = stride_shape.stride<int64_t>();
    if (input_interleaved) {
      stride = stride.gather<2, 0, 1>();
    }
    return {static_cast<F *>(input_ptr), shape, stride};
  }

  template<typename F>
  md_uview<F, int32_t, 3, int64_t> output(int32_t height, int32_t width) const {
    shape_t shape {3, height * scale_h, width * scale_h};

    shape_t stride_shape {output_channel_stride, height * scale_h, width * scale_h};
    if (output_interleaved) {
      stride_shape = stride_shape.gather<1, 2, 0>();
    }
    stride_t stride = stride_shape.stride<int64_t>();
    if (output_interleaved) {
      stride = stride.gather<2, 0, 1>();
    }
    return {static_cast<F *>(output_ptr), shape, stride};
  }
};
