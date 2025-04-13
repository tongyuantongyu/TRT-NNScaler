#include "nn-scaler.h"
#include "infer_engine.h"

#include "absl/flags/flag.h"
#include "absl/flags/declare.h"

#include <fstream>

#define COND_CHECK_EMPTY(cond, message)                                                                                \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::stringstream s;                                                                                             \
      s << "unsatisfied " << #cond ": " << message;                                                                    \
      return s.str();                                                                                                  \
    }                                                                                                                  \
  } while (0)

#define CUDA_CHECK(status)                                                                                             \
  do {                                                                                                                 \
    auto ret = (status);                                                                                               \
    if (ret != cudaSuccess) {                                                                                          \
      return "CUDA error: " + std::to_string(ret);                                                                     \
    }                                                                                                                  \
  } while (0)

ABSL_DECLARE_FLAG(std::string, model);

InferenceContext::InferenceContext(ScalerConfig config,
                                   nvinfer1::ILogger &logger,
                                   const std::filesystem::path &path_prefix)
  : config(config),
    logger(logger),
    runtime(nvinfer1::createInferRuntime(logger)),
    engine{} {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  path_engine = path_prefix / std::to_string(getInferLibVersion()) / prop.name;
}

bool InferenceContext::has_file() const {
  auto target = (path_engine / absl::GetFlag(FLAGS_model)).replace_extension();
  target += config.engine_name();
  return exists(target);
}

std::string InferenceContext::load_engine() {
  auto path = (path_engine / absl::GetFlag(FLAGS_model)).replace_extension();
  path += config.engine_name();
  std::ifstream file(path, std::ios::binary);
  COND_CHECK_EMPTY(file.good(), "can't open engine file: " << path);

  file.seekg(0, std::ifstream::end);
  auto size = file.tellg();
  file.seekg(0, std::ifstream::beg);
  auto modelStream = std::make_unique<char[]>(size);
  COND_CHECK_EMPTY(modelStream, "alloc " << size << " bytes failed.");
  file.read(modelStream.get(), size);
  file.close();

  engine = runtime->deserializeCudaEngine(modelStream.get(), size);
  COND_CHECK_EMPTY(engine, "failed deserializing engine");

  return "";
}

static void *ptr_add(void *b, size_t n) {
  return static_cast<uint8_t *>(b) + n;
}

static size_t alignment(size_t size, size_t alignment) {
  return (size + (alignment - 1)) & (~(alignment - 1));
}

InferenceSession::InferenceSession(InferenceContext &ctx)
  : ctx(ctx),
#if NV_TENSORRT_MAJOR >= 10
    context{ctx.engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED)}
#else
    context {ctx.engine->createExecutionContextWithoutDeviceMemory()}
#endif
  {}

std::string InferenceSession::init() {
  auto inputFormat = ctx.engine->getTensorFormat("input", 0);
  auto outputFormat = ctx.engine->getTensorFormat("output", 0);

  switch (inputFormat) {
    case nvinfer1::TensorFormat::kLINEAR: input_interleaved = false;
      input_channel_stride = 3;
      break;
    case nvinfer1::TensorFormat::kCHW4: input_interleaved = true;
      input_channel_stride = 4;
      break;
    case nvinfer1::TensorFormat::kHWC8: input_interleaved = true;
      input_channel_stride = 8;
      break;
    case nvinfer1::TensorFormat::kCHW32: input_interleaved = true;
      input_channel_stride = 32;
      break;
    default:
      COND_CHECK_EMPTY(false, "unsupported input format: " << int32_t(inputFormat));
  }

  switch (outputFormat) {
    case nvinfer1::TensorFormat::kLINEAR: output_interleaved = false;
      output_channel_stride = 3;
      break;
    case nvinfer1::TensorFormat::kCHW4: output_interleaved = true;
      output_channel_stride = 4;
      break;
    case nvinfer1::TensorFormat::kHWC8: output_interleaved = true;
      output_channel_stride = 8;
      break;
    case nvinfer1::TensorFormat::kCHW32: output_interleaved = true;
      output_channel_stride = 32;
      break;
    default:
      COND_CHECK_EMPTY(false, "unsupported output format: " << int32_t(outputFormat));
  }

  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaEventCreateWithFlags(&input_consumed, cudaEventBlockingSync | cudaEventDisableTiming));

  COND_CHECK_EMPTY(context->setOptimizationProfileAsync(0, stream), "bad TensorRT call.");

  COND_CHECK_EMPTY(context->setInputConsumedEvent(input_consumed), "bad TensorRT call.");
  CUDA_CHECK(cudaStreamSynchronize(stream));

  config(ctx.config.batch.opt, ctx.config.input_height.opt, ctx.config.input_width.opt);
  auto shape = context->getTensorShape("output");
  assert(shape.nbDims == 4);
  assert(shape.d[0] == last_batch);
  assert(shape.d[1] == 3);
  int32_t h_scale = shape.d[2] / ctx.config.input_height.opt;
  int32_t w_scale = shape.d[3] / ctx.config.input_width.opt;

  if (h_scale * ctx.config.input_height.opt != shape.d[2] || w_scale * ctx.config.input_width.opt != shape.d[3]) {
    return "non-integer scale ratio unsupported";
  }

  scale_w = w_scale;
  scale_h = h_scale;

  return "";
}

std::string InferenceSession::allocation() {
  auto &logger = ctx.logger;
  auto &config = ctx.config;
  const size_t eSize = config.use_fp16 ? 2 : 4;

#if NV_TENSORRT_MAJOR >= 10
  auto engine_alloc_size = ctx.engine->getDeviceMemorySizeV2();
#else
  auto engine_alloc_size = ctx.engine->getDeviceMemorySize();
#endif
  auto input_alloc_size = size_t(config.batch.max) * config.input_height.max * config.input_width.max * input_channel_stride * eSize;
  auto output_alloc_size = size_t(config.batch.max) * config.input_height.max * scale_h * config.input_width.max * scale_w * output_channel_stride * eSize;
  auto total_memory = engine_alloc_size + input_alloc_size + output_alloc_size;

  size_t free_memory{};
  CUDA_CHECK(cudaMemGetInfo(&free_memory, nullptr));
  logger.log(free_memory > total_memory ? nvinfer1::ILogger::Severity::kINFO : nvinfer1::ILogger::Severity::kWARNING,
             ("Device memory: " + std::to_string(free_memory) + " bytes free, " +
              std::to_string(engine_alloc_size) + " bytes needed.").c_str());
  CUDA_CHECK(cudaMallocAsync(&execution_memory, engine_alloc_size, stream));
  context->setDeviceMemory(execution_memory);

  CUDA_CHECK(cudaMallocAsync(&input_ptr, input_alloc_size, stream));
  CUDA_CHECK(cudaMallocAsync(&output_ptr, output_alloc_size, stream));

  COND_CHECK_EMPTY(context->setTensorAddress("input", input_ptr), "bad TensorRT call.");
  COND_CHECK_EMPTY(context->setTensorAddress("output", output_ptr), "bad TensorRT call.");

  CUDA_CHECK(cudaStreamSynchronize(stream));

  good_ = true;
  return "";
}

std::string InferenceSession::deallocation() {
  good_ = false;

  void *memories[]{execution_memory, input_ptr, output_ptr};

  for (auto *p: memories) {
    if (p != nullptr) {
      CUDA_CHECK(cudaFreeAsync(p, stream));
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return "";
}

InferenceSession::~InferenceSession() {
  auto result = [this]() -> std::string {
    if (stream == nullptr) {
      return "";
    }

    void *memories[]{execution_memory, input_ptr, output_ptr};

    for (auto *p: memories) {
      if (p != nullptr) {
        CUDA_CHECK(cudaFreeAsync(p, stream));
      }
    }

    CUDA_CHECK(cudaEventDestroy(input_consumed));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return "";
  }();

  if (!result.empty()) {
    ctx.logger.log(nvinfer1::ILogger::Severity::kWARNING, result.c_str());
  }
}

void InferenceSession::config(int32_t batch, int32_t height, int32_t width) {
  if (batch != last_batch || height != last_height || width != last_width) {
    auto config = &ctx.config;
    assert(config->batch.min <= batch && batch <= config->batch.max);
    assert(config->input_width.min <= width && width <= config->input_width.max);
    assert(config->input_height.min <= height && height <= config->input_height.max);

    context->setInputShape("input", {4, {batch, 3, height, width}});

    last_batch = batch;
    last_width = width;
    last_height = height;
  }
}

bool InferenceSession::inference() const {
  return context->enqueueV3(stream);
}
