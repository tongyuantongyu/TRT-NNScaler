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
                                   const std::filesystem::path& path_prefix)
    : config(config),
      logger(logger),
      runtime(nvinfer1::createInferRuntime(logger)),
      path_engine{},
      engine{} {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  path_engine = path_prefix / std::to_string(getInferLibVersion()) / prop.name;
}

bool InferenceContext::has_file() {
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
  COND_CHECK_EMPTY(runtime, "failed deserializing engine");

  return "";
}

static void *ptr_add(void *b, size_t n) {
  return static_cast<uint8_t *>(b) + n;
}
static size_t alignment(size_t size, size_t alignment) {
  return (size + (alignment - 1)) & (~(alignment - 1));
}

InferenceSession::InferenceSession(InferenceContext &ctx)
    : ctx(ctx), context {ctx.engine->createExecutionContextWithoutDeviceMemory()},
      last_batch(-1), last_width(-1), last_height(-1), good_ {}, stream {},
      execution_memory {}, input {}, output{}, input_consumed {} {}

std::string InferenceSession::init() {
  auto &logger = ctx.logger;
  auto &config = ctx.config;

  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaEventCreateWithFlags(&input_consumed, cudaEventBlockingSync | cudaEventDisableTiming));

  COND_CHECK_EMPTY(context->setOptimizationProfileAsync(0, stream), "bad TensorRT call.");

  COND_CHECK_EMPTY(context->setInputConsumedEvent(input_consumed), "bad TensorRT call.");
  CUDA_CHECK(cudaStreamSynchronize(stream));

  return "";
}

std::string InferenceSession::allocation() {
  auto &logger = ctx.logger;
  auto &config = ctx.config;
  const size_t eSize = config.use_fp16 ? 2 : 4;

  auto engine_alloc_size = ctx.engine->getDeviceMemorySize();
  auto input_alloc_size = size_t(config.batch.max) * config.input_height.max * config.input_width.max * 3 * eSize;
  auto [h_scale, w_scale] = detect_scale();
  auto output_alloc_size = h_scale * w_scale * input_alloc_size;
  auto total_memory = engine_alloc_size + input_alloc_size + output_alloc_size;

  size_t free_memory {};
  CUDA_CHECK(cudaMemGetInfo(&free_memory, nullptr));
  logger.log(free_memory > total_memory ? nvinfer1::ILogger::Severity::kINFO : nvinfer1::ILogger::Severity::kWARNING,
             ("Device memory: " + std::to_string(free_memory) + " bytes free, "+
             std::to_string(engine_alloc_size) + " bytes needed.").c_str());
  CUDA_CHECK(cudaMallocAsync(&execution_memory, engine_alloc_size, stream));
  context->setDeviceMemory(execution_memory);

  CUDA_CHECK(cudaMallocAsync(&input, input_alloc_size, stream));
  CUDA_CHECK(cudaMallocAsync(&output, output_alloc_size, stream));

  COND_CHECK_EMPTY(context->setTensorAddress("input", input), "bad TensorRT call.");
  COND_CHECK_EMPTY(context->setTensorAddress("output", output), "bad TensorRT call.");

  CUDA_CHECK(cudaStreamSynchronize(stream));

  good_ = true;
  return "";
}

std::string InferenceSession::deallocation() {
  good_ = false;

  void* memories[] {execution_memory, input, output};

  for (auto *p: memories) {
    if (p != nullptr) {
      CUDA_CHECK(cudaFreeAsync(p, stream));
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return "";
}

InferenceSession::~InferenceSession() {
  auto result = ([this]() -> std::string {
    if (stream == nullptr) {
      return "";
    }

    void* memories[] {execution_memory, input, output};

    for (auto *p: memories) {
      if (p != nullptr) {
        CUDA_CHECK(cudaFreeAsync(p, stream));
      }
    }

    CUDA_CHECK(cudaEventDestroy(input_consumed));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return "";
  })();

  if (!result.empty()) {
    ctx.logger.log(nvinfer1::ILogger::Severity::kWARNING, result.c_str());
  }
}

void InferenceSession::config(int32_t batch, int32_t height, int32_t width) {
  if (batch != last_batch || height != last_height || width != last_width) {
    auto config = &ctx.config;
    assert(config->batch.min <= batch <= config->batch.max);
    assert(config->input_width.min <= width <= config->input_width.max);
    assert(config->input_height.min <= height <= config->input_height.max);

    context->setInputShape("input", {4, {batch, 3, height, width}});

    last_batch = batch;
    last_width = width;
    last_height = height;
  }
}

bool InferenceSession::inference() {
  return context->enqueueV3(stream);
}

std::pair<int32_t, int32_t> InferenceSession::detect_scale() {
  if (last_batch == -1 || last_width == -1 || last_height == -1) {
    return {-1, -1};
  }

  auto shape = context->getTensorShape("output");
  assert(shape.nbDims == 4);
  assert(shape.d[0] == last_batch);
  assert(shape.d[1] == 3);
  int32_t h_scale = shape.d[2] / last_height;
  int32_t w_scale = shape.d[3] / last_width;

  if (h_scale * last_height != shape.d[2] || w_scale * last_width != shape.d[3]) {
    return {-1, -1};
  }

  return {h_scale, w_scale};
}


