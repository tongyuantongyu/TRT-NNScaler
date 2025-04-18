//
// Created by TYTY on 2023-02-09 009.
//

#include <fstream>

#include "infer_engine.h"
#include "NvOnnxParser.h"

#include "absl/flags/flag.h"
#include "logging.h"

ABSL_FLAG(std::string, model, "model.onnx", "Source model name");

#define COND_CHECK_EMPTY(cond, message)                                                                                \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::stringstream s;                                                                                             \
      s << "Unsatisfied " << #cond ": " << message;                                                                    \
      return s.str();                                                                                                  \
    }                                                                                                                  \
  } while (0)

nvinfer1::IBuilderConfig *OptimizationContext::prepareConfig() const {
  auto conf = builder->createBuilderConfig();
  if (!config.use_strong_type) {
    if (config.use_fp16) {
      conf->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (config.use_int8) {
      conf->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
  }
  conf->setFlag(nvinfer1::BuilderFlag::kTF32);
  conf->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
  if (config.force_precision) {
    conf->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
  }
  else {
#if NV_TENSORRT_MAJOR == 8
    conf->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
#endif
  }
  conf->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
#if NV_TENSORRT_MAJOR == 8
  conf->setPreviewFeature(nvinfer1::PreviewFeature::kPROFILE_SHARING_0806, true);
#endif
  if (config.aux_stream != -1) {
    conf->setMaxAuxStreams(config.aux_stream);
  }
#if NV_TENSORRT_MAJOR == 8
  if (config.external) {
    conf->setTacticSources(conf->getTacticSources() | nvinfer1::TacticSources(
        (1u << int32_t(nvinfer1::TacticSource::kCUDNN)) |
            (1u << int32_t(nvinfer1::TacticSource::kCUBLAS)) |
            (1u << int32_t(nvinfer1::TacticSource::kCUBLAS_LT))));
    conf->setPreviewFeature(nvinfer1::PreviewFeature::kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, false);
  } else {
    conf->setTacticSources(conf->getTacticSources() & ~nvinfer1::TacticSources(
        (1u << int32_t(nvinfer1::TacticSource::kCUDNN)) |
            (1u << int32_t(nvinfer1::TacticSource::kCUBLAS)) |
            (1u << int32_t(nvinfer1::TacticSource::kCUBLAS_LT))));
    conf->setPreviewFeature(nvinfer1::PreviewFeature::kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, true);
  }
#endif
  if (config.low_mem) {
    conf->setTacticSources(conf->getTacticSources() & ~nvinfer1::TacticSources(
                               (1u << int32_t(nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS))));
  }

  if (cache != nullptr) {
    conf->setTimingCache(*cache, false);
  }

  return conf;
}

OptimizationContext::OptimizationContext(ScalerConfig config, nvinfer1::ILogger &logger,
                                         std::filesystem::path path_prefix_)
  : config(config), logger(logger), path_prefix(std::move(path_prefix_)),
    builder(nvinfer1::createInferBuilder(logger)), cache(nullptr), prop{}, total_memory{} {
  auto conf = builder->createBuilderConfig();
  cudaMemGetInfo(nullptr, &total_memory);
  cudaGetDeviceProperties(&prop, 0);
  VLOG(1) << "Device has " << total_memory << " byte memory.";

  if (!config.use_fp16) {
    // CUDA Architecture 6.1 (Pascal, GTX10xx series) does not have really useful FP16.
    if (prop.major > 6 || (prop.major == 6 && prop.minor != 1)) {
      LOG(WARNING) << "Fast FP16 is available but not enabled.";
    }
  }

  path_engine = path_prefix / std::to_string(getInferLibVersion()) / prop.name;

  auto cache_file = path_engine / "timing.cache";
  std::ifstream input(cache_file, std::ios::binary | std::ios::in);
  if (input.is_open()) {
    VLOG(1) << "Loading timing.cache";
    auto size = std::filesystem::file_size(cache_file);
    auto *values = new char[size];
    input.read(values, size);
    cache = conf->createTimingCache(values, size);
    delete[] values;
    input.close();
  }
  if (cache == nullptr) {
    VLOG(1) << "Creating new timing.cache";
    cache = conf->createTimingCache(nullptr, 0);
  }
}

nvinfer1::INetworkDefinition *OptimizationContext::createNetwork() const {
#if NV_TENSORRT_MAJOR >= 10
  uint32_t flags = 0;
#else
  uint32_t flags = 1u << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  if (config.use_strong_type) {
    flags |= 1u << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
  }
  return builder->createNetworkV2(flags);
}

OptimizationContext::~OptimizationContext() {
  if (cache != nullptr) {
    std::ofstream output(path_engine / "timing.cache", std::ios::binary | std::ios::out);
    auto memory = cache->serialize();
    output.write(static_cast<char *>(memory->data()), memory->size());
    output.close();
  }
}

std::string OptimizationContext::optimize() {
  auto target = (path_engine / absl::GetFlag(FLAGS_model)).replace_extension();
  target += config.engine_name();
  if (exists(target)) {
    return "";
  }

  auto source_file = (path_prefix / absl::GetFlag(FLAGS_model)).replace_extension(".onnx");
  std::ifstream input(source_file, std::ios::binary | std::ios::in);
  COND_CHECK_EMPTY(input.is_open(), "source model file not exist: " << source_file);
  std::vector<uint8_t> source(std::filesystem::file_size(source_file));
  input.read((char *) (source.data()), source.size());
  auto network = createNetwork();
  auto profile = builder->createOptimizationProfile();
  auto parser = nvonnxparser::createParser(*network, logger);
  COND_CHECK_EMPTY(parser->parse(source.data(), source.size()), "Failed parse source model.");
  input.clear();
  VLOG(1) << "Source model loaded.";

  auto inputTensor = network->getInput(0);
  auto outputTensor = network->getOutput(0);

  inputTensor->setName("input");
  outputTensor->setName("output");

  auto ioDataType = config.use_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
  inputTensor->setType(ioDataType);
  outputTensor->setType(ioDataType);

  auto allowedFormats = 1u << uint32_t(nvinfer1::TensorFormat::kLINEAR) |//
                        1u << uint32_t(nvinfer1::TensorFormat::kCHW4) |//
                        1u << uint32_t(nvinfer1::TensorFormat::kHWC8) |//
                        1u << uint32_t(nvinfer1::TensorFormat::kCHW32);
  // auto allowedFormats = 1u << uint32_t(nvinfer1::TensorFormat::kLINEAR);
  inputTensor->setAllowedFormats(allowedFormats);
  outputTensor->setAllowedFormats(allowedFormats);

  auto height = config.input_height;
  auto width = config.input_width;
  auto batch = config.batch;
  profile->setDimensions("input",
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims4{batch.min, 3, height.min, width.min});
  profile->setDimensions("input",
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims4{batch.opt, 3, height.opt, width.opt});
  profile->setDimensions("input",
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims4{batch.max, 3, height.max, width.max});

  VLOG(1) << "Done define network.";

  auto optimize_config = prepareConfig();
  // value from experience
  //  optimize_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, total_memory / 2);
  optimize_config->addOptimizationProfile(profile);
  auto modelStream = builder->buildSerializedNetwork(*network, *optimize_config);
  COND_CHECK_EMPTY(modelStream != nullptr, "Failed build network, possibly out of memory.");
  VLOG(1) << "Done build network.";

  std::filesystem::create_directories(path_engine);
  std::ofstream p(target, std::ios::binary);
  COND_CHECK_EMPTY(p.is_open(), "Unable to open engine file for output.");
  p.write(static_cast<const char *>(modelStream->data()), modelStream->size());
  p.close();
  VLOG(1) << "Done save engine.";

  auto runtime = nvinfer1::createInferRuntime(logger);
  auto engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size());
  auto inspector = engine->createEngineInspector();

#if NV_TENSORRT_MAJOR >= 10
  auto context = engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
#else
  auto context = engine->createExecutionContextWithoutDeviceMemory();
#endif

  context->setOptimizationProfileAsync(0, nullptr);
  cudaStreamSynchronize(nullptr);

  context->setInputShape("input", nvinfer1::Dims4{batch.opt, 3, height.opt, width.opt});
  context->inferShapes(0, nullptr);

  inspector->setExecutionContext(context);

  auto path_layers = target;

  path_layers.replace_extension(".layers.json");
  std::ofstream info(path_layers, std::ios::binary);
  info << inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
  info.close();

  delete inspector;
  delete context;
  delete engine;
  delete runtime;

  return "";
}