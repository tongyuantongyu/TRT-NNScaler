//
// Created by TYTY on 2023-03-01 001.
//

#include <fstream>
#include <filesystem>

#include "NvInfer.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "logging.h"

static Logger gLogger;

ABSL_DECLARE_FLAG(int, stderrthreshold);
ABSL_FLAG(uint32_t, v, 0, "verbosity log level");

int main(int argc, char** argv) {
  absl::SetFlag(&FLAGS_stderrthreshold, int(absl::LogSeverity::kInfo));

  auto files = absl::ParseCommandLine(argc, argv);
  absl::SetProgramUsageMessage("The TensorRT Neural Network inspector, version v0.0.1.  Usage:\n");

  if (files.empty()) {
    LOG(ERROR) << "No model path provided.";
    return 1;
  }

  std::filesystem::path engine_file(files[0]);

  auto runtime = nvinfer1::createInferRuntime(gLogger);
  std::ifstream file(engine_file, std::ios::binary);
  if (!file.good()) {
    LOG(FATAL) << "can't open engine file: " << argv[1];
  }

  file.seekg(0, std::ifstream::end);
  auto size = file.tellg();
  file.seekg(0, std::ifstream::beg);
  auto modelStream = std::make_unique<char[]>(size);
  file.read(modelStream.get(), size);
  file.close();

  auto engine = runtime->deserializeCudaEngine(modelStream.get(), size);
  if (engine == nullptr) {
    LOG(FATAL) << "Failed deserialize engine.";
  }

  LOG(INFO) << "Engine has " << engine->getNbIOTensors() << " IO ports, " << engine->getNbLayers() << " layers.";

  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    auto name = engine->getIOTensorName(i);
    auto ioMode = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output";
    LOG(INFO) << "#" << i << ": " << ioMode << " '" << name << "'.";
  }

  auto inspector = engine->createEngineInspector();

  LOG(INFO) << "Export layer info.";

  auto layer_info_file = engine_file;
  layer_info_file.replace_extension();
  layer_info_file += "_layer_info.json";
  std::ofstream info(layer_info_file, std::ios::binary);
  for (int i = 0; i < engine->getNbLayers(); ++i) {
    info << "----- [ #" << i << " ] -----------------------------------------------------------------\n";
    info << inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON) << "\n\n";
  }

  info.close();
}