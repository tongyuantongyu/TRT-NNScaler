//
// Created by TYTY on 2023-03-01 001.
//

#include <fstream>
#include <filesystem>

#include "NvInfer.h"
#include "gflags/gflags.h"
#include "logging.h"

void custom_prefix(std::ostream &s, const google::LogMessageInfo &l, void *) {
  switch (l.severity[0]) {
    case 'I':s << "[INFO ]";
      break;
    case 'W':s << "[WARN ]";
      break;
    case 'E':s << "[ERROR]";
      break;
    case 'F':s << "[FATAL]";
      break;
    default:s << "[?    ]";
      break;
  }
}

static Logger gLogger;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage("TensorRT engine inspector.");
  gflags::SetVersionString("0.0.1");
  google::InitGoogleLogging(argv[0], custom_prefix);
  google::InstallFailureFunction([]() {
    exit(1);
  });

  if (argc == 1) {
    gflags::ShowUsageWithFlags(argv[0]);
    return 1;
  }

  std::filesystem::path engine_file(argv[1]);

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