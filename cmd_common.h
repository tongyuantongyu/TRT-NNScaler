#pragma once

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/declare.h"
#include "absl/flags/usage.h"
#include "absl/log/log.h"
#include "absl/log/initialize.h"

#include "cuda_fp16.h"

#include "nn-scaler.h"
#include "infer_engine.h"
#include "reformat/reformat.h"
#include "layers.h"
#include "image_io.h"
#include "logging_trt.h"

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

ABSL_DECLARE_FLAG(int, stderrthreshold);
ABSL_DECLARE_FLAG(bool, log_prefix);
ABSL_FLAG(std::string, model_path, "models", "path to the folder to save model files");

InferenceSession *session = nullptr;

int using_io = 0;

pixel_importer_cpu *importer_cpu = nullptr;
pixel_exporter_cpu *exporter_cpu = nullptr;

pixel_importer_gpu<float> *importer_gpu = nullptr;
pixel_exporter_gpu<float> *exporter_gpu = nullptr;

pixel_importer_gpu<half> *importer_gpu_fp16 = nullptr;
pixel_exporter_gpu<half> *exporter_gpu_fp16 = nullptr;

int32_t h_scale, w_scale;

#if defined(__GNUC__)
extern "C" __attribute__((weak)) int32_t getInferLibVersion() noexcept {
  return NV_TENSORRT_VERSION;
}
#elif defined(_MSC_VER)
extern "C" int32_t getInferLibVersion_UseHeader() noexcept {
  return NV_TENSORRT_VERSION;
}
#pragma comment(linker, "/alternatename:getInferLibVersion=getInferLibVersion_UseHeader")
#endif

static Logger gLogger;

ABSL_FLAG(bool, fp16, false, "use FP16 processing, allow FP16 in engine");
ABSL_FLAG(bool, int8, false, "allow INT8 in engine");
ABSL_FLAG(bool, force_precision, false, "Force precision config in model");
ABSL_FLAG(bool, external, false, "use external algorithms from cuDNN and cuBLAS");
ABSL_FLAG(bool, low_mem, false, "tweak configs to reduce memory consumption");
ABSL_FLAG(int32_t, aux_stream, -1, "Auxiliary streams to use");
ABSL_FLAG(std::string, reformatter, "auto", "reformatter used to import and export pixels: cpu, gpu, auto");

ABSL_FLAG(uint32_t, tile_width, 512, "tile width");
ABSL_FLAG(uint32_t, tile_height, 512, "tile height");
ABSL_FLAG(uint32_t, tile_pad, 16, "tile pad border to reduce tile block discontinuity");
ABSL_FLAG(uint32_t, extend_grace, 0, "grace limit to not split another tile");
ABSL_FLAG(uint32_t, alignment, 1, "model input alignment requirement");

ABSL_FLAG(bool, cuda_lazy_load, true, "enable CUDA lazying load.");

void setup_session(bool handle_alpha) {
  auto model_path = absl::GetFlag(FLAGS_model_path);
  auto tile_width = absl::GetFlag(FLAGS_tile_width);
  auto tile_height = absl::GetFlag(FLAGS_tile_height);
  auto tile_pad = absl::GetFlag(FLAGS_tile_pad);
  auto extend_grace = absl::GetFlag(FLAGS_extend_grace);
  auto alignment = absl::GetFlag(FLAGS_alignment);

  if (!exists(std::filesystem::path(model_path))) {
    LOG(QFATAL) << "model path " << std::quoted(model_path) << " not exist.";
  }

  if (tile_width == 0 || tile_height == 0) {
    LOG(QFATAL) << "Invalid tile size.";
  }

  if (tile_pad >= tile_width || tile_pad >= tile_height) {
    LOG(QFATAL) << "Invalid tile pad size.";
  }

  if (extend_grace >= (tile_width - tile_pad)
      || extend_grace >= (tile_height - tile_pad)) {
    LOG(QFATAL) << "Invalid tile extend grace.";
  }

  if (alignment == 0 || tile_width % alignment != 0 || tile_height % alignment != 0
      || tile_pad % alignment != 0 || extend_grace % alignment != 0) {
    LOG(QFATAL) << "Invalid tile alignment.";
  }

  // ----------------------------------
  // Lazy load
  if (absl::GetFlag(FLAGS_cuda_lazy_load)) {
    #ifdef _WIN32
    SetEnvironmentVariableW(L"CUDA_MODULE_LOADING", L"LAZY");
    #else
    setenv("CUDA_MODULE_LOADING", "LAZY", 1);
    #endif
  }


  // ----------------------------------
  // IO
  auto err = init_image_io();
  if (!err.empty()) {
    LOG(QFATAL) << "Failed init Image IO: " << err;
  }

  // ----------------------------------
  // Layers
  plugins::register_resize_plugin();

  // ----------------------------------
  // Engine
  auto max_width = absl::GetFlag(FLAGS_tile_width) + absl::GetFlag(FLAGS_extend_grace);
  auto max_height = absl::GetFlag(FLAGS_tile_height) + absl::GetFlag(FLAGS_extend_grace);

  InferenceContext ctx{
      {
          {int(std::min(
              std::max(
                  absl::GetFlag(FLAGS_extend_grace) + absl::GetFlag(FLAGS_tile_pad),
                  absl::GetFlag(FLAGS_alignment)
              ),
              MinDimension)),
           int(absl::GetFlag(FLAGS_tile_width)),
           int(max_width)},
          {int(std::min(
              std::max(
                  absl::GetFlag(FLAGS_extend_grace) + absl::GetFlag(FLAGS_tile_pad),
                  absl::GetFlag(FLAGS_alignment)
              ),
              MinDimension)),
           int(absl::GetFlag(FLAGS_tile_height)),
           int(max_height)},
          1,
          absl::GetFlag(FLAGS_aux_stream),
          absl::GetFlag(FLAGS_fp16),
          absl::GetFlag(FLAGS_int8),
          absl::GetFlag(FLAGS_force_precision),
          absl::GetFlag(FLAGS_external),
          absl::GetFlag(FLAGS_low_mem),
      },
      gLogger,
      absl::GetFlag(FLAGS_model_path)
  };

  if (!ctx.has_file()) {
    LOG(INFO) << "Building optimized engine for current tile config. This may take some time. "
                 "Some errors may occur, but as long as there are no fatal ones, this will be fine.";
    err = OptimizationContext(ctx.config, gLogger, absl::GetFlag(FLAGS_model_path)).optimize();
    if (!err.empty()) {
      LOG(QFATAL) << "Failed building optimized engine: " << err;
    }
  }

  err = ctx.load_engine();
  if (!err.empty()) {
    LOG(QFATAL) << "Failed loading engine: " << err;
  }

  session = new InferenceSession(ctx);
  session->config(1, absl::GetFlag(FLAGS_tile_height), absl::GetFlag(FLAGS_tile_width));
  err = session->init();
  if (!err.empty()) {
    LOG(QFATAL) << "Failed initialize context: " << err;
  }
  err = session->allocation();
  if (!err.empty()) {
    LOG(QFATAL) << "Failed allocate memory for context: " << err;
  }
  std::tie(h_scale, w_scale) = session->detect_scale();
  if (h_scale == -1 || w_scale == -1) {
    LOG(QFATAL) << "Bad model, can't detect scale ratio.";
  }

  if (h_scale != w_scale) {
    LOG(QFATAL) << "different width and height scale ratio unimplemented.";
  }

  // ------------------------------
  // Import & Export
  auto max_size = size_t(max_width) * max_height;

  if (absl::GetFlag(FLAGS_reformatter) == "auto") {
    absl::SetFlag(&FLAGS_reformatter, absl::GetFlag(FLAGS_fp16) ? "gpu" : "cpu");
  }
  if (absl::GetFlag(FLAGS_fp16) && absl::GetFlag(FLAGS_reformatter) == "cpu") {
    LOG(QFATAL) << "CPU reformatter can not handle FP16.";
  }

  if (absl::GetFlag(FLAGS_reformatter) == "cpu") {
    importer_cpu = new pixel_importer_cpu(max_size, handle_alpha);
    exporter_cpu = new pixel_exporter_cpu(h_scale * w_scale * max_size, handle_alpha);
    using_io = 0;
  }
  else if (absl::GetFlag(FLAGS_reformatter) == "gpu") {
    if (absl::GetFlag(FLAGS_fp16)) {
      importer_gpu_fp16 = new pixel_importer_gpu<half>(max_size, handle_alpha);
      exporter_gpu_fp16 =
          new pixel_exporter_gpu<half>(h_scale * w_scale * max_size, handle_alpha);
      using_io = 2;
    }
    else {
      importer_gpu = new pixel_importer_gpu<float>(max_size, handle_alpha);
      exporter_gpu =
          new pixel_exporter_gpu<float>(h_scale * w_scale * max_size, handle_alpha);
      using_io = 1;
    }
  }
  else {
    LOG(QFATAL) << "Unknown reformatter.";
  }
}

struct runner {
  chan works;
  std::thread pipeline;

  runner() : works{},  pipeline{launch_pipeline, std::ref(works), nullptr} {};
  ~runner() {
    works.close();
    pipeline.join();
  }
};
