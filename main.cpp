//
// Created by TYTY on 2023-02-09 009.
//

#include <iostream>
#include <chrono>
#include <vector>
#include <string_view>
#include <filesystem>
#include <future>
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/declare.h"
#include "absl/flags/usage.h"
#include "absl/log/log.h"
#include "absl/log/initialize.h"

#include "cuda_fp16.h"

#include "nn-scaler.h"
#include "reformat/reformat.h"
#include "image_io.h"
#include "logging.h"

ABSL_DECLARE_FLAG(int, stderrthreshold);
ABSL_DECLARE_FLAG(bool, log_prefix);
ABSL_FLAG(uint32_t, v, 0, "verbosity log level");
ABSL_FLAG(std::string, model_path, "models", "path to the folder to save model files");

InferenceSession *session = nullptr;

int using_io = 0;

pixel_importer_cpu *importer_cpu = nullptr;
pixel_exporter_cpu *exporter_cpu = nullptr;

pixel_importer_gpu<float> *importer_gpu = nullptr;
pixel_exporter_gpu<float> *exporter_gpu = nullptr;

pixel_importer_gpu<half> *importer_gpu_fp16 = nullptr;
pixel_exporter_gpu<half> *exporter_gpu_fp16 = nullptr;

static uint64_t total_processed = 0;

static std::string handle_image(const std::filesystem::path &input, const std::filesystem::path &output, chan& works) {
  ++total_processed;
  std::promise<std::string> err_promise;
  auto err_future = err_promise.get_future();
  works.put(Work{
      input,
      output,
      std::move(err_promise),
  });
  return err_future.get();
}

ABSL_FLAG(std::string, extensions, "jpg,png", "extensions that should be processed");
ABSL_FLAG(std::string, output, "output", "path to the folder to save processed results");
constexpr char output_default[] = "output";

static std::string exts_storage;
static std::vector<std::string_view> exts;

static std::string handle_folder(const std::filesystem::path &input, chan &works, bool spread) {
  assert(is_directory(input));

  std::error_code ec;
  auto output = std::filesystem::path(FLAGS_output.CurrentValue());
  if (!spread) output /= input.filename();

  for (auto &p: std::filesystem::recursive_directory_iterator(input,
                                                              std::filesystem::directory_options::skip_permission_denied,
                                                              ec)) {
    const auto &file = p.path();

    if (!p.is_regular_file()) {
      VLOG(2) << "Skip non-file path " << o(file);
      continue;
    }

    auto ext = file.extension().string();
    if (!ext.empty()) ext = ext.substr(1);
    if (std::find(exts.begin(), exts.end(), ext) == exts.end()) {
      VLOG(2) << "Skip extension mismatch " << o(file);
      continue;
    }

    auto target = output / relative(file, input, ec);
    if (!ec) std::filesystem::create_directories(target.parent_path(), ec);
    if (ec) {
      LOG(QFATAL) << "Failed prepare output directory: " << ec;
    }

    auto err = handle_image(file, target.replace_extension("png"), works);
    if (!err.empty()) {
      LOG(ERROR) << "Failed processing file " << o(file) << ": " << err;
    }
  }

  if (ec) {
    return "Failed listing files of path " + u8s(input) + ": " + ec.message();
  }

  return "";
}

static Logger gLogger;

ABSL_FLAG(bool, fp16, false, "use FP16 processing");
ABSL_FLAG(bool, external, false, "use external algorithms from cuDNN and cuBLAS");
ABSL_FLAG(bool, low_mem, false, "tweak configs to reduce memory consumption");
ABSL_FLAG(int32_t, aux_stream, -1, "Auxiliary streams to use");
ABSL_FLAG(std::string, reformatter, "auto", "reformatter used to import and export pixels: cpu, gpu, auto");

ABSL_DECLARE_FLAG(std::string, alpha);
ABSL_FLAG(uint32_t, tile_width, 512, "tile width");
ABSL_FLAG(uint32_t, tile_height, 512, "tile height");
ABSL_FLAG(uint32_t, tile_pad, 16, "tile pad border to reduce tile block discontinuity");
ABSL_FLAG(uint32_t, extend_grace, 0, "grace limit to not split another tile");
ABSL_FLAG(uint32_t, alignment, 1, "model input alignment requirement");

void verify_flags() {
  auto model_path = absl::GetFlag(FLAGS_model_path);
  auto tile_width = absl::GetFlag(FLAGS_tile_width);
  auto tile_height = absl::GetFlag(FLAGS_tile_height);
  auto tile_pad = absl::GetFlag(FLAGS_tile_pad);
  auto extend_grace = absl::GetFlag(FLAGS_extend_grace);
  auto alignment = absl::GetFlag(FLAGS_alignment);
  exts_storage = absl::GetFlag(FLAGS_extensions);

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

  auto ext_count = std::count(exts_storage.begin(), exts_storage.end(), ',');
  exts.reserve(ext_count + 1);
  exts.emplace_back(exts_storage);
  for (int i = 0; i < ext_count; ++i) {
    auto comma_pos = exts.back().find(',');
    if (comma_pos == 0 || comma_pos == std::string_view::npos) {
      LOG(QFATAL) << "Invalid extension list";
    }
    auto &split = exts.back();
    std::back_inserter(exts) = split.substr(comma_pos + 1);
    split = split.substr(0, comma_pos);
  }
}

ABSL_FLAG(bool, cuda_lazy_load, true, "enable CUDA lazying load.");

int32_t h_scale, w_scale;

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <unordered_map>
int wmain(int argc, wchar_t **wargv) {
  auto old_locale = GetConsoleOutputCP();
  SetConsoleOutputCP(CP_UTF8);

  std::vector<std::string> vArgvS(argc);
  std::vector<char *> vArgv(argc);
  std::unordered_map<char *, wchar_t *> argvM;
  for (int i = 0; i < argc; ++i) {
    auto size_needed = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
    if (size_needed == 0) {
      std::cerr << "Bad parameter.\n";
      return 1;
    }

    auto &arg = vArgvS[i];
    arg.resize(size_needed);
    WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, arg.data(), arg.size(), nullptr, nullptr);
    vArgv[i] = arg.data();
    argvM[arg.data()] = wargv[i];
  }
  auto argv = vArgv.data();

  std::wstring exe_path_buffer;
  exe_path_buffer.resize(4096);
  GetModuleFileNameW(nullptr, exe_path_buffer.data(), 4096);
  std::filesystem::path exe_path(std::move(exe_path_buffer));
  exe_path = exe_path.parent_path();

#else
  int main(int argc, char** argv) {
    std::filesystem::path exe_path;
#endif

  if (exists(exe_path / "flags.txt")) {
    absl::SetFlag(&FLAGS_flagfile, {u8s(exe_path / "flags.txt")});
  }
  absl::SetFlag(&FLAGS_model_path, u8s(exe_path / "models"));
  absl::SetFlag(&FLAGS_stderrthreshold, int(absl::LogSeverity::kInfo));

#ifdef NDEBUG
  absl::SetFlag(&FLAGS_log_prefix, false);
#endif

  auto files = absl::ParseCommandLine(argc, argv);
  files.erase(files.begin());
  absl::InitializeLog();
  absl::SetProgramUsageMessage("The TensorRT Neural Network Image scaler, version v0.0.1.  Usage:\n");
  verify_flags();

  if (absl::GetFlag(FLAGS_cuda_lazy_load)) {
    #ifdef _WIN32
    SetEnvironmentVariableW(L"CUDA_MODULE_LOADING", L"LAZY");
    #else
    setenv("CUDA_MODULE_LOADING", "LAZY", 1);
    #endif
  }

  std::error_code ec;
  std::filesystem::path output = absl::GetFlag(FLAGS_output);
  std::filesystem::create_directories(output, ec);
  if (ec) {
    LOG(QFATAL) << "Failed ensure output folder: " << ec;
  }

  // ----------------------------------
  // IO
  auto err = init_image_io();
  if (!err.empty()) {
    LOG(QFATAL) << "Failed init Image IO: " << err;
    return 1;
  }

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
  err = session->allocation();
  if (!err.empty()) {
    LOG(QFATAL) << "Failed initialize context: " << err;
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
    importer_cpu = new pixel_importer_cpu(max_size, absl::GetFlag(FLAGS_alpha) != "ignore");
    exporter_cpu = new pixel_exporter_cpu(h_scale * w_scale * max_size, absl::GetFlag(FLAGS_alpha) != "ignore");
    using_io = 0;
  }
  else if (absl::GetFlag(FLAGS_reformatter) == "gpu") {
    if (absl::GetFlag(FLAGS_fp16)) {
      importer_gpu_fp16 = new pixel_importer_gpu<half>(max_size, absl::GetFlag(FLAGS_alpha) != "ignore");
      exporter_gpu_fp16 =
          new pixel_exporter_gpu<half>(h_scale * w_scale * max_size, absl::GetFlag(FLAGS_alpha) != "ignore");
      using_io = 2;
    }
    else {
      importer_gpu = new pixel_importer_gpu<float>(max_size, absl::GetFlag(FLAGS_alpha) != "ignore");
      exporter_gpu =
          new pixel_exporter_gpu<float>(h_scale * w_scale * max_size, absl::GetFlag(FLAGS_alpha) != "ignore");
      using_io = 1;
    }
  }
  else {
    LOG(QFATAL) << "Unknown reformatter.";
  }

  chan works;
  std::thread pipeline(launch_pipeline, std::ref(works));

  LOG(INFO) << "Initialized.";

  auto start = hr_clock::now();

  for (auto file: files) {
#ifdef _WIN32
    std::filesystem::path target(argvM[file]);
#else
    std::filesystem::path target(argv[i]);
#endif
    err.clear();
    if (is_regular_file(target)) {
      err = handle_image(target, output / target.filename().replace_extension("png"), works);
    }
    else if (is_directory(target)) {
      err = handle_folder(target, works, argc == 2 && absl::GetFlag(FLAGS_output) != output_default);
    }
    else {
      err = "not a normal file or directory";
    }

    if (!err.empty()) {
      LOG(ERROR) << "Failed handling input " << target << ": " << err;
    }
  }

  works.close();
  pipeline.join();

  LOG(INFO) << "Done processing " << total_processed << " images in " << elapsed(start) / 1000 << "s.";

#ifdef _WIN32
  SetConsoleOutputCP(old_locale);
#endif
}
