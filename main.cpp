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

#include "gflags/gflags.h"
#include "cuda_fp16.h"

#include "nn-scaler.h"
#include "logging.h"
#include "reformat/reformat.h"
#include "image_io.h"

DEFINE_string(model_path, "models", "path to the folder to save model files");

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

DEFINE_string(extensions, "jpg,png", "extensions that should be processed");
DEFINE_string(output, "output", "path to the folder to save processed results");
constexpr char output_default[] = "output";

static std::vector<std::string_view> exts;

static std::string handle_folder(const std::filesystem::path &input, chan &works, bool spread) {
  assert(is_directory(input));

  std::error_code ec;
  auto output = std::filesystem::path(FLAGS_output);
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
      LOG(DFATAL) << "Failed prepare output directory: " << ec;
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

DEFINE_bool(fp16, false, "use FP16 processing");
DEFINE_bool(external, false, "use external algorithms from cuDNN and cuBLAS");
DEFINE_bool(low_mem, false, "tweak configs to reduce memory consumption");
DEFINE_string(reformatter, "auto", "reformatter used to import and export pixels: cpu, gpu, auto");

DECLARE_string(alpha);
DEFINE_int32(tile_width, 512, "tile width");
DEFINE_int32(tile_height, 512, "tile height");
DEFINE_int32(tile_pad, 16, "tile pad border to reduce tile block discontinuity");
DEFINE_int32(extend_grace, 0, "grace limit to not split another tile");
DECLARE_int32(alignment);

void verify_flags() {
  if (!exists(std::filesystem::path(FLAGS_model_path))) {
    LOG(FATAL) << "model path " << std::quoted(FLAGS_model_path) << " not exist.";
  }

  if (FLAGS_tile_width <= 0 || FLAGS_tile_height <= 0) {
    LOG(FATAL) << "Invalid tile size.";
  }

  if (FLAGS_tile_pad < 0 || FLAGS_tile_pad >= FLAGS_tile_width || FLAGS_tile_pad >= FLAGS_tile_height) {
    LOG(FATAL) << "Invalid tile pad size.";
  }

  if (FLAGS_extend_grace < 0 || FLAGS_extend_grace >= (FLAGS_tile_width - FLAGS_tile_pad)
      || FLAGS_extend_grace >= (FLAGS_tile_height - FLAGS_tile_pad)) {
    LOG(FATAL) << "Invalid tile extend grace.";
  }

  if (FLAGS_alignment < 1 || FLAGS_tile_width % FLAGS_alignment != 0 || FLAGS_tile_height % FLAGS_alignment != 0
      || FLAGS_tile_pad % FLAGS_alignment != 0 || FLAGS_extend_grace % FLAGS_alignment != 0) {
    LOG(FATAL) << "Invalid tile alignment.";
  }

  auto ext_count = std::count(FLAGS_extensions.begin(), FLAGS_extensions.end(), ',');
  exts.reserve(ext_count + 1);
  exts.emplace_back(FLAGS_extensions);
  for (int i = 0; i < ext_count; ++i) {
    auto comma_pos = exts.back().find(',');
    if (comma_pos == 0 || comma_pos == std::string_view::npos) {
      LOG(FATAL) << "Invalid extension list";
    }
    auto &split = exts.back();
    std::back_inserter(exts) = split.substr(comma_pos + 1);
    split = split.substr(0, comma_pos);
  }
}

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

DECLARE_string(flagfile);

DEFINE_bool(cuda_lazy_load, true, "enable CUDA lazying load.");

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

  FLAGS_logtostderr = true;
  if (exists(exe_path / "flags.txt")) {
    FLAGS_flagfile = (exe_path / "flags.txt").string();
  }
  FLAGS_model_path = (exe_path / "models").string();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage("The TensorRT Neural Network Image scaler.");
  gflags::SetVersionString("0.0.1");
  google::InitGoogleLogging(argv[0], custom_prefix);
  google::InstallFailureFunction([]() {
    exit(1);
  });
  verify_flags();

  if (argc == 1) {
    gflags::ShowUsageWithFlags(argv[0]);
    return 1;
  }

  if (FLAGS_cuda_lazy_load) {
    #ifdef _WIN32
    SetEnvironmentVariableW(L"CUDA_MODULE_LOADING", L"LAZY");
    #else
    setenv("CUDA_MODULE_LOADING", "LAZY", 1);
    #endif
  }

  std::error_code ec;
  std::filesystem::path output = FLAGS_output;
  std::filesystem::create_directories(output, ec);
  if (ec) {
    LOG(FATAL) << "Failed ensure output folder: " << ec;
  }

  // ----------------------------------
  // IO
  auto err = init_image_io();
  if (!err.empty()) {
    LOG(FATAL) << "Failed init Image IO: " << err;
    return 1;
  }

  // ----------------------------------
  // Engine
  auto max_width = FLAGS_tile_width + FLAGS_extend_grace;
  auto max_height = FLAGS_tile_height + FLAGS_extend_grace;

  InferenceContext ctx{
      {
          {std::min(std::max(FLAGS_extend_grace + FLAGS_tile_pad, FLAGS_alignment), MinDimension), FLAGS_tile_width, max_width},
          {std::min(std::max(FLAGS_extend_grace + FLAGS_tile_pad, FLAGS_alignment), MinDimension), FLAGS_tile_height, max_height},
          1,
          FLAGS_fp16,
          FLAGS_external,
          FLAGS_low_mem,
      },
      gLogger,
      FLAGS_model_path
  };

  if (!ctx.has_file()) {
    LOG(INFO) << "Building optimized engine for current tile config. This may take some time. "
                 "Some errors may occur, but as long as there are no fatal ones, this will be fine.";
    err = OptimizationContext(ctx.config, gLogger, FLAGS_model_path).optimize();
    if (!err.empty()) {
      LOG(FATAL) << "Failed building optimized engine: " << err;
    }
  }

  err = ctx.load_engine();
  if (!err.empty()) {
    LOG(FATAL) << "Failed loading engine: " << err;
  }

  session = new InferenceSession(ctx);
  session->config(1, FLAGS_tile_height, FLAGS_tile_width);
  err = session->allocation();
  if (!err.empty()) {
    LOG(FATAL) << "Failed initialize context: " << err;
  }
  std::tie(h_scale, w_scale) = session->detect_scale();
  if (h_scale == -1 || w_scale == -1) {
    LOG(FATAL) << "Bad model, can't detect scale ratio.";
  }

  if (h_scale != w_scale) {
    LOG(FATAL) << "different width and height scale ratio unimplemented.";
  }

  // ------------------------------
  // Import & Export
  auto max_size = size_t(max_width) * max_height;

  if (FLAGS_reformatter == "auto") {
    FLAGS_reformatter = FLAGS_fp16 ? "gpu" : "cpu";
  }
  if (FLAGS_fp16 && FLAGS_reformatter == "cpu") {
    LOG(FATAL) << "CPU reformatter can not handle FP16.";
  }

  if (FLAGS_reformatter == "cpu") {
    importer_cpu = new pixel_importer_cpu(max_size, FLAGS_alpha != "ignore");
    exporter_cpu = new pixel_exporter_cpu(h_scale * w_scale * max_size, FLAGS_alpha != "ignore");
    using_io = 0;
  } else if (FLAGS_reformatter == "gpu") {
    if (FLAGS_fp16) {
      importer_gpu_fp16 = new pixel_importer_gpu<half>(max_size, FLAGS_alpha != "ignore");
      exporter_gpu_fp16 = new pixel_exporter_gpu<half>(h_scale * w_scale * max_size, FLAGS_alpha != "ignore");
      using_io = 2;
    } else {
      importer_gpu = new pixel_importer_gpu<float>(max_size, FLAGS_alpha != "ignore");
      exporter_gpu = new pixel_exporter_gpu<float>(h_scale * w_scale * max_size, FLAGS_alpha != "ignore");
      using_io = 1;
    }
  } else {
    LOG(FATAL) << "Unknown reformatter.";
  }

  chan works;
  std::thread pipeline(launch_pipeline, std::ref(works));

  LOG(INFO) << "Initialized.";

  auto start = hr_clock::now();

  for (int i = 1; i < argc; ++i) {
#ifdef _WIN32
    std::filesystem::path target(argvM[argv[i]]);
#else
    std::filesystem::path target(argv[i]);
#endif
    err.clear();
    if (is_regular_file(target)) {
      err = handle_image(target, output / target.filename().replace_extension("png"), works);
    }
    else if (is_directory(target)) {
      err = handle_folder(target, works, argc == 2 && FLAGS_output != output_default);
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
