//
// Created by TYTY on 2023-02-09 009.
//

#include "cmd_common.h"

struct AlphaMode {
  explicit AlphaMode(std::string m = "nn") : mode(std::move(m)) {}

  std::string mode;
};

std::string AbslUnparseFlag(AlphaMode m) {
  return m.mode;
}

bool AbslParseFlag(absl::string_view text, AlphaMode* m, std::string* error) {
  if (!absl::ParseFlag(text, &m->mode, error)) {
    return false;
  }
  if (m->mode == "nn" || m->mode == "ignore") {
    return true;
  }
  if (m->mode == "filter") {
    *error = "filter process mode is unimplemented.";
    return false;
  }
  *error = "invalid value";
  return false;
}

ABSL_FLAG(std::string, alpha, "nn", "alpha process mode: nn, filter, ignore");

ABSL_FLAG(double, pre_scale, 1.0, "scale ratio before NN super resolution.");
ABSL_FLAG(double, post_scale, 1.0, "scale ratio before NN super resolution.");

static uint64_t total_processed = 0;

static std::string handle_image(const std::filesystem::path &input, const std::filesystem::path &output, chan& works) {
  ++total_processed;
  std::promise<std::string> err_promise;
  auto err_future = err_promise.get_future();
  works.put(Work{
      input,
      output,
      std::move(err_promise),

      absl::GetFlag(FLAGS_alpha),
      absl::GetFlag(FLAGS_pre_scale),
      absl::GetFlag(FLAGS_post_scale),
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

void setup_extensions() {
  exts_storage = absl::GetFlag(FLAGS_extensions);

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

#ifdef _WIN32
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

  absl::SetProgramUsageMessage("The TensorRT Neural Network Image scaler, version v0.0.1.  Usage:\n");
  auto files = absl::ParseCommandLine(argc, argv);
  files.erase(files.begin());
  absl::InitializeLog();
  setup_session(absl::GetFlag(FLAGS_alpha) != "ignore");
  setup_extensions();

  std::error_code ec;
  std::filesystem::path output = absl::GetFlag(FLAGS_output);
  std::filesystem::create_directories(output, ec);
  if (ec) {
    LOG(QFATAL) << "Failed ensure output folder: " << ec;
  }

  LOG(INFO) << "Initialized.";

  auto start = hr_clock::now();

  {
    runner r;
    std::string err;
    for (auto file: files) {
#ifdef _WIN32
      std::filesystem::path target(argvM[file]);
#else
      std::filesystem::path target(file);
#endif
      err.clear();
      if (is_regular_file(target)) {
        err = handle_image(target, output / target.filename().replace_extension("png"), r.works);
      }
      else if (is_directory(target)) {
        err = handle_folder(target, r.works, argc == 2 && absl::GetFlag(FLAGS_output) != output_default);
      }
      else {
        err = "not a normal file or directory";
      }

      if (!err.empty()) {
        LOG(ERROR) << "Failed handling input " << target << ": " << err;
      }
    }
  }

  LOG(INFO) << "Done processing " << total_processed << " images in " << elapsed(start) / 1000 << "s.";

#ifdef _WIN32
  SetConsoleOutputCP(old_locale);
#endif
}
