//
// Created by TYTY on 2023-02-09 009.
//

#include <fstream>
#include <thread>
#include "semaphore.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/declare.h"
#include "absl/flags/usage.h"
#include "absl/log/log.h"
#include "absl/log/initialize.h"

#include "scaler_server.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include "nn-scaler.h"
#include "logging.h"

ABSL_DECLARE_FLAG(int, stderrthreshold);
ABSL_DECLARE_FLAG(bool, log_prefix);
ABSL_FLAG(uint32_t, v, 0, "verbosity log level");

using grpc::Channel;
using grpc::ChannelArguments;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;

using namespace scaler_server;

struct AlphaMode {
  explicit AlphaMode(std::string m = "nn") : mode(std::move(m)) {}

  std::string mode;
};

std::string AbslUnparseFlag(AlphaMode m) {
  return m.mode;
}

bool AbslParseFlag(absl::string_view text, AlphaMode *m, std::string *error) {
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

ABSL_FLAG(std::string, server, "localhost:6000", "Scaler server address");

static uint64_t total_processed = 0;

class ScalerClient {
 public:
  explicit ScalerClient(std::shared_ptr<Channel> &&channel)
      : stub_(NNScaler::NewStub(channel)), sema(sema.max()) {}

  void Process(const std::filesystem::path &input, const std::filesystem::path &output) {
    sema.acquire();
    VLOG(2) << "Processing file " << u8s(input);

    auto *call = new AsyncClientCall;
    call->output = output;
    call->start = hr_clock::now();

    std::string buffer;
    buffer.resize(std::filesystem::file_size(input));
    std::ifstream input_file(input, std::ios::in | std::ios::binary);
    input_file.read(buffer.data(), buffer.size());
    input_file.close();

    ScaleRequest request;
    request.set_data(std::move(buffer));
    request.set_alpha_mode(absl::GetFlag(FLAGS_alpha));
    request.set_pre_scale(absl::GetFlag(FLAGS_pre_scale));
    request.set_post_scale(absl::GetFlag(FLAGS_post_scale));

    call->response_reader = stub_->PrepareAsyncProcess(&call->context, request, &cq_);
    call->response_reader->StartCall();
    call->response_reader->Finish(&call->reply, &call->status, (void *) call);

    VLOG(2) << "Committed file " << u8s(input);
  }

  void AsyncCompleteRpc() {
    void *got_tag;
    bool ok = false;

    while (cq_.Next(&got_tag, &ok)) {
      sema.release(1);
      auto *call = static_cast<AsyncClientCall *>(got_tag);

      LOG_IF(FATAL, !ok) << "RPC unexpected failure";
      if (!call->status.ok()) {
        LOG(ERROR) << "RPC transport failure " << call->status.error_code() << ": " << call->status.error_message();
        continue;
      }

      if (call->reply.has_error()) {
        LOG(ERROR) << "Scaler process error: " << call->reply.error();
        continue;
      }

      std::ofstream of(call->output, std::ios::out | std::ios::binary | std::ios::trunc);
      if (!of.is_open()) {
        LOG(ERROR) << "Can't open output file " << u8s(call->output);
      }

      const auto &data = call->reply.data();
      of.write(data.data(), data.size());

      LOG(INFO) << "Image " << u8s(call->output) << " saved in " << elapsed(call->start) << "ms";

      delete call;
    }
  }

  void Close() {
    cq_.Shutdown();
  }

 private:
  struct AsyncClientCall {
    ScaleResponse reply;
    std::filesystem::path output;

    std::chrono::time_point<hr_clock> start;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<ScaleResponse>> response_reader;
  };

  std::unique_ptr<NNScaler::Stub> stub_;
  CompletionQueue cq_;
  cyan::counting_semaphore<4> sema;
};

static std::string handle_image(const std::filesystem::path &input,
                                const std::filesystem::path &output,
                                ScalerClient &cli) {
  ++total_processed;
  std::promise<std::string> err_promise;
  auto err_future = err_promise.get_future();
  cli.Process(input, output);
  return "";
}

ABSL_FLAG(std::string, extensions, "jpg,png", "extensions that should be processed");
ABSL_FLAG(std::string, output, "output", "path to the folder to save processed results");
constexpr char output_default[] = "output";

static std::string exts_storage;
static std::vector<std::string_view> exts;

static std::string handle_folder(const std::filesystem::path &input, ScalerClient &cli, bool spread) {
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

    auto err = handle_image(file, target.replace_extension("png"), cli);
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

  if (exists(exe_path / "flags_client.txt")) {
    absl::SetFlag(&FLAGS_flagfile, {u8s(exe_path / "flags_client.txt")});
  }
  absl::SetFlag(&FLAGS_stderrthreshold, int(absl::LogSeverity::kInfo));

#ifdef NDEBUG
  absl::SetFlag(&FLAGS_log_prefix, false);
#endif

  absl::SetProgramUsageMessage("The TensorRT Neural Network Image scaler GRPC client, version v0.0.1.  Usage:\n");
  auto files = absl::ParseCommandLine(argc, argv);
  files.erase(files.begin());
  absl::InitializeLog();
  setup_extensions();

  std::error_code ec;
  std::filesystem::path output = absl::GetFlag(FLAGS_output);
  std::filesystem::create_directories(output, ec);
  if (ec) {
    LOG(QFATAL) << "Failed ensure output folder: " << ec;
  }

  std::string target_str = absl::GetFlag(FLAGS_server);
  ChannelArguments args;
  args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);
  args.SetMaxReceiveMessageSize(1024 * 1024 * 1024);
  ScalerClient cli(grpc::CreateCustomChannel(target_str, grpc::InsecureChannelCredentials(), args));

  // Spawn reader thread that loops indefinitely
  std::thread handler{&ScalerClient::AsyncCompleteRpc, &cli};

  LOG(INFO) << "Initialized.";

  auto start = hr_clock::now();

  std::string err;
  for (auto file: files) {
#ifdef _WIN32
    std::filesystem::path target(argvM[file]);
#else
    std::filesystem::path target(file);
#endif
    err.clear();
    if (is_regular_file(target)) {
      err = handle_image(target, output / target.filename().replace_extension("png"), cli);
    }
    else if (is_directory(target)) {
      err = handle_folder(target, cli, argc == 2 && absl::GetFlag(FLAGS_output) != output_default);
    }
    else {
      err = "not a normal file or directory";
    }

    if (!err.empty()) {
      LOG(ERROR) << "Failed handling input " << target << ": " << err;
    }
  }

  cli.Close();
  handler.join();

  LOG(INFO) << "Done processing " << total_processed << " images in " << elapsed(start) / 1000 << "s.";

#ifdef _WIN32
  SetConsoleOutputCP(old_locale);
#endif
}
