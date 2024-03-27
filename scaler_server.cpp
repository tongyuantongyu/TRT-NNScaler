//
// Created by TYTY on 2023-10-22 022.
//

#include "cmd_common.h"
#include <thread>

#include "scaler_server.grpc.pb.h"

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

using namespace scaler_server;

std::atomic<size_t> session_using;
std::atomic<std::chrono::steady_clock::time_point> last_done;
std::mutex idle_guard;

class NNScalerImpl final : public NNScaler::Service {
  runner r;

 public:
  NNScalerImpl() : r{} {};
  Status Process(ServerContext*, const ScaleRequest* request, ScaleResponse* response) override {
    struct guard {
      guard() { session_using.fetch_add(1); }
      ~guard() {
        auto done = std::chrono::steady_clock::now();
        auto last = last_done.load();
        while (last < done) {
          if (last_done.compare_exchange_strong(last, done)) {
            break;
          }
        }
        auto n = session_using.fetch_sub(1);
      }
    };

    guard g;

    if (!session->good()) {
      std::lock_guard lock(idle_guard);
      if (!session->good()) {
        auto err = session->allocation();
        if (!err.empty()) {
          LOG(QFATAL) << "Failed reallocate workspace: " << err;
        }
      }
    }

    auto input_ = request->data();
    auto data_ = reinterpret_cast<uint8_t*>(input_.data());
    std::vector<uint8_t> input(data_, data_ + input_.size());

    std::promise<std::string> err_promise;
    auto err_future = err_promise.get_future();

    std::promise<std::vector<uint8_t>> res_promise;
    auto res_future = res_promise.get_future();

    r.works.put({
      input,
      std::move(res_promise),
      std::move(err_promise),

      request->alpha_mode(),
      request->pre_scale(),
      request->post_scale()
    });

    auto err = err_future.get();
    if (!err.empty()) {
      response->set_error(err);
      return Status::OK;
    }

    auto res = res_future.get();
    response->set_data(reinterpret_cast<char*>(res.data()), res.size());
    return Status::OK;
  }
};

ABSL_FLAG(std::string, listen, "[::]:6000", "service listen address");

[[noreturn]] void idle_cleanup() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds{10});

    if (session_using == 0 && (std::chrono::steady_clock::now() - last_done.load()) > std::chrono::minutes{1}) {
      std::unique_lock lock(idle_guard, std::try_to_lock);
      if (lock.owns_lock()) {
        if (session->good()) {
          auto err = session->deallocation();
          if (!err.empty()) {
            LOG(QFATAL) << "Failed release workspace: " << err;
          } else {
            LOG(INFO) << "Workspace memory freed due to inactivity.";
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  std::filesystem::path exe_path;
#ifdef _WIN32
  if (setlocale(LC_ALL, ".UTF8") == nullptr) {
    LOG(QFATAL) << "Failed setlocale.";
  }

  {
    wchar_t exe_path_buffer[4096] = {};
    GetModuleFileNameW(nullptr, exe_path_buffer, 4096);
    exe_path = exe_path_buffer;
    exe_path = exe_path.parent_path();
  }
#endif

  if (exists(exe_path / "flags_server.txt")) {
    absl::SetFlag(&FLAGS_flagfile, {u8s(exe_path / "flags_server.txt")});
  }
  absl::SetFlag(&FLAGS_model_path, u8s(exe_path / "models"));
  absl::SetFlag(&FLAGS_stderrthreshold, int(absl::LogSeverity::kInfo));

#ifdef NDEBUG
  absl::SetFlag(&FLAGS_log_prefix, false);
#endif

  absl::SetProgramUsageMessage("The TensorRT Neural Network Image scaler GRPC Server, version v0.0.1.  Usage:\n");
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  setup_session(true);

  session_using = 0;
  last_done = std::chrono::steady_clock::now();
  LOG(INFO) << "Initialized.";
  std::thread(idle_cleanup).detach();

  NNScalerImpl service;

  auto server_address = absl::GetFlag(FLAGS_listen);
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.SetMaxReceiveMessageSize(128 * 1024 * 1024);
  builder.SetDefaultCompressionAlgorithm(GRPC_COMPRESS_NONE);
  builder.SetResourceQuota(grpc::ResourceQuota().SetMaxThreads(4));
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;
  server->Wait();
}
