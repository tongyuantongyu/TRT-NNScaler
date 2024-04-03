#pragma once

#include <cstring>
#include <unordered_map>
#include <array>
#include <mutex>
#include <vector>

#include "NvInfer.h"

#include "layers.h"

#define enqueue_require(cond)                                                                                          \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      assert(cond);                                                                                                    \
      return 1;                                                                                                        \
    }                                                                                                                  \
  } while (false)

#define check_require(cond)                                                                                            \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      assert(cond);                                                                                                    \
      return false;                                                                                                    \
    }                                                                                                                  \
  } while (false)

#define enqueue_require_set(cond, value)                                                                               \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      assert(cond);                                                                                                    \
      (value) = 1;                                                                                                     \
    }                                                                                                                  \
  } while (false)

namespace plugins {

using namespace nvinfer1;

struct Signature {
  uint16_t hdr;
  int padding_test;
  bool bool_test;
  float float_test;
};

constexpr static Signature signature = {0xfeff, 0x1, true, 1.0f};

struct plugin_field {
  const void *ptr;
  PluginFieldType type;
  int32_t length;
};

struct plugin_fields {
  std::unordered_map<std::string, plugin_field> fields;

  template<typename T, size_t N, typename = std::enable_if_t<N != 0>>
  std::pair<bool, std::array<T, N>> get(const std::string &field) const noexcept {
    auto it = fields.find(field);
    if (it == fields.end()) {
      return {false, {}};
    }

    auto data = it->second;
    if (N > data.length) {
      return {false, {}};
    }

    std::array<T, N> ret;
    std::memcpy(ret.data(), data.ptr, sizeof(T) * N);
    return {true, ret};
  }

  template<typename T, size_t N, typename = std::enable_if_t<N == 0>>
  std::pair<bool, std::vector<T>> get(const std::string &field) const {
    auto it = fields.find(field);
    if (it == fields.end()) {
      return {false, {}};
    }

    auto data = it->second;

    std::vector<T> ret{data.length};
    std::memcpy(ret.data(), data.ptr, sizeof(T) * data.length);
    return {true, std::move(ret)};
  }

  template<typename T>
  std::pair<bool, T> get(const std::string &field) const noexcept {
    auto it = fields.find(field);
    if (it == fields.end()) {
      return {false, {}};
    }

    return {true, *reinterpret_cast<const T *>(it->second.ptr)};
  }
};

template<class T, class C>
class LayerPlugin : public IPluginV2DynamicExt {
  struct Config {
    Signature s;
    C conf;
  };

 public:
  // Need implementation
  LayerPlugin() : config{} {}
  explicit LayerPlugin(C c) : config{c} {}

  int32_t enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                  const void *const *inputs, void *const *outputs, void *workspace,
                  cudaStream_t stream) noexcept override = 0;

  DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
                                nvinfer1::IExprBuilder &exprBuilder) noexcept override = 0;

  bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override = 0;

  // static IPluginV2DynamicExt* create(const std::unordered_map<std::string, plugin_field>&) noexcept

  // Has default applicable to lots of cases. Can be override if necessary.
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override{};

  void destroy() noexcept override{};

  [[nodiscard]] int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes,
                                       int32_t nbInputs) const noexcept override {
    return inputTypes[0];
  }

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override {
    return 0;
  };

  int initialize() noexcept override { return 0; };

  void terminate() noexcept override{};

  // Boilerplate.
  void attachToContext(cudnnContext *cudnn, cublasContext *cublas, IGpuAllocator *gpu) noexcept override {
    cudnn_context = cudnn;
    cublas_context = cublas;
    gpu_allocator = gpu;
  };

  [[nodiscard]] IPluginV2DynamicExt *clone() const noexcept override {
    LayerPlugin *p = new T(config);
    p->setPluginNamespace(getPluginNamespace());
    return p;
  }

  void detachFromContext() noexcept override {
    cudnn_context = nullptr;
    cublas_context = nullptr;
    gpu_allocator = nullptr;
  };

  [[nodiscard]] const char *getPluginNamespace() const noexcept override { return this->plugin_namespace.c_str(); };

  [[nodiscard]] const char *getPluginType() const noexcept override { return T::plugin_name; };

  [[nodiscard]] const char *getPluginVersion() const noexcept override { return "1"; };

  [[nodiscard]] size_t getSerializationSize() const noexcept override { return sizeof(Config); }

  void serialize(void *buffer) const noexcept override {
    Config c{signature, config};
    std::memcpy(buffer, &c, sizeof(c));
  }

  void setPluginNamespace(const char *pluginNamespace) noexcept override { this->plugin_namespace = pluginNamespace; };

  // Extras
  static LayerPlugin *deserialize(const void *data, size_t size) noexcept {
    if (size != sizeof(Config)) {
      return nullptr;
    }

    Config c{};
    std::memcpy(&c, data, size);
    if (std::memcmp(&c.s, &signature, sizeof(signature)) != 0) {
      return nullptr;
    }

    return new (std::nothrow) T(c.conf);
  }

  static PluginFieldCollection make_fc() noexcept { return {sizeof(T::fields) / sizeof(PluginField), T::fields}; }

  class Creator : public IPluginCreator {
   public:
    // boilerplate
    Creator() noexcept = default;
    ~Creator() noexcept override = default;

    [[nodiscard]] const char *getPluginName() const noexcept override { return T::plugin_name; };

    [[nodiscard]] const char *getPluginVersion() const noexcept override { return "1"; };

    const PluginFieldCollection *getFieldNames() noexcept override { return &T::fc; }

    IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
      std::unordered_map<std::string, plugin_field> fields;
      for (int32_t i = 0; i < fc->nbFields; ++i) {
        auto &field = fc->fields[i];
        fields[field.name] = {field.data, field.type, field.length};
      }

      IPluginV2DynamicExt *p = T::create(plugin_fields{std::move(fields)});
      if (p != nullptr) {
        p->setPluginNamespace(plugin_namespace.c_str());
      }
      return p;
    }

    IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData,
                                           size_t serialLength) noexcept override {
      IPluginV2DynamicExt *p = LayerPlugin::deserialize(serialData, serialLength);
      if (p != nullptr) {
        p->setPluginNamespace(plugin_namespace.c_str());
      }
      return p;
    }

    void setPluginNamespace(const char *libNamespace) noexcept override { plugin_namespace = libNamespace; }

    [[nodiscard]] const char *getPluginNamespace() const noexcept override { return plugin_namespace.c_str(); }

   private:
    std::string plugin_namespace;
  };

  struct Registrar {
    Registrar() noexcept {
#ifdef AUTO_REGISTER_PLUGIN
      register_plugin();
#endif
    }

    void register_plugin() noexcept { std::call_once(once, &Registrar::_register, this); }

   private:
    void _register() noexcept { getPluginRegistry()->registerCreator(instance, ""); }
    std::once_flag once;
    Creator instance{};
  };

 protected:
  std::string plugin_namespace;
  cudnnContext *cudnn_context = nullptr;
  cublasContext *cublas_context = nullptr;
  IGpuAllocator *gpu_allocator = nullptr;

  C config;
};

#if 0

template<class T, class C>
class LayerPluginV3 : public IPluginV3OneCore, IPluginV3OneBuild, IPluginV3OneRuntime {
  struct Config {
    Signature s;
    C conf;
  };

 public:
  // Need implementation
  LayerPluginV3() : config{} {}
  explicit LayerPluginV3(C c) : config{c} {}

//  int32_t enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
//                  const void *const *inputs, void *const *outputs, void *workspace,
//                  cudaStream_t stream) noexcept override = 0;
//
//  DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
//                                nvinfer1::IExprBuilder &exprBuilder) noexcept override = 0;
//
//  bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs,
//                                 int32_t nbOutputs) noexcept override = 0;
//
//  // static IPluginV2DynamicExt* create(const std::unordered_map<std::string, plugin_field>&) noexcept
//
//  // Has default applicable to lots of cases. Can be override if necessary.
//  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
//                       const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override{};
//
//  void destroy() noexcept override{};
//
//  [[nodiscard]] int getNbOutputs() const noexcept override { return 1; }
//
//  nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes,
//                                       int32_t nbInputs) const noexcept override {
//    return inputTypes[0];
//  }
//
//  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
//                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override {
//    return 0;
//  };
//
//  int initialize() noexcept override { return 0; };
//
//  void terminate() noexcept override{};
//
//  // Boilerplate.
//  void attachToContext(cudnnContext *cudnn, cublasContext *cublas, IGpuAllocator *gpu) noexcept override {
//    cudnn_context = cudnn;
//    cublas_context = cublas;
//    gpu_allocator = gpu;
//  };
//
//  [[nodiscard]] IPluginV2DynamicExt *clone() const noexcept override {
//    LayerPluginV3 *p = new T(config);
//    p->setPluginNamespace(getPluginNamespace());
//    return p;
//  }
//
//  void detachFromContext() noexcept override {
//    cudnn_context = nullptr;
//    cublas_context = nullptr;
//    gpu_allocator = nullptr;
//  };
//


  // IPluginV3OneCore
  [[nodiscard]] const char *getPluginName() const noexcept override { return T::plugin_name; };

  [[nodiscard]] const char *getPluginVersion() const noexcept override { return "1"; };

  [[nodiscard]] const char *getPluginNamespace() const noexcept override { return this->plugin_namespace.c_str(); };

//  [[nodiscard]] size_t getSerializationSize() const noexcept override { return sizeof(Config); }
//
//  void serialize(void *buffer) const noexcept override {
//    Config c{signature, config};
//    std::memcpy(buffer, &c, sizeof(c));
//  }
//
  // Extra support
  void setPluginNamespace(const char *pluginNamespace) noexcept { this->plugin_namespace = pluginNamespace; };

  // Extras
  static LayerPluginV3 *deserialize(const void *data, size_t size) noexcept {
    if (size != sizeof(Config)) {
      return nullptr;
    }

    Config c{};
    std::memcpy(&c, data, size);
    if (std::memcmp(&c.s, &signature, sizeof(signature)) != 0) {
      return nullptr;
    }

    return new (std::nothrow) T(c.conf);
  }

  static PluginFieldCollection make_fc() noexcept { return {sizeof(T::fields) / sizeof(PluginField), T::fields}; }

  class Creator : public IPluginCreator {
   public:
    // boilerplate
    Creator() noexcept = default;
    ~Creator() noexcept override = default;

    [[nodiscard]] const char *getPluginName() const noexcept override { return T::plugin_name; };

    [[nodiscard]] const char *getPluginVersion() const noexcept override { return "1"; };

    const PluginFieldCollection *getFieldNames() noexcept override { return &T::fc; }

    IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
      std::unordered_map<std::string, plugin_field> fields;
      for (int32_t i = 0; i < fc->nbFields; ++i) {
        auto &field = fc->fields[i];
        fields[field.name] = {field.data, field.type, field.length};
      }

      IPluginV2DynamicExt *p = T::create(plugin_fields{std::move(fields)});
      if (p != nullptr) {
        p->setPluginNamespace(plugin_namespace.c_str());
      }
      return p;
    }

    IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData,
                                           size_t serialLength) noexcept override {
      IPluginV2DynamicExt *p = LayerPluginV3::deserialize(serialData, serialLength);
      if (p != nullptr) {
        p->setPluginNamespace(plugin_namespace.c_str());
      }
      return p;
    }

    void setPluginNamespace(const char *libNamespace) noexcept override { plugin_namespace = libNamespace; }

    [[nodiscard]] const char *getPluginNamespace() const noexcept override { return plugin_namespace.c_str(); }

   private:
    std::string plugin_namespace;
  };

  struct Registrar {
    Registrar() noexcept {
#ifdef AUTO_REGISTER_PLUGIN
      register_plugin();
#endif
    }

    void register_plugin() noexcept { std::call_once(once, &Registrar::_register, this); }

   private:
    void _register() noexcept { getPluginRegistry()->registerCreator(instance, ""); }
    std::once_flag once;
    Creator instance{};
  };

 protected:
  std::string plugin_namespace;

  C config;
};

#endif

}// namespace sd_plugin
