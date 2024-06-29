//
// Created by TYTY on 2024-04-01 001.
//

#include "md_view.h"
#include "utils.h"
#include "plugin_base.h"

// N C H W -> N C 2H 2W

namespace plugins {

template<class LayerIODescriptor>
__global__ void ResizeKernelInt8x4(LayerIO<LayerIODescriptor> io) {
  auto [nc, h, w] = io.ints;

  md_view input = make_view<const uint32_t>(io.in[0], nc, h, w);
  md_view output = make_view<uint2>(io.out[0], nc, 2*h, w);

  for (int n = 0; n < nc; n += gridDim.x) {
    int cur_n = n + blockIdx.x;
    for (int y = 0; y < h; y += blockDim.y) {
      int cur_y = y + threadIdx.y;
      for (int x = 0; x < w; x += blockDim.x) {
        int cur_x = x + threadIdx.x;
        if (cur_n < nc && cur_y < h && cur_x < w) {
          uint32_t data = input.at(cur_n, cur_y, cur_x);
          uint2 data2 = {data, data};
          output.at(cur_n, 2 * cur_y + 0, cur_x) = data2;
          output.at(cur_n, 2 * cur_y + 1, cur_x) = data2;
        }
      }
    }
  }
}

struct ResizeConfig {
  int block_x;
  int thread_y;
  int thread_x;
};

struct ResizePlugin : LayerPlugin<ResizePlugin, ResizeConfig> {
  constexpr static const char *plugin_name = "Int8x4ResizePlugin";

  const static PluginField fields[1];
  const static PluginFieldCollection fc;

  static IPluginV2DynamicExt *create(const plugin_fields &) noexcept {
    int device = 0;
    cudaDeviceProp props {};

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    auto thread_per_block = props.maxThreadsPerMultiProcessor;
    auto block_per_sm = 1;
    if (thread_per_block > props.maxThreadsPerBlock) {
      thread_per_block /= 2;
      block_per_sm = 2;
    }

    assert(thread_per_block <= 1024);
    assert(thread_per_block % 32 == 0);

    return new (std::nothrow) ResizePlugin{
        {
            props.multiProcessorCount * block_per_sm,
            thread_per_block / 32,
            32
        }
    };
  }

  ResizePlugin() : LayerPlugin{} {};
  explicit ResizePlugin(ResizeConfig c) : LayerPlugin{c} {};

  DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
                                nvinfer1::IExprBuilder &exprBuilder) noexcept override {
    assert(nbInputs == 1);
    assert(outputIndex == 0);

    auto in = inputs[0];
    assert(in.nbDims == 4);

    return {
        4,
        {in.d[0], in.d[1],
         exprBuilder.operation(DimensionOperation::kPROD, *in.d[2], *exprBuilder.constant(2)),
         exprBuilder.operation(DimensionOperation::kPROD, *in.d[3], *exprBuilder.constant(2))
        }
    };
  }

  bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    auto *in = &inOut[0];
    auto *out = &inOut[nbInputs];

    switch (pos) {
      case 0: {
        auto cur = in[0];
        if (cur.format != nvinfer1::TensorFormat::kCHW4) {
          return false;
        }

        if (cur.type != nvinfer1::DataType::kINT8) {
          return false;
        }

        return true;
      }
      case 1: {
        auto cur = out[0];

        if (cur.format != nvinfer1::TensorFormat::kCHW4) {
          return false;
        }

        if (cur.type != in[0].type) {
          return false;
        }

        return true;
      }
      default: check_require(false);
    }
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                  const void *const *inputs, void *const *outputs, void *workspace,
                  cudaStream_t stream) noexcept override {
    enqueue_require(inputDesc[0].type == nvinfer1::DataType::kINT8);
    enqueue_require(outputDesc[0].type == inputDesc[0].type);
    enqueue_require(outputDesc[0].scale == inputDesc[0].scale);

    enqueue_require(inputDesc[0].format == nvinfer1::TensorFormat::kCHW4);
    enqueue_require(outputDesc[0].format == nvinfer1::TensorFormat::kCHW4);

    enqueue_require(inputDesc[0].dims.nbDims == 4);
    int32_t n = inputDesc[0].dims.d[0];
    int32_t c = inputDesc[0].dims.d[1];
    int32_t h = inputDesc[0].dims.d[2];
    int32_t w = inputDesc[0].dims.d[3];

    enqueue_require(outputDesc[0].dims.nbDims == 4);
    enqueue_require(outputDesc[0].dims.d[0] == n);
    enqueue_require(outputDesc[0].dims.d[1] == c);
    enqueue_require(outputDesc[0].dims.d[2] == 2 * h);
    enqueue_require(outputDesc[0].dims.d[3] == 2 * w);

    int32_t equiv_c = (c + 3) / 4 * n;

    LayerIO<LayerIODescriptor<1, 1, 3, 0>> io{{inputs[0]}, {outputs[0]}, {equiv_c, h, w}};

    dim3 thread, block;
    thread.x = config.thread_x;
    thread.y = config.thread_y;
    block.x = std::min(config.block_x, equiv_c);

    ResizeKernelInt8x4<<<block, thread, 0, stream>>>(io);

    return 0;
  }
};

const PluginField ResizePlugin::fields[] = {};
const PluginFieldCollection ResizePlugin::fc = {};

[[maybe_unused]] static ResizePlugin::Registrar Registrar;

void register_resize_plugin() {
  Registrar.register_plugin();
}

}// namespace sd_plugin