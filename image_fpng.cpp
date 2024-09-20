//
// Created by TYTY on 2023-02-10 010.
//

#include <mutex>
#include <fstream>

#include "nn-scaler.h"
#include "fpng.h"

static std::once_flag fpng_inited;

std::string save_image_png(Work::output_t file, md_view<uint8_t, int32_t, 3> data) {
  std::call_once(fpng_inited, fpng::fpng_init);

  auto [height, width, components] = data.shape;
  std::vector<uint8_t> output;
  if (!fpng::fpng_encode_image_to_memory(data.data, width, height, components, output)) {
    return "fpng encode fail";
  }

  if (file.index() == 0) {
    std::ofstream of(std::get<0>(file), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!of.is_open()) {
      return "can't open output file";
    }

    of.write(reinterpret_cast<char *>(output.data()), output.size());
  } else if (file.index() == 1) {
    std::get<1>(file).set_value(std::move(output));
  } else {
    return "unexpected";
  }

  return "";
}
