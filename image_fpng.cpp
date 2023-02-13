//
// Created by TYTY on 2023-02-10 010.
//

#include <mutex>
#include <fstream>

#include "nn-scaler.h"
#include "fpng/fpng.h"

static std::once_flag fpng_inited;

std::string save_image_png(const std::filesystem::path& file, md_view<uint8_t, 3> data) {
  std::call_once(fpng_inited, fpng::fpng_init);

  std::ofstream of(file, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!of.is_open()) {
    return "can't open output file";
  }

  auto [height, width, components] = data.shape;
  std::vector<uint8_t> output;
  if (!fpng::fpng_encode_image_to_memory(data.data, width, height, components, output)) {
    return "fpng encode fail";
  };

  of.write(reinterpret_cast<char *>(output.data()), output.size());

  return "";
}
