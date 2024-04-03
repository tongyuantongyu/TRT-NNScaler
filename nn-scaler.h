#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <variant>
#include <filesystem>
#include <vector>

#include "md_view.h"
#include "channel.h"
#include "timing.h"

constexpr uint32_t MinDimension = 16;

struct Work {
  using input_t = std::variant<std::filesystem::path, std::vector<uint8_t>>;
  using output_t = std::variant<std::filesystem::path, std::promise<std::vector<uint8_t>>>;
  input_t input;
  output_t output;
  std::promise<std::string> submitted;

  std::string alpha_mode;
  double pre_scale, post_scale;
};

typedef channel<Work> chan;

void launch_pipeline(chan& in, chan* out=nullptr);
