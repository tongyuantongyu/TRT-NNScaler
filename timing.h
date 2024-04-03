#pragma once

#include <chrono>

using ms = std::chrono::duration<double, std::ratio<1, 1000>>;
using hr_clock = std::chrono::high_resolution_clock;

static double elapsed(hr_clock::time_point start) {
  return static_cast<ms>(hr_clock::now() - start).count();
}