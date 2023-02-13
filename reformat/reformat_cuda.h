#pragma once

#include <string>

#include "cuda_runtime_api.h"

#include "md_view.h"

template<class F, class U>
void import_pixel_cuda(md_view<F, 3> dst, md_view<const U, 3> src, float a, float b, cudaStream_t stream);
