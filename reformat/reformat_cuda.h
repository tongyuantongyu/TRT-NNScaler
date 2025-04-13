#pragma once

#include <string>

#include "cuda_runtime_api.h"

#include "md_view.h"

template<typename T, size_t DIMS = 3>
using patch = md_uview<T, int32_t, DIMS, int64_t>;

template<class F, class U>
void import_pixel_cuda(patch<F> dst,
                       patch<const U> src,
                       float a,
                       float b,
                       bool is_alpha,
                       cudaStream_t stream);

template<class F, class U>
void export_pixel_cuda(patch<U> dst,
                       patch<const F> src,
                       float a,
                       float b,
                       float l,
                       float h,
                       bool is_alpha,
                       cudaStream_t stream);