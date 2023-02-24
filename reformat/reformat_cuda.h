#pragma once

#include <string>

#include "cuda_runtime_api.h"

#include "md_view.h"

template<class F, class U>
void import_pixel_cuda(md_view<F, 3> dst,
                       md_view<F, 2> dst_alpha,
                       md_view<const U, 3> src,
                       float a,
                       float b,
                       cudaStream_t stream);

template<class F, class U>
void export_pixel_cuda(md_view<U, 3> dst,
                       md_uview<const F, 3> src,
                       md_uview<const F, 2> src_alpha,
                       float a,
                       float b,
                       cudaStream_t stream);
