#pragma once

#include <type_traits>
#include <utility>

#include "cuda/std/array"
#include "cuda/std/utility"

#include <cuda_fp16.h>

#include "md_view.h"

template<size_t... I>
using ConstInt = cuda::std::index_sequence<I...>;

template<size_t NIn_, size_t NOut_, size_t NInt_, size_t NFloat_ = 0, class ConstInt_ = void, size_t NLargeInt_ = 0>
struct LayerIODescriptor {
  constexpr static size_t NIn = NIn_;
  constexpr static size_t NOut = NOut_;
  constexpr static size_t NInt = NInt_;
  constexpr static size_t NFloat = NFloat_;
  constexpr static size_t NLargeInt = NLargeInt_;

  using ConstInt = ConstInt_;
};

template<class desc>
struct LayerIO {
  simple_array<const void *, desc::NIn> in;
  simple_array<void *, desc::NOut> out;
  simple_array<int32_t, desc::NInt> ints;
  simple_array<float, desc::NFloat> floats;
  simple_array<int64_t, desc::NLargeInt> large_ints;

  template<size_t N>
  constexpr util_attrs static size_t const_int() {
    return ([]<size_t... Ints>(cuda::std::index_sequence<Ints...>) {
      static_assert(N < sizeof...(Ints));
      size_t arr[sizeof...(Ints)] = {Ints...};
      return arr[N];
    })(typename desc::ConstInt{});
  }

  template<typename ConstInt_, typename ndesc = LayerIODescriptor<desc::NIn, desc::NOut, desc::NInt, desc::NFloat,
                                                                  ConstInt_, desc::NLargeInt>>
  constexpr LayerIO<ndesc> with_const() {
    LayerIO<ndesc> result;
    static_assert(sizeof(*this) == sizeof(result));
    std::memcpy(&result, this, sizeof(*this));
    return result;
  }

  template<size_t... I, typename ndesc = LayerIODescriptor<desc::NIn, desc::NOut, desc::NInt, desc::NFloat,
                                                           ConstInt<I...>, desc::NLargeInt>>
  constexpr LayerIO<ndesc> with_const() {
    return with_const<ConstInt<I...>>();
  }
};

constexpr util_attrs static size_t get_alignment(size_t s) {
  size_t align = 1;
  while (align != 16 && s % 2 == 0) {
    align <<= 1;
    s >>= 1;
  }

  return align;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700

#include "cuda/pipeline"

template<size_t s>
constexpr util_attrs static auto aligned_size() {
  return cuda::aligned_size_t<get_alignment(s)>(s);
}

#endif

template<typename T, size_t N>
struct __builtin_align__(get_alignment(sizeof(T) * N)) vec_type {
  T v[N];
};

template<typename T>
constexpr util_attrs static void *ptr_offset(void *ptr, ptrdiff_t diff) {
  return reinterpret_cast<T *>(ptr) + diff;
}

template<typename T>
constexpr util_attrs static const void *ptr_offset(const void *ptr, ptrdiff_t diff) {
  return reinterpret_cast<const T *>(ptr) + diff;
}

constexpr util_attrs static void *byte_offset(void *ptr, ptrdiff_t diff) {
  return ptr_offset<uint8_t>(ptr, diff);
}

constexpr util_attrs static const void *byte_offset(const void *ptr, ptrdiff_t diff) {
  return ptr_offset<uint8_t>(ptr, diff);
}

template<typename T, typename T2>
constexpr util_attrs static T ceil_div(T n, T2 d) {
  return (n + d - 1) / d;
}

template<typename T>
constexpr util_attrs static T max(T a, T b) {
  if (a > b) {
    return a;
  }
  return b;
}

template<size_t N, typename E = uint8_t>
constexpr util_attrs static void vec_copy(void *dst, const void *src) {
  constexpr size_t size = N * sizeof(E);
  constexpr size_t align = get_alignment(size);
  using helper = vec_type<uint8_t, align>;
  auto vdst = reinterpret_cast<helper *>(dst);
  auto vsrc = reinterpret_cast<const helper *>(src);
  for (size_t i = 0; i < size / align; i += 1) {
    vdst[i] = vsrc[i];
  }
}

util_attrs static half operator""_h(long double v) {
  return __float2half_rn(float(v));
}

util_attrs static half2 operator""_h2(long double v) {
  return __float2half2_rn(float(v));
}

util_attrs static half operator""_h(unsigned long long int v) {
  return __float2half_rn(float(v));
}

util_attrs static half2 operator""_h2(unsigned long long int v) {
  return __float2half2_rn(float(v));
}

template<auto v>
using ConstCase = std::integral_constant<decltype(v), v>;

template<typename T, typename Filter, T... cases_, typename Fn>
constexpr static bool dispatch_if(T value, Fn fn) {
  auto helper = [&]<T case_>(ConstCase<case_> c) -> bool {
    if constexpr (Filter{}(case_)) {
      if (value == case_) {
        fn(c);
        return true;
      }
    }
    return false;
  };

  return (helper(ConstCase<cases_>{}) || ...);
}

template<typename T, T... cases_, typename Fn>
constexpr static bool dispatch(T value, Fn fn) {
  struct true_type {
    constexpr bool operator()(T) { return true; }
  };
  return dispatch_if<T, true_type, cases_...>(value, fn);
}

template<typename T, int32_t... Sizes>
struct smem_cache_impl : md_view<T, int32_t, sizeof...(Sizes)> {
  util_attrs constexpr static size_t count() {
    cuda::std::array sizes{Sizes...};
    size_t result = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
      result *= sizes[i];
    }
    return result;
  }
  using Storage = vec_type<T, count()>;

  host_dev smem_cache_impl() : md_view<T, int32_t, sizeof...(Sizes)>{{}, {Sizes...}}{};
  util_attrs smem_cache_impl &operator=(Storage& storage) {
    this->data = storage.v;
    return *this;
  }
};

#define make_smem_cache(name, T, ...)                                                                                  \
  smem_cache_impl<T, __VA_ARGS__> name;                                                                                \
  __shared__ typename decltype(name)::Storage name##_Storage;                                                          \
  name = name##_Storage

template<typename T, int32_t... Sizes>
struct rmem_cache : md_view<T, int32_t, sizeof...(Sizes)> {
  util_attrs constexpr static size_t count() {
    cuda::std::array sizes{Sizes...};
    size_t result = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
      result *= sizes[i];
    }
    return result;
  }

  vec_type<T, count()> storage;
  host_dev rmem_cache() : md_view<T, int32_t, sizeof...(Sizes)>{storage.v, {Sizes...}}{};
};
