#ifndef MDVIEW_H_
#define MDVIEW_H_

#include <cassert>
#include <cstdint>
#include <cstring>

#include "cuda/std/array"
#include "cuda/std/utility"

#if defined(__CUDACC__)
#define host_dev __host__ __device__
#define util_attrs __host__ __device__ inline
#else
#define host_dev
#define util_attrs inline
#endif

template<class T, std::size_t DIMS>
struct simple_array : cuda::std::array<T, DIMS> {
  template<std::size_t begin, std::size_t count>
  constexpr util_attrs simple_array<T, count> slice() const noexcept {
    static_assert((begin + count) <= DIMS, "Slice out of range");
    return this->_slice<begin>(std::make_index_sequence<count>{});
  }

  template<size_t... Idx>
  constexpr util_attrs simple_array<T, sizeof...(Idx)> gather() const noexcept {
    return {(*this)[Idx]...};
  }

  template<class A, class... IdxT>
  constexpr util_attrs void gather_from(A arr, IdxT... idx) {
    static_assert(sizeof...(IdxT) == DIMS, "Indices count mismatch");
    *this = {(static_cast<T>(arr[idx]))...};
  };

  constexpr util_attrs simple_array<int64_t, DIMS> widen() const noexcept {
    simple_array<int64_t, DIMS> ret;
    for (size_t i = 0; i < DIMS; ++i) {
      ret[i] = (*this)[i];
    }
    return ret;
  }

  constexpr util_attrs simple_array<int32_t, DIMS> narrow() const noexcept {
    using limit = cuda::std::numeric_limits<int32_t>;
    simple_array<int32_t, DIMS> ret;
    for (size_t i = 0; i < DIMS; ++i) {
      assert((*this)[i] <= limit::max() && (*this)[i] >= limit::min());
      ret[i] = static_cast<int32_t>((*this)[i]);
    }
    return ret;
  }

 private:
  template<std::size_t begin, std::size_t... I>
  constexpr util_attrs simple_array<T, sizeof...(I)> _slice(std::index_sequence<I...>) const noexcept {
    return {(*this)[begin + I]...};
  }
};

namespace std {
template<typename T, size_t DIMS>
struct tuple_size<simple_array<T, DIMS>> : integral_constant<size_t, DIMS> {};

template<size_t Idx, typename T, size_t DIMS>
struct tuple_element<Idx, simple_array<T, DIMS>> {
  using type = T;
};
}// namespace std

template<typename offset_t, std::size_t DIMS = 1>
struct stride_t : simple_array<offset_t, DIMS> {
  static_assert(DIMS != 0, "Can not make empty stride");
  using _base = simple_array<offset_t, DIMS>;

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) <= DIMS, offset_t> offset(Tp... indexes) const noexcept {
    simple_array<offset_t, DIMS> offsets{static_cast<offset_t>(indexes)...};

    offset_t offset = 0;
    for (std::size_t i = 0; i < DIMS; ++i) {
      offset += (*this)[i] * offsets[i];
    }

    return offset;
  }

  constexpr util_attrs simple_array<offset_t, DIMS> indexes(offset_t offset) const noexcept {
    simple_array<offset_t, DIMS> indexes;
    for (std::size_t dim = 0; dim < DIMS; ++dim) {
      assert((*this)[dim] != 0);
      indexes[dim] = offset / (*this)[dim];
      offset = offset % (*this)[dim];
    }

    return indexes;
  }

  template<std::size_t begin, std::size_t count>
  constexpr util_attrs stride_t<offset_t, count> slice() const noexcept {
    return {_base::template slice<begin, count>()};
  }
};

template<typename... Tp, typename = std::enable_if_t<(std::is_convertible_v<Tp, int32_t> && ...) &&
                                                     (sizeof(std::common_type_t<Tp...>) <= 4)>>
host_dev stride_t(Tp...) -> stride_t<int32_t, sizeof...(Tp)>;

template<typename... Tp, typename = std::enable_if_t<(std::is_convertible_v<Tp, int64_t> && ...) &&
                                                     (sizeof(std::common_type_t<Tp...>) > 4)>>
host_dev stride_t(Tp...) -> stride_t<int64_t, sizeof...(Tp)>;

template<typename offset_t, std::size_t DIMS = 1>
struct shape_t : simple_array<offset_t, DIMS> {
  static_assert(DIMS != 0, "Can not make empty shape");
  using _base = simple_array<offset_t, DIMS>;

  template<size_t SDIMS>
  constexpr util_attrs offset_t offset(const shape_t<offset_t, SDIMS> &offsets) const noexcept {
    static_assert(SDIMS <= DIMS, "Too many indices");
    offset_t offset = 0;
    for (std::size_t i = 0; i < DIMS; ++i) {
      offset = offset * (*this)[i];
      if (i < SDIMS) {
        offset += offsets[i];
      }
    }

    return offset;
  }

  template<class... Tp>
  constexpr util_attrs offset_t offset(Tp... indexes) const noexcept {
    static_assert(sizeof...(Tp) <= DIMS, "Too many indices");
    return offset<sizeof...(Tp)>({static_cast<offset_t>(indexes)...});
  }

  constexpr util_attrs simple_array<offset_t, DIMS> indexes(offset_t offset) const noexcept {
    simple_array<offset_t, DIMS> indexes;
    for (std::size_t dim = 0; dim < DIMS; ++dim) {
      auto pos = DIMS - dim - 1;
      indexes[pos] = offset % (*this)[pos];
      offset /= (*this)[pos];
    }

    return indexes;
  }

  constexpr util_attrs offset_t count() const noexcept {
    offset_t size = 1;
    for (const auto &s: *this) {
      size *= s;
    }

    return size;
  }

  template<std::size_t begin, std::size_t count>
  constexpr util_attrs shape_t<offset_t, count> slice() const noexcept {
    return {_base::template slice<begin, count>()};
  }

  template<std::size_t... Idx>
  constexpr util_attrs shape_t<offset_t, sizeof...(Idx)> gather() const noexcept {
    return {_base::template gather<Idx...>()};
  }

  template<typename stride_offset_t = offset_t>
  constexpr util_attrs stride_t<stride_offset_t, DIMS> stride() const noexcept {
    stride_t<stride_offset_t, DIMS> stride;
    stride_offset_t current = 1;

    for (std::size_t dim = 0; dim < DIMS; ++dim) {
      auto pos = DIMS - dim - 1;
      stride[pos] = current;
      current *= (*this)[pos];
    }

    return stride;
  }
};

template<typename... Tp, typename = std::enable_if_t<(std::is_convertible_v<Tp, int32_t> && ...) &&
                                                     (sizeof(std::common_type_t<Tp...>) <= 4)>>
host_dev shape_t(Tp...) -> shape_t<int32_t, sizeof...(Tp)>;

template<typename... Tp, typename = std::enable_if_t<(std::is_convertible_v<Tp, int64_t> && ...) &&
                                                     (sizeof(std::common_type_t<Tp...>) > 4)>>
host_dev shape_t(Tp...) -> shape_t<int64_t, sizeof...(Tp)>;

namespace std {
template<typename offset_t, size_t DIMS>
struct tuple_size<shape_t<offset_t, DIMS>> : integral_constant<size_t, DIMS> {};

template<size_t Idx, typename offset_t, size_t DIMS>
struct tuple_element<Idx, shape_t<offset_t, DIMS>> {
  using type = offset_t;
};
}// namespace std

template<class T_, typename offset_t, std::size_t DIMS = 1>
struct md_view;

template<class T_, typename offset_t, std::size_t DIMS = 1, typename stride_offset_t = offset_t>
struct md_uview;

template<class T_, typename offset_t, std::size_t DIMS>
constexpr md_uview<T_, offset_t, DIMS> to_uview_fwd(md_view<T_, offset_t, DIMS>);

template<class T_, typename offset_t, std::size_t DIMS>
struct md_view {
  using T = std::remove_reference_t<T_>;
  constexpr static std::size_t D = DIMS;

  template<std::size_t NDIMS>
  using of_dim = md_view<T, offset_t, NDIMS>;

  T *data;
  shape_t<offset_t, DIMS> shape;

  // Implicit conversion from md_view<T> to md_view<const T> if T itself is not const
  template<class CT = T>
  constexpr util_attrs
  operator std::enable_if_t<!std::is_const_v<T>, md_view<const CT, offset_t, DIMS>>() const noexcept {
    return {data, shape};
  }

  constexpr util_attrs operator md_uview<T, offset_t, DIMS>() const noexcept { return this->as_uview(); }

  template<class... Tp>
  constexpr util_attrs T *ptr(Tp... indexes) const noexcept {
    return &data[shape.offset(indexes...)];
  }

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) < DIMS, of_dim<DIMS - sizeof...(Tp)>>
  at(Tp... indexes) const noexcept {
    ptrdiff_t offset = shape.offset(indexes...);

    auto sub_span = of_dim<DIMS - sizeof...(Tp)>{data + offset};
    for (std::size_t i = sizeof...(Tp); i < DIMS; ++i) {
      sub_span.shape[i - sizeof...(Tp)] = shape[i];
    }
    return sub_span;
  }

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) == DIMS, T &> at(Tp... indexes) const noexcept {
    return data[shape.offset(indexes...)];
  }

  template<size_t SDIMS, typename = std::enable_if_t<(SDIMS < DIMS)>>//
  constexpr util_attrs of_dim<DIMS - SDIMS> at(const shape_t<offset_t, SDIMS> &offsets) const noexcept {
    ptrdiff_t offset = shape.offset(offsets);

    auto sub_span = of_dim<DIMS - SDIMS>{data + offset};
    for (std::size_t i = SDIMS; i < DIMS; ++i) {
      sub_span.shape[i - SDIMS] = shape[i];
    }
    return sub_span;
  }

  template<size_t SDIMS, typename = std::enable_if_t<SDIMS == DIMS>>
  constexpr util_attrs T &at(const shape_t<offset_t, SDIMS> &offsets) const noexcept {
    return data[shape.offset(offsets)];
  }

  template<std::size_t N_DIMS>
  constexpr util_attrs of_dim<N_DIMS> reshape(shape_t<offset_t, N_DIMS> new_shape) const noexcept {
    return {this->data, new_shape};
  }

  template<class... Tp>
  constexpr util_attrs of_dim<sizeof...(Tp)> reshape(Tp... indexes) const noexcept {
    return {this->data, {static_cast<offset_t>(indexes)...}};
  }

  template<class T2>
  constexpr util_attrs md_view<T2, offset_t, DIMS> reinterpret() const noexcept {
    shape_t new_shape = this->shape;
    new_shape[DIMS - 1] = new_shape[DIMS - 1] * sizeof(T) / sizeof(T2);
    assert(this->shape[DIMS - 1] * sizeof(T) == new_shape[DIMS - 1] * sizeof(T2));
    return {reinterpret_cast<T2 *>(this->data), new_shape};
  }

  template<class T2, std::size_t N_DIMS>
  constexpr util_attrs md_view<T2, offset_t, DIMS> reinterpret(shape_t<offset_t, N_DIMS> new_shape) const noexcept {
    return {reinterpret_cast<T2 *>(this->data), new_shape};
  }

  [[nodiscard]] constexpr util_attrs offset_t size() const noexcept { return this->shape.count(); }

  constexpr util_attrs md_uview<T, offset_t, DIMS> as_uview() const noexcept { return to_uview_fwd(*this); }
  constexpr util_attrs md_uview<T, offset_t, DIMS, int64_t> as_wuview() const noexcept { return to_wuview(*this); }
};

template<typename T, typename... Tp,
         typename =
             std::enable_if_t<(std::is_convertible_v<Tp, int32_t> && ...) && (sizeof(std::common_type_t<Tp...>) <= 4)>>
host_dev md_view(T *, Tp...) -> md_view<T, int32_t, sizeof...(Tp)>;

template<
    typename T, typename... Tp,
    typename = std::enable_if_t<(std::is_convertible_v<Tp, int64_t> && ...) && (sizeof(std::common_type_t<Tp...>) > 4)>>
host_dev md_view(T *, Tp...) -> md_view<T, int64_t, sizeof...(Tp)>;

template<class T, typename offset_t, std::size_t DIMS>
host_dev md_view(T *t, const offset_t (&shape)[DIMS]) -> md_view<T, offset_t, DIMS>;

template<class T, typename offset_t, std::size_t DIMS>
host_dev md_view(T *t, shape_t<offset_t, DIMS> shape) -> md_view<T, offset_t, DIMS>;

template<class T, class... Tp>
constexpr util_attrs md_view<T, int32_t, sizeof...(Tp)> make_view(T *ptr, Tp... shape) {
  return {ptr, {static_cast<int32_t>(shape)...}};
}

template<class T, class T2, class... Tp>
constexpr util_attrs md_view<T, int32_t, sizeof...(Tp)> make_view(T2 *ptr, Tp... shape) {
  return {reinterpret_cast<T *>(ptr), {static_cast<int32_t>(shape)...}};
}

template<class T, typename offset_t, size_t DIMS>
constexpr util_attrs md_view<T, offset_t, DIMS> make_view(T *ptr, shape_t<offset_t, DIMS> shape) {
  return {ptr, shape};
}

template<class T, class T2, typename offset_t, size_t DIMS>
constexpr util_attrs md_view<T, offset_t, DIMS> make_view(T2 *ptr, shape_t<offset_t, DIMS> shape) {
  return {reinterpret_cast<T *>(ptr), shape};
}

template<class T_, typename offset_t, std::size_t DIMS, typename stride_offset_t>
struct md_uview {
  using T = std::remove_reference_t<T_>;
  constexpr static std::size_t D = DIMS;

  template<std::size_t NDIMS>
  using of_dim = md_uview<T, offset_t, NDIMS, stride_offset_t>;

  T *data;
  shape_t<offset_t, DIMS> shape;
  stride_t<stride_offset_t, DIMS> stride;

  template<class CT = T>
  constexpr util_attrs operator std::enable_if_t<!std::is_const_v<T>, md_uview<const CT, offset_t, DIMS, stride_offset_t>>() const {
    return {data, shape, stride};
  }

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) < DIMS, of_dim<DIMS - sizeof...(Tp)>>
  at(Tp... indexes) const noexcept {
    ptrdiff_t offset = stride.offset(indexes...);

    auto sub_span = of_dim<DIMS - sizeof...(Tp)>{data + offset};
    for (std::size_t i = sizeof...(Tp); i < DIMS; ++i) {
      sub_span.shape[i - sizeof...(Tp)] = shape[i];
      sub_span.stride[i - sizeof...(Tp)] = stride[i];
    }
    return sub_span;
  }

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) == DIMS, T &> at(Tp... indexes) const noexcept {
    return data[stride.offset(indexes...)];
  }

  template<class... Tp>
  constexpr util_attrs T* ptr(Tp... indexes) const noexcept {
    return &data[stride.offset(indexes...)];
  }

  [[nodiscard]] constexpr util_attrs offset_t size() const noexcept { return this->shape.count(); }

  template<std::size_t pos, typename = std::enable_if_t<(pos >= 0 && pos < DIMS)>>
  constexpr util_attrs md_uview slice(offset_t begin = 0, offset_t end = 0) const {
    begin = begin < 0 ? begin + this->shape[pos] : begin;
    end = end <= 0 ? end + this->shape[pos] : end;
    assert(begin < end);

    md_uview result = *this;
    result.data += this->stride[pos] * begin;
    result.shape[pos] = end - begin;
    return result;
  }

  constexpr util_attrs bool is_contiguous() const { return this->shape.stride() == this->stride; }

  template<std::size_t N_DIMS>
  constexpr util_attrs of_dim<N_DIMS> reshape(shape_t<offset_t, N_DIMS> new_shape) const {
    return {this->data, new_shape, this->stride};
  }

  template<class T2>
  constexpr util_attrs md_uview<T2, offset_t, DIMS, stride_offset_t> reinterpret() const {
    shape_t new_shape = this->shape;
    new_shape[DIMS - 1] = new_shape[DIMS - 1] * sizeof(T) / sizeof(T2);
    assert(this->shape[DIMS - 1] * sizeof(T) == new_shape[DIMS - 1] * sizeof(T2));

    stride_t new_stride = this->stride;
    for (std::size_t i = 0; i < DIMS; ++i) {
      new_stride[i] = new_stride[i] * sizeof(T) / sizeof(T2);
      assert(this->stride[DIMS - 1] * sizeof(T) == new_stride[DIMS - 1] * sizeof(T2));
    }

    return {reinterpret_cast<T2>(this->data), new_shape, new_stride};
  }

  constexpr util_attrs md_view<T, offset_t, DIMS> as_view() const noexcept {
    assert(this->is_contiguous());
    return to_view(*this);
  }
};

template<class T, typename offset_t, std::size_t DIMS, typename offset_stride_t>
md_uview(T *t, shape_t<offset_t, DIMS> shape, stride_t<offset_stride_t, DIMS> stride)
    -> md_uview<T, offset_t, DIMS, offset_stride_t>;

template<class T_, typename offset_t, std::size_t DIMS>
static constexpr md_uview<T_, offset_t, DIMS, offset_t> util_attrs to_uview(md_view<T_, offset_t, DIMS> v) {
  return {v.data, v.shape, v.shape.stride()};
}

template<class T_, typename offset_t, std::size_t DIMS>
static constexpr md_uview<T_, offset_t, DIMS, int64_t> util_attrs to_wuview(md_view<T_, offset_t, DIMS> v) {
  return {v.data, v.shape, v.shape.template stride<int64_t>()};
}

template<class T_, typename offset_t, std::size_t DIMS, typename stride_offset_t>
static constexpr md_view<T_, offset_t, DIMS> util_attrs to_view(md_uview<T_, offset_t, DIMS, stride_offset_t> uv) {
  return {uv.data, uv.shape};
}

template<class T_, typename offset_t, std::size_t DIMS>
constexpr md_uview<T_, offset_t, DIMS> to_uview_fwd(md_view<T_, offset_t, DIMS> v) {
  return to_uview(v);
}

namespace {
namespace detail {

template<class T, typename offset_t, class Memcpy>
void util_attrs copy_impl(const md_uview<T, offset_t, 1> &dst, const md_uview<const T, offset_t, 1> &src, Memcpy cp) {
  for (int i = 0; i < dst.shape[0]; ++i) {
    cp(&dst.at(i), &src.at(i), sizeof(T));
  }
}

template<class T, typename offset_t, std::size_t DIMS, class Memcpy>
void util_attrs copy_impl(const md_uview<T, offset_t, DIMS> &dst,
                          const md_uview<const T, offset_t, DIMS> &src,
                          Memcpy cp) {
  if (dst.at(0).is_contiguous() && src.at(0).is_contiguous()) {
    for (int i = 0; i < dst.shape[0]; ++i) {
      copy(dst.at(i).as_view(), src.at(i).as_view(), cp);
    }
  }
  else {
    for (int i = 0; i < dst.shape[0]; ++i) {
      copy_impl(dst.at(i), src.at(i), cp);
    }
  }
}

}
}

template<class T, typename offset_t, std::size_t DIMS, class Memcpy>
static void util_attrs copy(const md_view<T, offset_t, DIMS> &dst, const md_view<const T, offset_t, DIMS> &src, Memcpy cp) {
  assert(dst.shape == src.shape);
  cp(dst.data, src.data, dst.size() * sizeof(T));
}

template<class T, typename offset_t, std::size_t DIMS, class Memcpy>
static void util_attrs copy(const md_view<T, offset_t, DIMS> &dst, const md_view<T, offset_t, DIMS> &src, Memcpy cp) {
  md_view<const T, offset_t, DIMS> csrc = src;
  copy(dst, csrc, cp);
}

template<class T, typename offset_t, std::size_t DIMS, class Memcpy>
static void util_attrs copy(const md_view<T, offset_t, DIMS> &dst, const md_view<const T, offset_t, DIMS> &src) {
  copy(dst, src, std::memset);
}

template<class T, typename offset_t, std::size_t DIMS>
static void util_attrs copy(const md_view<T, offset_t, DIMS> &dst, const md_view<T, offset_t, DIMS> &src) {
  md_view<const T, offset_t, DIMS> csrc = src;
  copy(dst, csrc, std::memset);
}

template<class T, typename offset_t, std::size_t DIMS, class Memcpy>
static void util_attrs copy(const md_uview<T, offset_t, DIMS> &dst, const md_uview<const T, offset_t, DIMS> &src, Memcpy cp) {
  detail::copy_impl(dst, src, cp);
}

template<class T, typename offset_t, std::size_t DIMS, class Memcpy>
static void util_attrs copy(const md_uview<T, offset_t, DIMS> &dst, const md_uview<T, offset_t, DIMS> &src, Memcpy cp) {
  md_uview<const T, offset_t, DIMS> csrc = src;
  detail::copy_impl(dst, csrc, cp);
}

template<class T, typename offset_t, std::size_t DIMS>
static void util_attrs copy(const md_uview<T, offset_t, DIMS> &dst, const md_uview<const T, offset_t, DIMS> &src) {
  detail::copy_impl(dst, src, std::memcpy);
}

template<class T, typename offset_t, std::size_t DIMS>
static void util_attrs copy(const md_uview<T, offset_t, DIMS> &dst, const md_uview<T, offset_t, DIMS> &src) {
  md_uview<const T, offset_t, DIMS> csrc = src;
  detail::copy_impl(dst, csrc, std::memcpy);
}


#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

template<typename offset_t, std::size_t DIMS>
static std::string describe(const shape_t<offset_t, DIMS> &view) {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < DIMS; ++i) {
    ss << view[i];
    if (i + 1 != DIMS) {
      ss << ",";
    }
  }
  ss << "]";
  return ss.str();
}

template<class T, typename offset_t, std::size_t DIMS>
static std::string describe(const md_view<T, offset_t, DIMS> &view) {
  std::stringstream ss;
  ss << std::internal << std::hex << std::setw(16) << std::setfill('0') << (void *) (view.data);
  ss << "-" << std::setw(16) << (void *) ((uint8_t *) (view.data) + view.size() * sizeof(T));
  ss << std::resetiosflags(ss.basefield);
  ss << "(" << view.size() * sizeof(T);
  ss << ", ";
  ss << describe(view.shape);
  ss << ")";
  return ss.str();
}

#endif//MDVIEW_H_
