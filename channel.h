#pragma once

#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <type_traits>
#include <utility>
#include <optional>

template <typename T>
class channel {
 public:
  void put(T&&);
  std::optional<T> get();
  inline void close() noexcept;

 private:
  enum state_t : uint8_t {
    Empty,
    Filled,
    Closed,
  };

  std::mutex mw_, mh_;
  std::condition_variable wait_, hand_;
  state_t state_{Empty};
  std::optional<T> store_;
};

template<typename T>
void channel<T>::close() noexcept {
  {
    std::lock_guard<std::mutex> lk(mw_);
    assert(state_ != Closed);
    state_ = Closed;
  }

  wait_.notify_all();
  hand_.notify_all();
}


template<typename T>
void channel<T>::put(T && v) {
  std::unique_lock<std::mutex> lw(mw_);
  wait_.wait(lw, [this]() { return state_ != Filled; });

  switch (state_) {
    case Closed:
      assert(false);
      std::terminate();

    case Empty:
      store_ = std::move(v);
      state_ = Filled;
      lw.unlock();
      wait_.notify_one();
      {
        // wait for reader
        std::unique_lock<std::mutex> lh(mh_);
        hand_.wait(lh, [this]() { return !store_; });
        lh.unlock();
      }
      break;

    case Filled:
      assert(false);
  }
}

template<typename T>
std::optional<T> channel<T>::get() {
  std::optional<T> v;

  std::unique_lock<std::mutex> lw(mw_);
  wait_.wait(lw, [this]() { return state_ != Empty; });

  switch (state_) {
    case Closed:
      break;

    case Filled:
    {
      std::lock_guard<std::mutex> lh(mh_);
      v.swap(store_);
      state_ = Empty;
      // notify writer
      hand_.notify_one();
      lw.unlock();
      break;
    }

    case Empty:
      assert(false);
  }

  return v;
}
