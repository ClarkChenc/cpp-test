#include <chrono>

int64_t time_cost(const std::chrono::steady_clock::time_point& st,
                  const std::chrono::steady_clock::time_point& en) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(en - st).count();
}
