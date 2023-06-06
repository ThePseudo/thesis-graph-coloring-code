#ifndef _BENCHMARK_H
#define _BENCHMARK_H

#include <array>
#include <chrono>
#include <unordered_map>
#include <vector>

class Benchmark {
private:
  std::array<double, 5> timeMap;
  std::chrono::high_resolution_clock::time_point prev;
  std::chrono::high_resolution_clock::time_point now;

  static std::vector<Benchmark *> _instances;
  Benchmark();

  static double countToMicroseconds(long long int count);
  static double countToMilliseconds(long long int count);
  static double countToSeconds(long long int count);

public:
  static Benchmark *getInstance(int z);
  static double getAvgOfFlag(int flagId);
  static double getAvgOfTotal();
  static double getEffectiveAvg();

  void clear(const int flagId);
  void sampleTime();
  void sampleTimeToFlag(const int flagId);
  double getTimeOfFlag(const int flagId);
  double getTotalTime();
};

#endif // !_BENCHMARK_H
