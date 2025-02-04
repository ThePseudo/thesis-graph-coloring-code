#ifndef _BENCHMARK_H
#define _BENCHMARK_H

#include <chrono>
#include <vector>
#include <unordered_map>

class Benchmark {
private:
	std::unordered_map<int, long long int>* timeMap;
	std::chrono::steady_clock::time_point prev;
	std::chrono::steady_clock::time_point now;

	static std::vector<Benchmark*> _instances;
	Benchmark();

	static double countToMicroseconds(long long int count);
	static double countToMilliseconds(long long int count);
	static double countToSeconds(long long int count);

public:

	static Benchmark* getInstance(int z);
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