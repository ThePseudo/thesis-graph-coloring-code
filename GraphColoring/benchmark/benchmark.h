#ifndef _BENCHMARK_H
#define _BENCHMARK_H

#include <ctime>
#include <vector>
#include <unordered_map>

class Benchmark {
private:
	std::unordered_map<int, clock_t>* timeMap;
	clock_t prev;
	clock_t now;

	static Benchmark* _instance;
	Benchmark();

public:

	static Benchmark* getInstance();
	void clear(const int flagId);
	void sampleTime();
	void sampleTimeToFlag(const int flagId);
	float getTimeOfFlag(const int flagId);
	float getTotalTime();
};

#endif // !_BENCHMARK_H