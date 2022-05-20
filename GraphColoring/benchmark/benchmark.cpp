#include "benchmark.h"

Benchmark* Benchmark::_instance = nullptr;;

Benchmark::Benchmark() {
	this->timeMap = new std::unordered_map<int, clock_t>();
	this->prev = 0;
	this->now = 0;
}

Benchmark* Benchmark::getInstance() {
	if (Benchmark::_instance == nullptr) {
		Benchmark::_instance = new Benchmark();
	}

	return Benchmark::_instance;
}


void Benchmark::clear(const int flagId) {
	this->timeMap->insert_or_assign(flagId, 0);
}

void Benchmark::sampleTime() {
	this->prev = this->now;
	now = clock();
}

void Benchmark::sampleTimeToFlag(const int flagId) {
	this->sampleTime();
	this->timeMap->at(flagId) += now - prev;
}

const float Benchmark::getTimeOfFlag(const int flagId) {
	return (float)(this->timeMap->at(flagId)) / CLOCKS_PER_SEC;
}

const float Benchmark::getTotalTime() {
	clock_t sum = 0;
	for (auto& it : *this->timeMap) {
		sum += it.second;
	}
	return (float)(sum) / CLOCKS_PER_SEC;
}