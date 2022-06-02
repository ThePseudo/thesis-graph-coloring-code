#include "benchmark.h"

std::vector<Benchmark*> Benchmark::_instances = std::vector<Benchmark*>(0);

Benchmark::Benchmark() {
	this->timeMap = new std::unordered_map<int, clock_t>();
	this->prev = 0;
	this->now = 0;

	this->clear(0);	// File load
	this->clear(1);	// Preprocess
	this->clear(2);	// Process
	this->clear(3);	// Postprocess
}

Benchmark* Benchmark::getInstance(int z) {
	if (Benchmark::_instances.size() <= z) {
		int old_size = Benchmark::_instances.size();
		int new_size = z + 1;
		Benchmark::_instances.resize(new_size);
		std::fill(Benchmark::_instances.begin() + old_size, Benchmark::_instances.end(), new Benchmark());
	}

	return Benchmark::_instances[z];
}

float Benchmark::getAvgOfFlag(int flagId) {
	clock_t sum = 0;

	for (auto& inst : Benchmark::_instances) {
		sum += inst->timeMap->at(flagId);
	}

	return ((float)sum) / Benchmark::_instances.size() / CLOCKS_PER_SEC;
}

float Benchmark::getAvgOfTotal() {
	clock_t sum = 0;

	for (auto& inst : Benchmark::_instances) {
		for (auto& it : *inst->timeMap) {
			sum += it.second;
		}
	}

	return ((float)sum) / Benchmark::_instances.size() / CLOCKS_PER_SEC;
}

void Benchmark::clear(const int flagId) {
	this->timeMap->emplace(flagId, 0);
}

void Benchmark::sampleTime() {
	this->prev = this->now;
	now = clock();
}

void Benchmark::sampleTimeToFlag(const int flagId) {
	this->sampleTime();
	this->timeMap->at(flagId) += now - prev;
}

float Benchmark::getTimeOfFlag(const int flagId) {
	return (float)(this->timeMap->at(flagId)) / CLOCKS_PER_SEC;
}

float Benchmark::getTotalTime() {
	clock_t sum = 0;
	for (auto& it : *this->timeMap) {
		sum += it.second;
	}
	return (float)(sum) / CLOCKS_PER_SEC;
}