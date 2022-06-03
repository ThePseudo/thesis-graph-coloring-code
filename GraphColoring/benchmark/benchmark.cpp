#include "benchmark.h"

std::vector<Benchmark*> Benchmark::_instances = std::vector<Benchmark*>(0);

Benchmark::Benchmark() {
	this->timeMap = new std::unordered_map<int, long long int>();
	this->prev = {};
	this->now = {};

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

double Benchmark::getAvgOfFlag(int flagId) {
	long long int sum = 0.0;

	for (auto& inst : Benchmark::_instances) {
		sum += inst->timeMap->at(flagId);
	}

	return (double)(sum) / Benchmark::_instances.size() / 1000 / 1000;
}

double Benchmark::getAvgOfTotal() {
	long long int sum = 0;

	for (auto& inst : Benchmark::_instances) {
		for (auto& it : *inst->timeMap) {
			sum += it.second;
		}
	}

	return (double)(sum) / Benchmark::_instances.size() / 1000 / 1000;
}

void Benchmark::clear(const int flagId) {
	this->timeMap->emplace(flagId, 0.0);
}

void Benchmark::sampleTime() {
	this->prev = this->now;
	now = std::chrono::steady_clock::now();
}

void Benchmark::sampleTimeToFlag(const int flagId) {
	this->sampleTime();
	this->timeMap->at(flagId) += std::chrono::duration_cast<std::chrono::microseconds>(now - prev).count();
}

double Benchmark::getTimeOfFlag(const int flagId) {
	return (double)(this->timeMap->at(flagId)) / 1000 / 1000;
}

double Benchmark::getTotalTime() {
	long long int sum = 0;
	for (auto& it : *this->timeMap) {
		sum += it.second;
	}
	return (double)(sum) / 1000 / 1000;
}