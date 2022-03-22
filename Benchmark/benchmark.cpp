#include "benchmark.h"

float loadTime = 0.0f;
float sortTime = 0.0f;
float colorTime = 0.0f;
float conflictsTime = 0.0f;

clock_t prev;
clock_t now;

void sampleTime() {
	prev = now;
	now = clock();
}

float getElapsedTime() {
	return (float)(now - prev) / CLOCKS_PER_SEC;
}