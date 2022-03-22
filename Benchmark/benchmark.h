#ifndef _BENCHMARK_H
#define _BENCHMARK_H

#include <ctime>

extern float loadTime;
extern float sortTime;
extern float colorTime;
extern float conflictsTime;

void sampleTime();
float getElapsedTime();

#endif // !_BENCHMARK_H