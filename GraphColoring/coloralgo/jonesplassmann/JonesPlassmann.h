#ifndef _JONES_PLASSMANN_H
#define _JONES_PLASSMANN_H

#include <cstdlib>
#include <ctime>

// #include "configuration.h"

#include <set>
#include <vector>

#ifdef PARALLEL_GRAPH_COLOR
#include <atomic>
#include <mutex>
#include <thread>

#include "barrier.h"
#endif

#include "ColoringAlgorithm.h"
#include "GraphRepresentation.h"

class JonesPlassmann : public ColoringAlgorithm {

  define_super(ColoringAlgorithm);

private:
  std::vector<int> vWeights;
  int nIterations;

#ifdef SEQUENTIAL_GRAPH_COLOR
  std::vector<int> nWaits;
#endif
#ifdef PARALLEL_GRAPH_COLOR
  std::mutex mutex;
  std::vector<std::atomic_int> nWaits;
  std::vector<std::pair<int, int>> firstAndLasts;
  std::vector<int> n_colors;
  Barrier *barrier;

  void partitionVertices();
  void partitionVerticesEqually();
  void partitionVerticesByEdges();
#endif

  const int solve();
  void coloringHeuristic(int const first, int const last, int &n_cols);
  void calcWaitTime(int const first, int const last);
  void colorWhileWaiting(int const first, int const last, int &n_cols);

public:
#ifdef PARALLEL_GRAPH_COLOR
  int MAX_THREADS_SOLVE = std::thread::hardware_concurrency();
#endif

  JonesPlassmann(std::string const filepath);

  void init() override;
  void reset() override;
  const int startColoring() override;

  void printExecutionInfo() const override;
  void printBenchmarkInfo() const override;

  const int getIterations() const;

  const int colorWithCuda();
};

#endif // !_JONES_PLASSMANN_H
