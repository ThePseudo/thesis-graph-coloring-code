#include "CusparseColoring.h"

#include "benchmark.h"

#include "cudaKernels.h"

#include <fstream>
#include <set>

CusparseColoring::CusparseColoring(std::string const filepath)
    : ColoringAlgorithm() {
  Benchmark &bm = *Benchmark::getInstance(0);

  this->_adj = new GRAPH_REPR_T();

  std::ifstream fileIS;
  fileIS.open(filepath);
  std::istream &is = fileIS;

  bm.sampleTime();
  is >> *this->_adj;
  bm.sampleTimeToFlag(0);

  if (fileIS.is_open()) {
    fileIS.close();
  }
}

void CusparseColoring::init() { __super::init(); }

void CusparseColoring::reset() { __super::reset(); }

const int CusparseColoring::startColoring() {
#ifdef GRAPH_REPRESENTATION_CSR
  int const n = this->adj().nV();
  const int *Ao = this->adj().getRowPointers();
  const int *Ac = this->adj().getColIndexes();

  int *colors = this->col.data();
  int possibly_wrong_return_value =
      color_cusparse(n, Ao, Ac, colors, __super::resetCount);

  if (possibly_wrong_return_value == -1) { // Error
    return possibly_wrong_return_value;
  }

  std::set<int> colorSet(colors, colors + n);

  return colorSet.size();
#else
  std::cout << "Please enable GRAPH_REPRESENTATION_CSR to use this algorithm."
            << std::endl;
  return -1;
#endif
}

void CusparseColoring::printExecutionInfo() const { return; }

void CusparseColoring::printBenchmarkInfo() const {
  __super::printBenchmarkInfo();

  Benchmark &bm = *Benchmark::getInstance(__super::resetCount);
  std::cout << "TXfer to GPU:\t\t" << bm.getTimeOfFlag(1) << " s" << std::endl;
  std::cout << "Vertex color:\t\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
  std::cout << "TXfer from GPU:\t\t" << bm.getTimeOfFlag(3) << " s"
            << std::endl;

  std::cout << "Total:\t\t" << bm.getTotalTime() << " s" << std::endl;
}
