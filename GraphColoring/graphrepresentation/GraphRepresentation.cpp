#include "GraphRepresentation.h"
#include "benchmark.h"

int GraphRepresentation::nV() const { return _nV; }

size_t GraphRepresentation::nE() const { return _nE; }

int GraphRepresentation::maxD() const { return _maxD; }

int GraphRepresentation::minD() const { return _minD; }

float GraphRepresentation::avgD() const { return _avgD; }

int GraphRepresentation::countNeighs(int v) const {
  if (0 > v || v >= this->nV()) {
    throw std::out_of_range("v index out of range: " + std::to_string(v));
  }

  return this->endNeighs(v) - this->beginNeighs(v);
}

void GraphRepresentation::printGraphRepresentationConfs() {
  std::cout << "Graph Representation: ";
#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX
  std::cout << "Adjacency Matrix";
#endif
#ifdef GRAPH_REPRESENTATION_CSR
  std::cout << "Compressed Sparse Row";
#endif
  std::cout << std::endl;
}

void GraphRepresentation::printGraphInfo() const {
  std::cout << "V: " << this->nV() << ", E: " << this->nE() //<< std::endl
            << ", maxD: " << this->maxD() << ", minD: " << this->minD()
            << ", avgD: " << this->avgD() << std::endl;
}

void GraphRepresentation::printBenchmarkInfo() const {
  Benchmark &bm = *Benchmark::getInstance(0);
  std::cout << "File load:\t\t" << bm.getTimeOfFlag(0) << " s" << std::endl;
}
