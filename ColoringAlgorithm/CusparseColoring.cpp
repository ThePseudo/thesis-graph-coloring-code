#include "CusparseColoring.h"

#ifdef COMPUTE_ELAPSED_TIME
#include "benchmark.h"
#endif

#include "cudaKernels.h"

#include <fstream>

CusparseColoring::CusparseColoring(std::string const filepath) {
#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(0);
#endif

	this->_adj = new GRAPH_REPR_T();

	std::ifstream fileIS;
	fileIS.open(filepath);
	std::istream& is = fileIS;
	is >> *this->_adj;

	if (fileIS.is_open()) {
		fileIS.close();
	}

	this->col = std::vector<int>(this->adj().nV(), CusparseColoring::INVALID_COLOR);
}

const int CusparseColoring::startColoring() {
#ifdef COMPUTE_ELAPSED_TIME
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(1);
	bm.clear(2);
	bm.clear(3);
#endif
	
#ifdef GRAPH_REPRESENTATION_CSR
	int const n = this->adj().nV();
	const int* Ao = this->adj().getRowPointers();
	const int* Ac = this->adj().getColIndexes();
	int* colors = this->col.data();
	return color_cusparse(n, Ao, Ac, colors);
#else
	std::cout << "Please enable GRAPH_REPRESENTATION_CSR to use this algorithm." << std::endl;
	return -1;
#endif
}