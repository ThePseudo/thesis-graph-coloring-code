#include "CusparseColoring.h"

#include "benchmark.h"

#include "cudaKernels.h"

#include <fstream>

CusparseColoring::CusparseColoring(std::string const filepath) {
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(0);

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
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(1);
	bm.clear(2);
	bm.clear(3);
	
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

void CusparseColoring::printExecutionInfo() const {
	return;
}

void CusparseColoring::printBenchmarkInfo() const {
	__super::printBenchmarkInfo();

	Benchmark& bm = *Benchmark::getInstance();
	std::cout << "TXfer to GPU:\t\t" << bm.getTimeOfFlag(1) << " s" << std::endl;
	std::cout << "Vertex color:\t\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
	std::cout << "TXfer from GPU:\t\t" << bm.getTimeOfFlag(3) << " s" << std::endl;

	std::cout << "Total:\t\t" << bm.getTotalTime() << " s" << std::endl;
}