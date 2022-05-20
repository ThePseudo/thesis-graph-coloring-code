#ifdef COLORING_ALGORITHM_GREEDY
#include "Greedy.h"
//#include "configuration.h"

#include "benchmark.h"

#include <algorithm>
#include <fstream>

Greedy::Greedy(std::string const filepath) {
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

	this->col = std::vector<int>(this->adj().nV(), Greedy::INVALID_COLOR);
	this->recolor = std::vector<int>(this->adj().nV());
	for (int i = 0; i < this->adj().nV(); ++i) {
		this->recolor[i] = i;
	}

	this->nConflicts = 0;
	this->nIterations = 0;

	this->MAX_THREADS_SOLVE = std::min(this->adj().nV(), this->MAX_THREADS_SOLVE);
}

const int Greedy::startColoring() {
	Benchmark& bm = *Benchmark::getInstance();
	bm.clear(1);
	bm.clear(2);
#ifdef PARALLEL_GRAPH_COLOR
	bm.clear(3);
#endif

	return this->solve();
}

int Greedy::colorGraph(int n_cols) {
	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();

#ifdef PARALLEL_GRAPH_COLOR
	std::vector<std::thread> threadPool;
	int parallelIdx = 0;
	for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
		threadPool.emplace_back([&, n_cols] { this->colorGraphParallel(n_cols, parallelIdx); });
	}

	for (auto& t : threadPool) {
		t.join();
	}

	n_cols = *std::max_element(this->col.begin(), this->col.end()) + 1;
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
	auto const end = this->recolor.end();
	for (auto it = this->recolor.begin();
		it != end; ++it) {
		n_cols = this->computeVertexColor(*it, n_cols, &this->col[*it]);
	}
#endif

	bm.sampleTimeToFlag(2);

	return n_cols;
}

int Greedy::colorGraphParallel(int n_cols, int& i) {
	this->mutex.lock();
	while (i < this->recolor.size()) {
		int v = this->recolor[i];
		++i;
		this->mutex.unlock();

		n_cols = this->computeVertexColor(v, n_cols, &this->col[v]);

		this->mutex.lock();
	}
	this->mutex.unlock();

	return n_cols;
}

int Greedy::detectConflicts() {
	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();

	this->recolor.clear();
	std::vector<std::thread> threadPool;
	for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
		threadPool.emplace_back([&, i] { this->detectConflictsParallel(i); });
	}

	for (auto& t : threadPool) {
		t.join();
	}

	int recolorSize = this->recolor.size();

	bm.sampleTimeToFlag(3);

	return recolorSize;
}

void Greedy::detectConflictsParallel(const int i) {
	auto const nV = this->adj().nV();
	for (int v = i; v < nV; v += this->MAX_THREADS_SOLVE) {
		if (this->col[v] == Greedy::INVALID_COLOR) {
			this->mutex.lock();
			this->recolor.push_back(v);
			this->mutex.unlock();
			continue;
		}

		auto const end = this->adj().endNeighs(v);
		for (auto neighIt = this->adj().beginNeighs(v);
			neighIt != end;	++neighIt) {
			int w = *neighIt;
			if (v < w) continue;

			if (this->col[v] == this->col[w]) {
				this->mutex.lock();
				this->recolor.push_back(v);
				this->mutex.unlock();
				break;
			}
		}
	}

	return;
}

void Greedy::sortGraphVerts() {
#ifdef SORT_LARGEST_DEGREE_FIRST
	auto sort_lambda = [&](const int v, const int w) { return this->adj().countNeighs(v) > this->adj().countNeighs(w); };
#endif
#ifdef SORT_SMALLEST_DEGREE_FIRST
	auto sort_lambda = [&](const int v, const int w) { return this->adj().countNeighs(v) < this->adj().countNeighs(w); };
#endif
#ifdef SORT_VERTEX_ORDER
	auto sort_lambda = [&](const int v, const int w) { return v < w; };
#endif
#ifdef SORT_VERTEX_ORDER_REVERSED
	auto sort_lambda = [&](const int v, const int w) { return v > w; };
#endif

	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();

	std::sort(this->recolor.begin(), this->recolor.end(), sort_lambda);

	bm.sampleTimeToFlag(1);
}

const int Greedy::solve() {
	int n_cols = 0;

#ifdef SEQUENTIAL_GRAPH_COLOR
	this->sortGraphVerts();
	n_cols = this->colorGraph(n_cols);
#endif
#ifdef PARALLEL_GRAPH_COLOR
#ifdef PARALLEL_RECOLOR
	int partial_confs;
	do {
		this->sortGraphVerts();
		n_cols = this->colorGraph(n_cols);

		++this->nIterations;

		partial_confs = this->detectConflicts();
		this->nConflicts += partial_confs;
	} while (partial_confs > 0);
#endif
#ifdef SEQUENTIAL_RECOLOR
	this->sortGraphVerts();
	n_cols = this->colorGraph(n_cols);
	++this->nIterations;
	this->nConflicts = this->detectConflicts();

	if (this->nConflicts > 0) {
		this->sortGraphVerts();
		int index = 0;

		Benchmark& bm = *Benchmark::getInstance();
		bm.sampleTime();

		n_cols = this->colorGraphParallel(n_cols, index);

		bm.sampleTimeToFlag(1);
		++this->nIterations;
	}
#endif
#endif

	return n_cols;
}

const int Greedy::getConflicts() const {
	return this->nConflicts;
}

const int Greedy::getIterations() const {
	return this->nIterations;
}

void Greedy::printExecutionInfo() const {
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Solution converged to in " << this->getIterations() << " iterations." << std::endl;
	std::cout << "Detected a total of " << this->getConflicts() << " conflicts." << std::endl;
#endif

	return;
}

void Greedy::printBenchmarkInfo() const {
	__super::printBenchmarkInfo();

	Benchmark& bm = *Benchmark::getInstance();
	std::cout << "Vertex sort:\t\t" << bm.getTimeOfFlag(1) << " s" << std::endl;
	std::cout << "Vertex color:\t\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Conflict search:\t" << bm.getTimeOfFlag(3) << " s" << std::endl;
#endif

	std::cout << "Total:\t\t" << bm.getTotalTime() << " s" << std::endl;
}

#endif