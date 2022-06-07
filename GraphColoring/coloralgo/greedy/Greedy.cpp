#ifdef COLORING_ALGORITHM_GREEDY
#include "Greedy.h"

#include "benchmark.h"

#include <algorithm>
#include <fstream>

Greedy::Greedy(std::string const filepath) {
	Benchmark& bm = *Benchmark::getInstance(0);

	this->_adj = new GRAPH_REPR_T();

	std::ifstream fileIS;
	fileIS.open(filepath);
	std::istream& is = fileIS;

	bm.sampleTime();
	is >> *this->_adj;
	bm.sampleTimeToFlag(0);

	if (fileIS.is_open()) {
		fileIS.close();
	}

	this->nConflicts = 0;
	this->nIterations = 0;
}

void Greedy::init() {
	__super::init();
	this->MAX_THREADS_SOLVE = std::min(this->adj().nV(), this->MAX_THREADS_SOLVE);
	this->recolor = std::vector<int>(this->adj().nV());
}

void Greedy::reset() {
	__super::reset();

	Benchmark& bm = *Benchmark::getInstance(__super::resetCount);
	bm.sampleTime();

	for (int i = 0; i < this->adj().nV(); ++i) {
		this->recolor[i] = i;
	}

	this->nConflicts = 0;
	this->nIterations = 0;

	bm.sampleTimeToFlag(1);
}

const int Greedy::startColoring() {
	return this->solve();
}

const int Greedy::solve() {
	Benchmark& bm = *Benchmark::getInstance(__super::resetCount);
	int n_cols = 0;

	bm.sampleTime();

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

		n_cols = this->colorGraphParallel(n_cols, index);

		++this->nIterations;
	}
#endif
#endif

	bm.sampleTimeToFlag(2);

	return n_cols;
}

int Greedy::colorGraph(int n_cols) {
#ifdef PARALLEL_GRAPH_COLOR
	std::vector<std::future<int>> threadPool;
	int parallelIdx = 0;
	for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
		threadPool.push_back(std::async(std::launch::async, &Greedy::colorGraphParallel, this, n_cols, std::ref(parallelIdx)));
	}

	for (auto& t : threadPool) {
		n_cols = std::max(n_cols, t.get());
	}

	//n_cols = *std::max_element(this->col.begin(), this->col.end()) + 1;
#endif
#ifdef SEQUENTIAL_GRAPH_COLOR
	auto const end = this->recolor.end();
	for (auto it = this->recolor.begin();
		it != end; ++it) {
		n_cols = this->computeVertexColor(*it, n_cols, &this->col[*it]);
	}
#endif

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
	this->recolor.clear();
	std::vector<std::future<void>> threadPool;
	for (int i = 0; i < this->MAX_THREADS_SOLVE; ++i) {
		threadPool.push_back(std::async(std::launch::async, &Greedy::detectConflictsParallel, this, i));
	}
	for (auto& t : threadPool) {
		t.get();
	}
	int recolorSize = this->recolor.size();

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
	std::sort(this->recolor.begin(), this->recolor.end(), sort_lambda);
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

	Benchmark& bm = *Benchmark::getInstance(__super::resetCount);
	std::cout << "Vertex sort:\t\t" << bm.getTimeOfFlag(1) << " s" << std::endl;
	std::cout << "Vertex color:\t\t" << bm.getTimeOfFlag(2) << " s" << std::endl;
#ifdef PARALLEL_GRAPH_COLOR
	std::cout << "Conflict search:\t" << bm.getTimeOfFlag(3) << " s" << std::endl;
#endif

	std::cout << "Total:\t\t" << bm.getTotalTime() << " s" << std::endl;
}

#endif
