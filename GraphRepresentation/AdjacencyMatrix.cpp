#include "AdjacencyMatrix.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#if defined(PARALLEL_INPUT_LOAD) || defined(PARTITIONED_INPUT_LOAD)
#include <thread>
#endif

#include "benchmark.h"

AdjacencyMatrix::AdjacencyMatrix() : AdjacencyMatrix(0) { }

AdjacencyMatrix::AdjacencyMatrix(int nV) {
	this->adj.reserve(nV);
	this->_nV = 0;
	this->_nE = 0;
}

AdjacencyMatrix::AdjacencyMatrix(const AdjacencyMatrix& to_copy) {
	for (auto& v : this->adj) {
		v.clear();
	}
	this->adj.clear();
	this->adj.reserve(to_copy._nV);
	for (unsigned int i = 0; i < to_copy._nV; ++i) {
		auto& v = this->adj[i];
		auto& w = to_copy.adj[i];
		v.assign(w.begin(), w.end());
	}
	this->_nV = to_copy._nV;
	this->_nE = to_copy._nE;
}

AdjacencyMatrix::~AdjacencyMatrix() { }

const size_t AdjacencyMatrix::nE() const {
	return this->_nE;
}

const int AdjacencyMatrix::nV() const {
	return this->_nV;
}

const bool AdjacencyMatrix::get(int v, int w) const {
	if (0 > v || v >= this->nV()) {
		throw std::out_of_range("v index out of range: " + v);
	}
	if (0 > w || w >= this->nV()) {
		throw std::out_of_range("w index out of range: " + w);
	}

	auto& vNeighs = this->adj[v];
	auto found = std::find(vNeighs.begin(), vNeighs.end(), w);

	return found != vNeighs.end();
}

const ::std::vector<int>::const_iterator AdjacencyMatrix::beginNeighs(int v) const {
	if (0 > v || v >= this->nV()) {
		throw std::out_of_range("v index out of range: " + v);
	}

	return this->adj[v].begin();
}

const ::std::vector<int>::const_iterator AdjacencyMatrix::endNeighs(int v) const {
	if (0 > v || v >= this->nV()) {
		throw std::out_of_range("v index out of range: " + v);
	}

	return this->adj[v].end();
}

std::istream& operator>>(std::istream& is, AdjacencyMatrix& m) {
	if (m.parseInput(is)) {

	}

	return is;
}
//////////////////////////////////////////////////
/////////////////////////////////////////// LOADER
//////////////////////////////////////////////////

#if defined(PARALLEL_INPUT_LOAD) || defined(PARTITIONED_INPUT_LOAD)
const auto MAX_THREADS_LOAD = 3 * std::thread::hardware_concurrency();
#endif

struct header {
	int nV;
};

struct vertex {
	int v{};
	std::vector<int> adj;
};

bool AdjacencyMatrix::parseInput(std::istream& is) {

	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();

	struct header head = {};
	if (!readHeader(is, head)) {
		return false;
	}

	this->adj.resize(head.nV);
	this->_nV = 0;
	this->_nE = 0;
	// G.col.resize(head.nV);
	// G.recolor.resize(head.nV);

	// std::fill(G.adj.begin(), G.adj.end(), std::vector<int>());
	// std::fill(G.col.begin(), G.col.end(), INVALID_COLOR);

#ifdef PARALLEL_INPUT_LOAD
	std::vector<std::thread> threadPool;
	unsigned int nThreads = MAX_THREADS_LOAD;

	nThreads = std::min(head.nV, static_cast<size_t>(nThreads));

	std::mutex m;
	for (int i = 0; i < nThreads; ++i) {
		threadPool.emplace_back([&] { this->parseInputParallel(is, m); });
	}

	for (auto& t : threadPool) {
		t.join();
	}
#endif
#ifdef PARTITIONED_INPUT_LOAD
	// Get file length
	auto const currPos = is.tellg();
	is.seekg(0, std::ios_base::end);
	auto const endPos = is.tellg();
	auto const fileLength = endPos - currPos;
	is.seekg(currPos, std::ios_base::beg);

	// Read file into memory
	std::string fileContents;
	fileContents.reserve(fileLength);
	fileContents.assign(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>());

	// Calculate number of working threads
	int nThreads = MAX_THREADS_LOAD;
	nThreads = std::min(nThreads, head.nV);

	// Calculate rough size of every partition
	// Every thread will analyze a portion of the file roughly this big
	size_t partitionSize = fileLength / nThreads;

	std::vector<std::thread> threadPool;
	std::mutex m;
	for (int i = 0; i < nThreads; ++i) {
		threadPool.emplace_back(
			[&, i, partitionSize] { this->parseInputParallel(fileContents, m, i, partitionSize); }
		);
	}

	for (auto& t : threadPool) {
		t.join();
	}
#endif
#ifdef SEQUENTIAL_INPUT_LOAD
	struct vertex vert;
	for (int cnt = 0; cnt < head.nV && is.good(); ++cnt) {
		vert.adj.erase(vert.adj.begin(), vert.adj.end());
		if (!readVertex(is, vert)) {
			return false;
		}
		//G.recolor[vert.v] = vert.v;
		++this->_nV;

		for (auto w : vert.adj) {
			this->adj[vert.v].push_back(w);
			//G.adj[w].push_back(v);
		}
		this->_nE += vert.adj.size();
	}
#endif

	bm.sampleTimeToFlag(0);

	return true;
}

#ifdef PARALLEL_INPUT_LOAD
void AdjacencyMatrix::parseInputParallel(std::istream& file, std::mutex& mutex) {
	const char* start;
	char* end;
	std::string line;

	struct vertex vert;
	while (true) {
		vert.adj.erase(vert.adj.begin(), vert.adj.end());
		if (!readVertex(file, vert, mutex)) {
			// XXX: Terminate thread with error status
			break;
		}

		//G.recolor[vert.v] = vert.v;
		for (auto w : vert.adj) {
			this->adj[vert.v].push_back(w);
			//G.adj[w].push_back(v);
		}

		mutex.lock();
		++this->_nV;
		this->_nE += vert.adj.size();
		mutex.unlock();
	}

	return;
}
#endif

#ifdef PARTITIONED_INPUT_LOAD
void AdjacencyMatrix::parseInputParallel(std::string& fileContents, std::mutex& mutex, int const i, size_t const partitionSize) {
	// Search start of partition to the right
	// Start of partition = char after next \n
	size_t end = std::min(i * partitionSize + partitionSize, fileContents.length() - 1);
	for (; end > 0 && end < fileContents.length() - 1 && fileContents[end] != '\n'; ++end);
	size_t start;
	for (start = i * partitionSize; start > 0 && start < end && fileContents[start] != '\n'; ++start);
	if (start > 0) {
		++start;
	}
	if (start == fileContents.length()) {
		// This thread is starting at the end
		// Maybe don't bother calling it then
		std::cerr << "Thread " << i << " is starting at the end" << std::endl;
		return;
	}
	if (end < start) {
		// This thread is useless
		// Assigned partition is in the middle of a single row
		// Thread i - 1 will take care of that
		std::cerr << "Partition " << i << " does not encompass a full row" << std::endl;
		return;
	}

	struct vertex vert;
	while (start < end) {
		size_t lineEnd = fileContents.find('\n', start);
		if (lineEnd == std::string::npos) {
			// Not found
			if (fileContents[end] == '#') {
				// Assuming reached eof and file does not end with \n
				lineEnd = end;
			} else {
				// Error in file format
				return;
			}
		}

		_ASSERT(lineEnd <= end);

		std::string line = fileContents.substr(start, lineEnd - start);
		vert.adj.erase(vert.adj.begin(), vert.adj.end());
		if (!readVertex(line, vert)) {
			// An error occurred
			break;
		}

		//G.recolor[vert.v] = vert.v;
		for (auto w : vert.adj) {
			this->adj[vert.v].push_back(w);
			//G.adj[w].push_back(v);
		}

		mutex.lock();
		++this->_nV;
		this->_nE += vert.adj.size();
		mutex.unlock();

		start = lineEnd + 1;
	}

	return;
}
#endif


bool AdjacencyMatrix::readHeader(std::istream& file, struct header& head) {
	std::string line;
	const char* start;
	char* end;

	std::getline(file, line);
	start = line.c_str();
	head.nV = strtol(start, &end, 10);

	return true;
}

#ifdef PARALLEL_INPUT_LOAD
bool AdjacencyMatrix::readVertex(std::istream& file, struct vertex& vert, std::mutex& m) {
#endif
#ifdef PARTITIONED_INPUT_LOAD
bool AdjacencyMatrix::readVertex(std::string & line, struct vertex& vert) {
#endif
#ifdef SEQUENTIAL_INPUT_LOAD
bool AdjacencyMatrix::readVertex(std::istream & file, struct vertex& vert) {
#endif
	const char* start;
	char* end;
#if defined(PARALLEL_INPUT_LOAD) || defined(SEQUENTIAL_INPUT_LOAD)
	std::string line;
#endif

#ifdef PARALLEL_INPUT_LOAD
	m.lock();
#endif
#if defined(PARALLEL_INPUT_LOAD) || defined(SEQUENTIAL_INPUT_LOAD)
	std::getline(file, line);
#endif
#ifdef PARALLEL_INPUT_LOAD
	m.unlock();
#endif
	if (line.empty()) {
		return false;
	}

	start = line.c_str();
	vert.v = static_cast<int>(strtol(start, &end, 10));

	if (vert.v == 0 && start == end) {
		return true;
	}
	start = end + 1;
	while (start != end) {
		int w = static_cast<int>(strtol(start, &end, 10));
		if (w == 0 && start == end) break;
		vert.adj.push_back(w);
		start = end + 1;
	}

	return true;
}