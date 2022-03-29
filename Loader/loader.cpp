#include "benchmark.h"
#include "loader.h"
#include "GM.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include <iostream>

#if defined(PARALLEL_INPUT_LOAD) || defined(PARTITIONED_INPUT_LOAD)
#include <thread>
#endif

struct header {
	size_t nV;
};

struct vertex {
	size_t v;
	std::vector<size_t> adj;
};

bool readHeader(std::ifstream&, struct header&);


#ifdef PARALLEL_INPUT_LOAD
bool readVertex(std::ifstream&, struct vertex&, std::mutex&);
#endif
#ifdef PARTITIONED_INPUT_LOAD
bool readVertex(std::string&, struct vertex&);
#endif
#ifdef SEQUENTIAL_INPUT_LOAD
	bool readVertex(std::ifstream&, struct vertex&);
#endif

bool parseInput(struct graph& G, const char* path) {
	std::ifstream file;

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

	file.open(path);
	if (file.is_open()) {
		
		struct header head = {};
		if (!readHeader(file, head)) {
			file.close();
			return false;
		}
		G.adj.resize(head.nV);
		G.col.resize(head.nV);
		G.recolor.resize(head.nV);

		std::fill(G.adj.begin(), G.adj.end(), std::vector<int>());
		std::fill(G.col.begin(), G.col.end(), INVALID_COLOR);

#ifdef PARALLEL_INPUT_LOAD
		std::vector<std::thread> threadPool;
		unsigned int nThreads = MAX_THREADS;

		nThreads = std::min(head.nV, static_cast<size_t>(nThreads));
		for (int i = 0; i < nThreads; ++i) {
			threadPool.emplace_back([&G, &file] { parseInputParallel(G, file); });
		}

		for (auto& t : threadPool) {
			t.join();
		}
#endif
#ifdef PARTITIONED_INPUT_LOAD
		// Get file length
		int const currPos = file.tellg();
		file.seekg(0, std::ios_base::end);
		int const endPos = file.tellg();
		int const fileLength = endPos - currPos;
		file.seekg(currPos, std::ios_base::beg);

		// Read file into memory
		std::string fileContents;
		fileContents.reserve(fileLength);
		fileContents.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

		// Calculate number of working threads
		unsigned int nThreads = 3 * MAX_THREADS;
		nThreads = std::min(head.nV, static_cast<size_t>(nThreads));

		// Calculate rough size of every partition
		// Every thread will analyze a portion of the file roughly this big
		int partitionSize = fileLength / nThreads;

		std::vector<std::thread> threadPool;
		for (int i = 0; i < nThreads; ++i) {
			threadPool.emplace_back(
				[&, i, partitionSize] { parseInputParallel(G, fileContents, i, partitionSize); }
			);
		}

		for (auto& t : threadPool) {
			t.join();
		}
#endif
#ifdef SEQUENTIAL_INPUT_LOAD
		struct vertex vert;
		for (int cnt = 0; cnt < head.nV && file.good(); ++cnt) {
			vert.adj.erase(vert.adj.begin(), vert.adj.end());
			if (!readVertex(file, vert)) {
				file.close();
				return false;
			}
			G.recolor[vert.v] = vert.v;
			++G.nV;

			for (auto w : vert.adj) {
				G.adj[vert.v].push_back(w);
				//G.adj[w].push_back(v);
			}
			G.nE += vert.adj.size();
		}
#endif
	} else {
		file.close();
		return false;
	}

	file.close();

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
	loadTime += getElapsedTime();
#endif

	return true;
}

#ifdef PARALLEL_INPUT_LOAD
void parseInputParallel(struct graph& G, std::ifstream& file) {
	const char* start;
	char* end;
	std::string line;

	struct vertex vert;
	while (true) {
		vert.adj.erase(vert.adj.begin(), vert.adj.end());
		if (!readVertex(file, vert, G.mutex)) {
			// XXX: Terminate thread with error status
			break;
		}

		G.recolor[vert.v] = vert.v;
		for (auto w : vert.adj) {
			G.adj[vert.v].push_back(w);
			//G.adj[w].push_back(v);
		}

		G.mutex.lock();
		++G.nV;
		G.nE += vert.adj.size();
		G.mutex.unlock();
	}
	
	return;
}
#endif

#ifdef PARTITIONED_INPUT_LOAD
void parseInputParallel(struct graph& G, std::string& const fileContents, int const i, size_t const partitionSize) {
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

		G.recolor[vert.v] = vert.v;
		for (auto w : vert.adj) {
			G.adj[vert.v].push_back(w);
			//G.adj[w].push_back(v);
		}

		G.mutex.lock();
		++G.nV;
		G.nE += vert.adj.size();
		G.mutex.unlock();

		start = lineEnd + 1;
	}

	return;
}
#endif


bool readHeader(std::ifstream& file, struct header& head) {
	std::string line;
	const char* start;
	char* end;

	std::getline(file, line);
	start = line.c_str();
	head.nV = strtol(start, &end, 10);

	return true;
}

#ifdef PARALLEL_INPUT_LOAD
bool readVertex(std::ifstream& file, struct vertex& vert, std::mutex& m) {
#endif
#ifdef PARTITIONED_INPUT_LOAD
	bool readVertex(std::string& line, struct vertex& vert) {
#endif
#ifdef SEQUENTIAL_INPUT_LOAD
bool readVertex(std::ifstream & file, struct vertex& vert) {
#endif
	const char* start;
	char* end;

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
		size_t w = static_cast<int>(strtol(start, &end, 10));
		if (w == 0 && start == end) break;
		vert.adj.push_back(w);
		start = end + 1;
	}

	return true;
}