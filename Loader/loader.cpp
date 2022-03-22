#include "benchmark.h"
#include "loader.h"
#include "graph.h"

#include <string>
#include <vector>

#ifdef PARALLEL_INPUT_LOAD
#include <thread>
#endif

bool parseInput(struct graph& G, const char* path) {
	const char* start;
	char* end;
	std::ifstream file;

#ifdef COMPUTE_ELAPSED_TIME
	sampleTime();
#endif

	file.open(path);
	std::string line;
	if (file.is_open()) {
		std::getline(file, line);
		start = line.c_str();
		size_t cap = strtol(start, &end, 10);
		G.adj.resize(cap);
		G.col.resize(cap);
		G.recolor.resize(cap);

		std::fill(G.adj.begin(), G.adj.end(), std::vector<int>());
		std::fill(G.col.begin(), G.col.end(), INVALID_COLOR);

#ifdef PARALLEL_INPUT_LOAD
		std::vector<std::thread> threadPool;
		for (int i = 0; i < MAX_THREADS; ++i) {
			threadPool.emplace_back([&G, &file] { parseInputParallel(G, file); });
		}

		for (auto& t : threadPool) {
			t.join();
		}
#endif
#ifdef SEQUENTIAL_INPUT_LOAD
		while (file.good()) {
			std::getline(file, line);
			start = line.c_str();

			size_t v = static_cast<int>(strtol(start, &end, 10));

			if (v == 0 && start == end) break;

			G.recolor[v] = v;
			++G.nV;

			start = end + 1;
			while (start != end) {
				size_t w = static_cast<int>(strtol(start, &end, 10));

				if (w == 0 && start == end) break;

				G.adj[v].push_back(w);
				++G.nE;

				start = end + 1;
			}
			// Reached end of line
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
	//std::vector<int> neighs;

	G.mutex.lock();
	while (file.good()) {
		std::getline(file, line);
		G.mutex.unlock();

		start = line.c_str();
		size_t v = static_cast<int>(strtol(start, &end, 10));

		if (v == 0 && start == end) {
			G.mutex.lock();
			break;
		}

		G.mutex.lock();
		G.recolor[v] = v;
		++G.nV;
		G.mutex.unlock();

		start = end + 1;

		//neighs.erase(neighs.begin(), neighs.end());
		while (start != end) {
			size_t w = static_cast<int>(strtol(start, &end, 10));

			if (w == 0 && start == end) break;

			// XXX: maybe store neighs in a vector and acquire lock only once at end of line
			//// It seems to be a bad idea. File parsing is faster with acquiring the lock here.
			//neighs.push_back(w);
			G.mutex.lock();
			G.adj[v].push_back(w);
			//G.adj[w].push_back(v);
			G.nE++;
			G.mutex.unlock();

			start = end + 1;
		}
		// Reached end of line
		/*G.mutex.lock();
		for (auto w : neighs) {
			G.adj[v].push_back(w);
			G.adj[w].push_back(v);
		}
		G.nE += neighs.size();*/
		G.mutex.lock();
	}
	G.mutex.unlock();

	return;
}
#endif