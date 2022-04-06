#ifndef _BARRIER_H
#define _BARRIER_H

#include <mutex>
#include <condition_variable>

class Barrier {
private:
	unsigned int size;
	unsigned int waiting_in;
	unsigned int waiting_out;
	std::mutex* mutex;
	std::condition_variable* cond;
public:
	Barrier(unsigned int size);
	~Barrier();

	void wait();
};

#endif // !_BARRIER_H