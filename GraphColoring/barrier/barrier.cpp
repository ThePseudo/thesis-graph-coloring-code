#include "Barrier.h"

Barrier::Barrier(unsigned int size) {
	this->size = size;
	this->waiting_in = 0;
	this->waiting_out = 0;
	this->mutex = new std::mutex();
	this->cond = new std::condition_variable();
}

Barrier::~Barrier() {
	delete this->mutex;
	delete this->cond;
}

void Barrier::wait() {
	std::unique_lock<std::mutex> lk(*this->mutex);
	++this->waiting_in;

	if (this->waiting_in == this->size) {
		this->cond->notify_all();
	} else {
		while (this->waiting_in != this->size) {
			this->cond->wait(lk);
		}
	}

	++this->waiting_out;
	if (this->waiting_out == this->size) {
		this->waiting_in = 0;
		this->waiting_out = 0;
		this->cond->notify_all();
	} else {
		while (this->waiting_out != 0) {
			this->cond->wait(lk);
		}
	}
}