/*
 * profiler.cpp
 *
 *  Created on: Sep 13, 2016
 *      Author: dmarce1
 */

#include "profiler.hpp"
#include <unordered_map>
#include <memory>
#include <string>
#include <mpi/mpi.h>
#include <stack>
#include <map>

static thread_local std::stack<std::string> callstack;
static thread_local real t = 0.0;
std::unordered_map<std::string, std::shared_ptr<real> > map;
std::atomic<int> lock(0);

static void accumulate() {
	const real told = t;
	t = MPI_Wtime();
	if (!callstack.empty()) {
		const std::string& str(callstack.top());
		while (lock++ != 0) {
			lock--;
		}
		auto ptr = map[str];
		lock--;
		real dt = t - told;
		(*ptr) += dt;
	}
}

void profiler_enter(const char* func) {
	accumulate();
	std::string str(func);
	if (callstack.empty()) {
		if (strcmp("OTHER", func) != 0) {
			profiler_enter("OTHER");
		}
	}
	if (map.find(str) == map.end()) {
		while (lock++ != 0) {
			lock--;
		}
		auto cntptr = std::make_shared < real > (0.0);
		std::pair<std::string, std::shared_ptr<real> > entry;
		entry.first = str;
		entry.second = cntptr;
		map.insert(entry);
		lock--;
	}
	callstack.push(str);
}

void profiler_exit() {
	accumulate();
	callstack.pop();
}

void profiler_output(FILE* fp) {
	while (lock++ != 0) {
		lock--;
	}
	std::map<real, std::string> ranks;
	real ttot = 0.0;
	for (auto i = map.begin(); i != map.end(); ++i) {
		real tm = *(i->second);
		ranks[tm] = i->first;
		ttot += tm;
	}
	int r = 1;
	fprintf(fp, "\n");
	for (auto i = ranks.end(); i != ranks.begin(); r <= 10) {
		i--;
		fprintf(fp, "%4i %52s %.2f %%\n", r++, i->second.c_str(), i->first * 100.0 / ttot);
	}
	fprintf(fp, "\n");
	lock--;
}

