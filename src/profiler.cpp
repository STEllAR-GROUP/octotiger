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
#include <cstring>
#include <stack>
#include <map>

#include <hpx/util/high_resolution_clock.hpp>

static thread_local std::stack<std::string> callstack;
static thread_local real t = 0.0;
std::unordered_map<std::string, std::shared_ptr<real> > map;

std::string make_name(const char* f, int l) {
	std::string str = f;
	str += "+";
	str += std::string(std::to_string(l));
	return str;
}

std::atomic<int>& lock() {
	static std::atomic<int> a(0);
	return a;
}

profiler_register::profiler_register(const char* func, int line) {
	std::string str = make_name(func, line);
	while (lock()++ != 0) {
		lock()--;}
auto 	cntptr = std::make_shared < real > (0.0);
	std::pair < std::string, std::shared_ptr<real> > entry;
	entry.first = str;
	entry.second = cntptr;
	map.insert(entry);
	lock()--;}

static/**/void accumulate() {
	const real told = t;
	t = hpx::util::high_resolution_clock::now() / 1e9;
	if (!callstack.empty()) {
		const std::string& str(callstack.top());
		while (lock()++ != 0) {
			lock()--;
			/* */
		}
		auto ptr = map[str];
		lock()--;
		real dt = t - told;
		(*ptr) += dt;
	}
}

static profiler_register prof_reg("OTHER", 0);

void profiler_enter(const char* func, int line) {
	accumulate();
	std::string str(make_name(func, line));
	if (callstack.empty()) {
		if (strncmp("OTHER", func, 5) != 0) {
			profiler_enter("OTHER", 0);
		}
	}
	callstack.push(str);
	t = hpx::util::high_resolution_clock::now() / 1e9;
}
void profiler_exit() {
	accumulate();
	callstack.pop();
	t = hpx::util::high_resolution_clock::now() / 1e9;
}

void profiler_output(FILE* _fp) {
#ifndef PROFILE_OFF
	std::map < real, std::string > ranks;
	real ttot = 0.0;
	for (auto i = map.begin(); i != map.end(); ++i) {
		real& tm = *(i->second);
		ranks[tm] = i->first;
		ttot += tm;
		tm = 0.0;
	}
	FILE* fps[2];
	fps[0] = _fp;
	fps[1] = stdout;
	for (int f = 0; f != 2; f++) {
		int r = 1;
		FILE* fp = fps[f];
		fprintf(fp, "%f total seconds\n", ttot);
		for (auto i = ranks.end(); i != ranks.begin(); r <= 10) {
			i--;
			fprintf(fp, "%4i %60s %8.2f %% %8.2f\n", r++, i->second.c_str(), i->first * 100.0 / ttot, i->first);
		}
		fprintf(fp, "\n");
	}
#endif
}

