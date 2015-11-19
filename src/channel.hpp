/*
 * channel.hpp
 *
 *  Created on: Jun 19, 2015
 *      Author: dmarce1
 */

#ifndef CHANNEL_HPP_
#define CHANNEL_HPP_

#include "defs.hpp"
#include <condition_variable>

template<class T>
class channel {
private:
	std::list<hpx::promise<T>> pipe;
	hpx::lcos::local::spinlock m;
	void add_one() {
		pipe.push_back(hpx::promise<T>());
	}
public:
	channel() {
		add_one();
	}
	~channel() = default;
	channel(const channel&) = delete;
	channel(channel&& other) = delete;
	channel& operator=(channel&& other) = delete;
	channel& operator=(const channel& other) = delete;

	template<class U>
	void set_value(U&& value) {
		std::lock_guard<hpx::lcos::local::spinlock> lk(m);
		auto& ref = pipe.back();
		add_one();
		ref.set_value(std::forward < U > (value));
	}

	hpx::future<T> get_future() {
		return hpx::async([=]() {
			std::unique_lock<hpx::lcos::local::spinlock> lk(m);
			auto fut = pipe.front().get_future();
			lk.unlock();
			fut.wait();
			lk.lock();
			pipe.pop_front();
			return fut.get();
		});
	}

};

#endif /* CHANNEL_HPP_ */
