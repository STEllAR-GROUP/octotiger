/*
 * channel.hpp
 *
 *  Created on: Jun 19, 2015
 *      Author: dmarce1
 */

#ifndef CHANNEL_HPP_
#define CHANNEL_HPP_

#include "defs.hpp"
//#ifndef MINI_HPX

#include <hpx/lcos/local/receive_buffer.hpp>

template<class T>
class channel {
private:
	hpx::lcos::local::receive_buffer<T> buffer;
    boost::atomic<std::size_t> store_step;
    boost::atomic<std::size_t> recv_step;
public:
	channel() : store_step(0), recv_step(0) {
	}
	~channel() = default;
	channel(const channel&) = delete;
	channel(channel&& other ) = delete;
	channel& operator=(channel&& other ) = delete;
	channel& operator=(const channel& other ) = delete;

    void set_value( T value ) {
        buffer.store_received(store_step++, std::move(value));
	}

	hpx::future<T> get_future() {
		return buffer.receive(recv_step++);
	}

};
/*
#else

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

#endif */
#endif /* CHANNEL_HPP_ */
