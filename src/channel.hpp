/*
 * channel.hpp
 *
 *  Created on: Jun 19, 2015
 *      Author: dmarce1
 */

#ifndef CHANNEL_HPP_
#define CHANNEL_HPP_

#include "defs.hpp"

#include <hpx/lcos/local/receive_buffer.hpp>

//#define DEBUG_CHANNEL

template<class T>
class channel {
private:
	hpx::lcos::local::receive_buffer<T> buffer;
	boost::atomic<std::size_t> store_step;
	boost::atomic<std::size_t> recv_step;
#ifdef DEBUG_CHANNEL
	boost::atomic<integer> count;
	boost::atomic<integer> use;
#endif
public:
	channel() :
			store_step(0), recv_step(0)
#ifdef DEBUG_CHANNEL
					, count(0), use(0)
#endif
	{
	}
	~channel() = default;
	channel(const channel&) = delete;
	channel(channel&& other) = delete;
	channel& operator=(channel&& other) = delete;
	channel& operator=(const channel& other) = delete;

#ifdef DEBUG_CHANNEL
	void set_use(integer i) {
		if( use == 0 ) {
			use = i;
		} else if( use*i < 0 ) {
			printf( "MIXED CHANNEL USE\n");
			abort();
		}
	}
#endif

	void set_value(T value) {
#ifdef DEBUG_CHANNEL
		set_use(+1);
		++count;
		if (count > 1) {
			printf("EXCESS CHANNEL COUNT\n");
			abort();
		}
#endif
		buffer.store_received(store_step++, std::move(value));
	}

	hpx::future<T> get_future() {
#ifdef DEBUG_CHANNEL
		set_use(+1);
		return buffer.receive(recv_step++).then([this](hpx::future<T> fut) {
			--count;
			return fut.get();
		});
#else
		return buffer.receive(recv_step++);
#endif
	}

	void set_value(T value, std::size_t cycle) {
#ifdef DEBUG_CHANNEL
		set_use(-1);
#endif
		buffer.store_received(cycle, std::move(value));
	}

	hpx::future<T> get_future(std::size_t cycle) {
#ifdef DEBUG_CHANNEL
		set_use(-1);
#endif
		return buffer.receive(cycle);
	}

};

#endif /* CHANNEL_HPP_ */
