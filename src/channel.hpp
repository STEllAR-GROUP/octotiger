/*
 * channel.hpp
 *
 *  Created on: Jun 19, 2015
 *      Author: dmarce1
 */

#ifndef CHANNEL_HPP_
#define CHANNEL_HPP_


//#define DEBUG_CHANNEL


#include "defs.hpp"
#include <boost/atomic.hpp>
#include <hpx/lcos/local/receive_buffer.hpp>

#ifdef DEBUG_CHANNEL
#include <unordered_set>
#include <typeinfo>
using used_cycles_t = std::unordered_set<std::size_t>;
#endif



template<class T>
class unordered_channel {
private:
	hpx::lcos::local::receive_buffer<T> buffer;
public:
	unordered_channel()
	{
	}
	~unordered_channel() = default;
	unordered_channel(const unordered_channel&) = delete;
	unordered_channel(unordered_channel&& other) = delete;
	unordered_channel& operator=(unordered_channel&& other) = delete;
	unordered_channel& operator=(const unordered_channel& other) = delete;


	void set_value(T value, std::size_t cycle) {
		buffer.store_received(cycle, std::move(value));
	}

	hpx::future<T> get_future(std::size_t cycle) {
		return buffer.receive(cycle);
	}
};



template<class T>
class channel {
private:
	unordered_channel<T> ch;
	boost::atomic<std::size_t> store_step;
	boost::atomic<std::size_t> recv_step;
public:
	channel() :
			store_step(0), recv_step(0)
	{
	}
	~channel() = default;
	channel(const channel&) = delete;
	channel(channel&& other) = delete;
	channel& operator=(channel&& other) = delete;
	channel& operator=(const channel& other) = delete;

	void set_value(T value) {
		ch.set_value(value,store_step++);
	}

	hpx::future<T> get_future() {
		return ch.get_future(recv_step++);
	}

};



#endif /* CHANNEL_HPP_ */

