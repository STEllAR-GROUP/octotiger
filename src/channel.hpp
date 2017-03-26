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

    void set_value( T value, std::size_t cycle ) {
        buffer.store_received(cycle, std::move(value));
	}

	hpx::future<T> get_future(std::size_t cycle) {
		return buffer.receive(cycle);
	}

};

#endif /* CHANNEL_HPP_ */
