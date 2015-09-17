/*
 * channel.hpp
 *
 *  Created on: Jun 19, 2015
 *      Author: dmarce1
 */

#ifndef CHANNEL_HPP_
#define CHANNEL_HPP_

#include "defs.hpp"

template<class T>
class channel {
private:
	hpx::promise<T> promise;
public:
	channel() {
	}
	~channel() = default;
	channel(const channel&) = delete;
	channel(channel&& other ) = delete;
	channel& operator=(channel&& other ) = delete;
	channel& operator=(const channel& other ) = delete;

    void set_value( T&& value ) {
		promise.set_value(std::move(value));
	}

	void set_value( const T& value ) {
		promise.set_value(value);
	}

	hpx::future<T> get_future() {
		return hpx::async([=]() {
			std::shared_ptr<T> data_ptr;
			auto fut = promise.get_future();
			data_ptr = std::make_shared<T>(fut.get());
			promise = hpx::promise<T>();
			return std::move(*data_ptr);
		});
	}

};


#endif /* CHANNEL_HPP_ */
