
#pragma once

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <stdexcept>

template <typename... Args>
inline constexpr auto print2string(const char *format, Args &&...args) {
	auto const convert = [](auto v) {
		if constexpr (std::is_same_v<std::remove_cvref_t<decltype(v)>, std::string>) return v.c_str();
		return v;
	};
	char *ptr = nullptr;
	auto const count = asprintf(&ptr, format, convert(std::forward<Args>(args))...);
	if (count < 0 || ptr == nullptr) {
		throw std::runtime_error("asprintf failed");
	}
	std::string str(ptr, static_cast<std::size_t>(count));
	free(ptr);
	return str;
}

template <std::floating_point T>
std::string fp2str(T x) {
	constexpr int cnt = std::numeric_limits<T>::max_digits10;

	std::string format;
	if constexpr (std::is_same_v<T, long double>) {

		format = "%+" + std::to_string(cnt + 7) + "." + std::to_string(cnt - 1) + "Le";
	} else {
		format = "%+" + std::to_string(cnt + 7) + "." + std::to_string(cnt - 1) + "e";
	}

	char *ptr = nullptr;
	auto const count = asprintf(&ptr, format.c_str(), x);
	if (count < 0 || ptr == nullptr) {
		throw std::runtime_error("asprintf failed");
	}
	std::string str(ptr, static_cast<std::size_t>(count));
	free(ptr);
	return str;
}