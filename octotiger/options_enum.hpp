//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SRC_OPTIONS_ENUM_HPP_
#define SRC_OPTIONS_ENUM_HPP_

#include <boost/algorithm/string.hpp>

#include "octotiger/print.hpp"

#include <cstddef>
#include <iostream>
#include <istream>
#include <string>
#include <vector>

#define COMMAND_LINE_ENUM(enum_name, ...)                                      \
    enum enum_name : integer                                                   \
    {                                                                          \
        __VA_ARGS__                                                            \
    };                                                                         \
    static inline std::istream& operator>>(std::istream& in, enum_name& e)     \
    {                                                                          \
        std::vector<std::string> strings;                                      \
        boost::split(strings, #__VA_ARGS__, boost::is_any_of(" ,\t"),          \
            boost::token_compress_on);                                         \
        static enum_name enums[] = {__VA_ARGS__};                              \
        bool success = false;                                                  \
        std::string token;                                                     \
        in >> token;                                                           \
        for (std::size_t i = 0; i < strings.size(); i++)                       \
        {                                                                      \
            if (boost::iequals(strings[i], token))                             \
            {                                                                  \
                success = true;                                                \
                e = enums[i];                                                  \
            }                                                                  \
        }                                                                      \
        if (!success)                                                          \
        {                                                                      \
            in.setstate(std::ios_base::failbit);                               \
        }                                                                      \
        return in;                                                             \
    }                                                                          \
    static inline std::string to_string(enum_name e)                           \
    {                                                                          \
        std::vector<std::string> strings;                                      \
        boost::split(strings, #__VA_ARGS__, boost::is_any_of(" ,\t"),          \
            boost::token_compress_on);                                         \
        static enum_name enums[] = {__VA_ARGS__};                              \
        static std::size_t sz1 =                                                       \
            std::vector<enum_name>(std::begin(enums), std::end(enums)).size(); \
        if (sz1 != strings.size())                                             \
        {                                                                      \
            print("Different sizes %i %i\n", int(sz1), int(strings.size()));  \
        }                                                                      \
        std::string rc;                                                        \
        for (std::size_t i = 0; i < strings.size(); i++)                       \
        {                                                                      \
            if (e == enums[i])                                                 \
            {                                                                  \
                rc = strings[i];                                               \
                break;                                                         \
            }                                                                  \
        }                                                                      \
        if (rc.empty())                                                        \
        {                                                                      \
            std::cout << "invalid enum value for " << #enum_name;              \
            std::cout << "\n";                                                 \
            throw;                                                             \
        }                                                                      \
        return rc;                                                             \
    }

#endif /* SRC_OPTIONS_ENUM_HPP_ */
