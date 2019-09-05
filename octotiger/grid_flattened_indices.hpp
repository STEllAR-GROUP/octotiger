//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

constexpr int to_ab_idx_map3[3][3] = {
    {  4,  5,  6  },
    {  5,  7,  8  },
    {  6,  8,  9  }
};

constexpr int cb_idx_map[6] = {
     4,  5,  6,  7,  8,  9,
};

constexpr int to_abc_idx_map3[3][6] = {
    {  10, 11, 12, 13, 14, 15, },
    {  11, 13, 14, 16, 17, 18, },
    {  12, 14, 15, 17, 18, 19, }
};

constexpr int to_abcd_idx_map3[3][10] = {
    { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 },
    { 21, 23, 24, 26, 27, 28, 30, 31, 32, 33 },
    { 22, 24, 25, 27, 28, 29, 31, 32, 33, 34 }
};

constexpr int bcd_idx_map[10] = {
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19
};

constexpr int to_abc_idx_map6[6][3] = {
    { 10, 11, 12 }, { 11, 13, 14 },
    { 12, 14, 15 }, { 13, 16, 17 },
    { 14, 17, 18 }, { 15, 18, 19 }
};

constexpr int ab_idx_map6[6] = {
    4, 5, 6, 7, 8, 9
};
