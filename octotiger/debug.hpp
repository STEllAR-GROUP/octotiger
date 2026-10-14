//  Copyright (c) 2026 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if !defined(_MSC_VER)
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#endif

#include <array>

struct ThreadDebugger {
    void touch();
    ThreadDebugger();
    ~ThreadDebugger();

private:
#if !defined(_MSC_VER)
    static constexpr auto signals =
        std::array{SIGABRT, SIGINT, SIGSEGV, SIGFPE, SIGILL, SIGTERM, SIGHUP, SIGABRT};
    static constexpr auto sigCount = signals.size();
    std::array<struct sigaction, sigCount> oldActions;
#endif
    static void handler(int);
    void installHandler(int);
    void restoreHandler(int);
    static char const* name(int);
};

static thread_local ThreadDebugger threadDebugger;

#if defined(_MSC_VER)
#define ENABLE_THREAD_DEBUG()
#else
#define ENABLE_THREAD_DEBUG() threadDebugger.touch()
#endif