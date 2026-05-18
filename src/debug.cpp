//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/* MODIFIED FROM THIS ORIGINAL SOURCE :
 * http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes
 */

#include "octotiger/debug.hpp"

#define BOOST_STACKTRACE_USE_ADDR2LINE
// #define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>

char const* ThreadDebugger::name(int sig) {
    switch (sig) {
    case SIGSEGV:
        return "SIGSEGV";
    case SIGABRT:
        return "SIGABRT";
    case SIGFPE:
        return "SIGFPE";
    case SIGILL:
        return "SIGILL";
    case SIGBUS:
        return "SIGBUS";
    case SIGTRAP:
        return "SIGTRAP";
    case SIGTERM:
        return "SIGTERM";
    case SIGINT:
        return "SIGINT";
    default:
        return "SIGUNKNOWN";
    }
}

ThreadDebugger::ThreadDebugger() {
#if !defined(_MSC_VER)
    for (std::size_t i = 0; i < signals.size(); i++) {
        installHandler(i);
    }
#endif
}

ThreadDebugger::~ThreadDebugger() {
#if !defined(_MSC_VER)
    for (std::size_t i = 0; i < signals.size(); i++) {
        restoreHandler(i);
    }
#endif
}

void ThreadDebugger::installHandler(int sig) {
#if !defined(_MSC_VER)
    struct sigaction newAction
    {
    };
    newAction.sa_handler = handler;
    sigemptyset(&newAction.sa_mask);
    newAction.sa_flags = 0;
    sigaction(sig, &newAction, &oldActions[sig]);
#endif
}

void ThreadDebugger::restoreHandler(int sig) {
#if !defined(_MSC_VER)
    sigaction(sig, &oldActions[sig], nullptr);
#endif
};

void ThreadDebugger::handler(int sig) {
#if !defined(_MSC_VER)
    static std::atomic<int> called(false);
    if (!called++) {
        std::cout << boost::stacktrace::stacktrace();
    }
    exit(-1);
#endif
}

void ThreadDebugger::touch() {}
