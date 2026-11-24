// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <signal.h>

#include "octotiger/debug.hpp"

namespace {

constexpr auto tested_signals =
    std::array{SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGINT, SIGTERM, SIGHUP};
constexpr auto fatal_signals = std::array{SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL};
constexpr auto preserved_signals = std::array{SIGINT, SIGTERM, SIGHUP};

void saved_handler(int) {}
void replacement_handler(int) {}

bool read_action(int signal, struct sigaction &action) {
    if (sigaction(signal, nullptr, &action) != 0) {
        std::perror("sigaction query");
        return false;
    }
    return true;
}

bool matches_saved_action(int signal) {
    struct sigaction action{};
    return read_action(signal, action) && action.sa_handler == saved_handler &&
           (action.sa_flags & SA_RESTART) != 0 && sigismember(&action.sa_mask, SIGUSR1) == 1;
}

bool lifecycle() {
    std::array<struct sigaction, tested_signals.size()> originals{};
    struct sigaction saved{};
    saved.sa_handler = saved_handler;
    saved.sa_flags = SA_RESTART;
    sigemptyset(&saved.sa_mask);
    sigaddset(&saved.sa_mask, SIGUSR1);
    for (size_t index = 0; index < tested_signals.size(); ++index) {
        if (sigaction(tested_signals[index], &saved, &originals[index]) != 0) {
            std::perror("sigaction setup");
            return false;
        }
    }

    bool okay = true;
    {
        octotiger::debug::stacktrace_handler handler;
        handler.install(7);
        for (int const signal : fatal_signals) {
            struct sigaction action{};
            okay = read_action(signal, action) && okay;
            if (action.sa_handler == saved_handler || action.sa_handler == SIG_DFL ||
                action.sa_handler == SIG_IGN) {
                std::fprintf(stderr, "Fatal signal %d was not installed\n", signal);
                okay = false;
            }
        }
        for (int const signal : preserved_signals) {
            if (!matches_saved_action(signal)) {
                std::fprintf(stderr, "Unrelated signal %d was changed\n", signal);
                okay = false;
            }
        }
        // Re-installation must not save our own handler as the original action.
        handler.install(7);
    }
    for (int const signal : tested_signals) {
        if (!matches_saved_action(signal)) {
            std::fprintf(stderr, "Signal %d was not restored completely\n", signal);
            okay = false;
        }
    }

    {
        octotiger::debug::stacktrace_handler handler;
        handler.install(7);
        struct sigaction replacement = saved;
        replacement.sa_handler = replacement_handler;
        if (sigaction(SIGABRT, &replacement, nullptr) != 0) {
            std::perror("sigaction replacement");
            okay = false;
        }
    }
    struct sigaction action{};
    if (!read_action(SIGABRT, action) || action.sa_handler != replacement_handler) {
        std::fprintf(stderr, "Destructor overwrote a later signal handler\n");
        okay = false;
    }

    for (size_t index = 0; index < tested_signals.size(); ++index) {
        if (sigaction(tested_signals[index], &originals[index], nullptr) != 0) {
            std::perror("sigaction test cleanup");
            okay = false;
        }
    }
    return okay;
}

} // namespace

// raise has a return site within this function even in optimized builds;
// a noreturn abort call can put its return PC just past the function's DWARF range.
[[gnu::noinline]] void crash_marker_stacktrace_thread() { raise(SIGABRT); }

int main(int argc, char *argv[]) {
    if (argc != 2) {
        return 2;
    }
    if (std::strcmp(argv[1], "lifecycle") == 0) {
        return lifecycle() ? 0 : 1;
    }
    if (std::strcmp(argv[1], "crash") == 0) {
        octotiger::debug::stacktrace_handler handler;
        handler.install(7);
        std::thread worker(crash_marker_stacktrace_thread);
        worker.join();
        return 3;
    }
    return 2;
}
