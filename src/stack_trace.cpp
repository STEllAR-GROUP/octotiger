// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#if !defined(_MSC_VER)
#include <boost/stacktrace.hpp>
#include <signal.h>
#include <unistd.h>
#endif

#include "octotiger/debug.hpp"

namespace octotiger {
namespace debug {

#if !defined(_MSC_VER)
namespace {

constexpr std::array<int, 5> fatal_signals{SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL};
std::atomic<bool> owner_active{false};
std::atomic_flag reporting = ATOMIC_FLAG_INIT;
std::array<char, 384> process_header{};
size_t process_header_size = 0;

void write_diagnostic(char const *text, size_t length) noexcept {
    while (length != 0) {
        auto const written = ::write(STDERR_FILENO, text, length);
        if (written > 0) {
            text += written;
            length -= static_cast<size_t>(written);
        } else if (written == -1 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
}

template <size_t N> void write_diagnostic(char const (&text)[N]) noexcept {
    write_diagnostic(text, N - 1);
}

[[noreturn]] void terminate_with_signal(int signal) noexcept {
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    sigaction(signal, &action, nullptr);
    sigset_t unblocked;
    sigemptyset(&unblocked);
    sigaddset(&unblocked, signal);
    sigprocmask(SIG_UNBLOCK, &unblocked, nullptr);
    raise(signal);
    std::_Exit(128 + signal);
}

void report_signal(int signal) noexcept {
    if (reporting.test_and_set()) {
        terminate_with_signal(signal);
    }
    // Print the cached process identity even if stack unwinding/symbolization fails.
    // No HPX calls are made from the signal handler.
    write_diagnostic(process_header.data(), process_header_size);
    switch (signal) {
    case SIGABRT:
        write_diagnostic("Signal: SIGABRT\n");
        break;
    case SIGSEGV:
        write_diagnostic("Signal: SIGSEGV\n");
        break;
    case SIGBUS:
        write_diagnostic("Signal: SIGBUS\n");
        break;
    case SIGFPE:
        write_diagnostic("Signal: SIGFPE\n");
        break;
    case SIGILL:
        write_diagnostic("Signal: SIGILL\n");
        break;
    }
    // Source-line formatting is best effort: Boost/libbacktrace can allocate and
    // is not async-signal-safe. This is a diagnostic, not crash recovery.
    try {
        auto const trace = boost::stacktrace::to_string(boost::stacktrace::stacktrace(0, 64));
        write_diagnostic(trace.data(), trace.size());
    } catch (...) {
        write_diagnostic("Unable to symbolize the stacktrace.\n");
    }
    // Preserve signal termination (and core dumps), rather than exit(-1).
    terminate_with_signal(signal);
}

} // namespace

struct stacktrace_handler::state {
    std::array<struct sigaction, fatal_signals.size()> previous{};
    std::array<bool, fatal_signals.size()> installed{};

    explicit state(unsigned int locality_id) {
        // Resolve the backend before handling a fault, outside signal context.
        (void)boost::stacktrace::to_string(boost::stacktrace::stacktrace(0, 1));
        bool expected = false;
        if (!owner_active.compare_exchange_strong(expected, true)) {
            throw std::logic_error("Only one stacktrace handler may own a process");
        }
        std::array<char, 256> hostname{};
        if (gethostname(hostname.data(), hostname.size() - 1) != 0) {
            std::snprintf(hostname.data(), hostname.size(), "unknown");
        }
        auto const count = std::snprintf(process_header.data(), process_header.size(),
                                         "Octo-Tiger stacktrace: host=%s pid=%ld locality=%u\n",
                                         hostname.data(), static_cast<long>(getpid()), locality_id);
        process_header_size = count > 0 ? static_cast<size_t>(count) : 0;
        if (process_header_size >= process_header.size()) {
            process_header_size = process_header.size() - 1;
        }
        reporting.clear();
    }

    void install() {
        struct sigaction action{};
        action.sa_handler = report_signal;
        sigemptyset(&action.sa_mask);
        // Use an existing alternate signal stack, e.g. HPX's, when available.
        action.sa_flags = SA_ONSTACK;
        for (size_t i = 0; i < fatal_signals.size(); ++i) {
            if (sigaction(fatal_signals[i], &action, &previous[i]) != 0) {
                throw std::system_error(errno, std::generic_category(),
                                        "Unable to install the Octo-Tiger stacktrace handler");
            }
            installed[i] = true;
        }
    }

    ~state() {
        for (size_t i = fatal_signals.size(); i > 0; --i) {
            auto const index = i - 1;
            struct sigaction current{};
            if (installed[index] && sigaction(fatal_signals[index], nullptr, &current) == 0 &&
                !(current.sa_flags & SA_SIGINFO) && current.sa_handler == report_signal) {
                // Do not overwrite a handler subsequently installed by another subsystem.
                sigaction(fatal_signals[index], &previous[index], nullptr);
            }
        }
        owner_active.store(false);
    }
};
#else
struct stacktrace_handler::state {};
#endif

stacktrace_handler::stacktrace_handler() = default;
stacktrace_handler::~stacktrace_handler() = default;

void stacktrace_handler::install(unsigned int locality_id) {
#if !defined(_MSC_VER)
    if (!state_) {
        auto candidate = std::make_unique<state>(locality_id);
        candidate->install();
        state_ = std::move(candidate);
    }
#else
    (void)locality_id;
#endif
}

} // namespace debug
} // namespace octotiger
