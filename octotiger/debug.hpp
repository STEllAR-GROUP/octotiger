// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#pragma once

#include <memory>

namespace octotiger {
namespace debug {

// One owner per process, kept alive until the HPX runtime has stopped.
// Install from an HPX startup callback, which runs on worker localities too.
class stacktrace_handler {
  public:
    stacktrace_handler();
    ~stacktrace_handler();

    stacktrace_handler(stacktrace_handler const &) = delete;
    stacktrace_handler &operator=(stacktrace_handler const &) = delete;

    void install(unsigned int locality_id = 0);

  private:
    struct state;
    std::unique_ptr<state> state_;
};

} // namespace debug
} // namespace octotiger
