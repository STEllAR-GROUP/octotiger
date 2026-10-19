// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <signal.h>

#include "octotiger/debug.hpp"

namespace {

std::atomic<unsigned int> main_calls{0};
std::atomic<bool> installed{false};

} // namespace

unsigned int inspect_stacktrace_worker() {
    unsigned int const locality = hpx::get_locality_id();
    unsigned int const calls = main_calls.load();
    bool const active = installed.load();
    std::fprintf(stderr, "WORKER_READY locality=%u hpx_main_calls=%u installed=%u\n", locality,
                 calls, active ? 1U : 0U);
    std::fflush(stderr);
    return locality == 1 && calls == 0 && active ? 0U : 1U;
}
HPX_PLAIN_ACTION(inspect_stacktrace_worker, inspect_stacktrace_worker_action);

[[gnu::noinline]] void crash_marker_stacktrace_worker() { std::abort(); }
HPX_PLAIN_ACTION(crash_marker_stacktrace_worker, crash_stacktrace_worker_action);

int hpx_main(hpx::program_options::variables_map &arguments) {
    ++main_calls;
    std::fprintf(stderr, "CONSOLE_MAIN locality=%u\n", hpx::get_locality_id());
    std::fflush(stderr);
    auto const remote = hpx::find_remote_localities();
    if (hpx::get_locality_id() != 0 || remote.size() != 1 ||
        hpx::async<inspect_stacktrace_worker_action>(remote.front()).get() != 0) {
        hpx::finalize();
        return 1;
    }
    if (arguments.count("crash-worker") != 0) {
        hpx::async<crash_stacktrace_worker_action>(remote.front()).get();
        return 3;
    }
    return hpx::finalize();
}

int main(int argc, char *argv[]) {
    octotiger::debug::stacktrace_handler handler;
    hpx::register_startup_function([&handler]() {
        struct sigaction before{};
        struct sigaction after{};
        bool const queried_before = sigaction(SIGABRT, nullptr, &before) == 0;
        handler.install(hpx::get_locality_id());
        bool const queried_after = sigaction(SIGABRT, nullptr, &after) == 0;
        installed = queried_before && queried_after && before.sa_handler != after.sa_handler;
    });
    hpx::program_options::options_description description("Stacktrace fixture");
    description.add_options()("crash-worker", "Crash the remote locality after inspecting it");
    hpx::init_params parameters;
    parameters.desc_cmdline = description;
    return hpx::init(argc, argv, parameters);
}
