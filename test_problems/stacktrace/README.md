# Stacktrace diagnostics

Octo-Tiger registers its fatal-signal handler from ordinary `main` through an
HPX startup callback. The callback runs on every locality after HPX installs
its own handlers, including workers which never execute `hpx_main`. A single
process-wide owner lives until the runtime has stopped. Unitiger installs the
same handler directly in its standalone `main`.

Diagnostics go to standard error and identify the hostname, PID and locality.
They include up to 64 frames, with source lines when debug information is
available. The build uses Boost's header-only libbacktrace backend to resolve
position-independent executable and shared-library addresses. CMake finds an
installed libbacktrace or GCC's bundled copy; `Libbacktrace_ROOT` can identify
an explicit installation. The Jenkins Spack recipe declares that dependency
for non-GCC compiler configurations too. Use a `RelWithDebInfo` or `Debug` build
for source locations. The handler covers SIGABRT, SIGSEGV, SIGBUS, SIGFPE and SIGILL; it
leaves SIGINT, SIGTERM and SIGHUP unchanged. After reporting, it restores the
default disposition and re-raises the fatal signal so launchers can detect a
signal failure and the OS can produce a core dump when enabled.

Source-line symbolization is best effort, not async-signal-safe: Boost can
allocate or acquire library locks while unwinding/formatting, and a damaged
process may fail to print a complete trace. The process header is written
before symbolization, and no HPX calls are made in the handler. This does not
provide recovery, a backtrace of every thread, or a coordinated backtrace of
all ranks. HPX builds with stack-overflow detection may subsequently install
their own SIGSEGV handler; its abort path still reaches this SIGABRT handler.

## History

- `3e394fdc` introduced Boost source-line traces with `hpx_main`/grid activation.
- `a24ce517` added its missing header, and `7485b798` corrected signal indexing
  and restoration.
- The earlier `a2b9a3b6` handler wrote `gdb.<hostname>.<pid>.txt` and used GDB
  on that host to collect all thread stacks. The current change preserves
  process identification without restoring the SSH/GDB mechanism.

## Focused validation

Build `stacktrace_fixture` and `stacktrace_hpx_fixture`, then run:

```sh
ctest --test-dir BUILD -R '^test_problems\.stacktrace\.' --output-on-failure -j1
```

The tests cover complete signal-action restoration, repeated installation,
preserving unrelated/replaced handlers, a fatal signal on a `std::thread`,
and two real HPX localities over loopback TCP. The distributed crash test
requires a source-line trace from locality 1 while only locality 0 executes
`hpx_main`. The supervisor disables core files and stops surviving processes.
These local tests do not establish behavior on a cluster launcher or MPI.
