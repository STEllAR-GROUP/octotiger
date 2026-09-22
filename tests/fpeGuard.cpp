#include "octotiger/math/Debug.hpp"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#if hasFpeEnableExcept
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static_assert(!std::is_copy_constructible_v<FpeGuard>);
static_assert(!std::is_move_constructible_v<FpeGuard>);
static_assert(!std::is_copy_assignable_v<FpeGuard>);
static_assert(!std::is_move_assignable_v<FpeGuard>);
static_assert(toMask({Fpe::invalid, Fpe::divByZero}) == (FE_INVALID | FE_DIVBYZERO));
static_assert(expectPositiveImpl(2.0, __FILE__, __LINE__) == 2.0);
static_assert(expectRangeImpl(1.0, 2.0, 3.0, __FILE__, __LINE__) == 2.0);

#ifdef expectFinite
#error "expectFinite must not be defined"
#endif

namespace {

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

#if hasFpeEnableExcept
constexpr int guardedExceptions = FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW;

void resetEnvironment() {
    fedisableexcept(FE_ALL_EXCEPT);
    std::feclearexcept(FE_ALL_EXCEPT);
}

void testNestingAndUnwinding() {
    resetEnvironment();
    feenableexcept(FE_UNDERFLOW);
    {
        FpeGuard outer{};
        require(fegetexcept() == (FE_UNDERFLOW | guardedExceptions), "Outer mask incorrect");
        {
            FpeGuard inner{};
            require(fegetexcept() == (FE_UNDERFLOW | guardedExceptions), "Nested mask incorrect");
        }
        require(fegetexcept() == (FE_UNDERFLOW | guardedExceptions), "Nested mask not restored");
        try {
            FpeGuard duringException(FE_INEXACT);
            require(fegetexcept() == FE_ALL_EXCEPT, "Additional exception not enabled");
            throw 7;
        } catch (int value) {
            require(value == 7, "Wrong C++ exception");
        }
        require(fegetexcept() == (FE_UNDERFLOW | guardedExceptions), "Unwinding changed mask");
    }
    require(fegetexcept() == FE_UNDERFLOW, "Caller mask not restored");
    resetEnvironment();

    // Also restore masks explicitly modified while an all-enabled guard lives.
    feenableexcept(guardedExceptions);
    {
        FpeGuard guard{};
        fedisableexcept(FE_INVALID);
        std::feraiseexcept(FE_INVALID);
    }
    require(fegetexcept() == guardedExceptions, "Explicit mask change not restored");
    require(std::fetestexcept(FE_INVALID) == 0, "Restoration enabled a stale exception");
    resetEnvironment();
}

void testPendingFlags() {
    resetEnvironment();
    std::feraiseexcept(guardedExceptions);
    int const previousFlags = std::fetestexcept(FE_ALL_EXCEPT);
    {
        FpeGuard outer{};
        require((std::fetestexcept(FE_ALL_EXCEPT) & guardedExceptions) == 0,
            "Stale exceptions not cleared before enabling");
        {
            FpeGuard inner{};
            volatile double one = 1.0;
            volatile double two = 2.0;
            volatile double result = one + two;
            require(result == 3.0, "Exact arithmetic failed after clearing stale flags");
        }
        // Flags for exceptions we do not unmask must remain observable.
        std::feraiseexcept(FE_INEXACT);
    }
    require(fegetexcept() == 0, "Pending-flag test changed caller mask");
    require(std::fetestexcept(FE_ALL_EXCEPT) == (previousFlags | FE_INEXACT),
        "Caller flags or unrelated new flags lost");
    resetEnvironment();
}

void triggerException(int exception) {
    resetEnvironment();
    FpeGuard guard{};
    volatile double zero = 0.0;
    volatile double one = 1.0;
    volatile double largest = std::numeric_limits<double>::max();
    if (exception == FE_INVALID) {
        volatile double result = zero / zero;
        (void)result;
    } else if (exception == FE_DIVBYZERO) {
        volatile double result = one / zero;
        (void)result;
    } else {
        volatile double result = largest * largest;
        (void)result;
    }
}

void testTrap(int exception) {
    pid_t const child = fork();
    require(child >= 0, "fork failed");
    if (child == 0) {
        struct rlimit const noCore{0, 0};
        setrlimit(RLIMIT_CORE, &noCore);
        std::signal(SIGFPE, SIG_DFL);
        triggerException(exception);
        _exit(1);
    }
    int status = 0;
    require(waitpid(child, &status, 0) == child, "waitpid failed");
    require(WIFSIGNALED(status) && WTERMSIG(status) == SIGFPE,
        "Floating-point exception did not trigger SIGFPE");
}

void testContracts() {
    resetEnvironment();
    assertRange(0.0, 0.5, 1.0, "value", __FILE__, __LINE__);
    assertPositive(0.5, "value", __FILE__, __LINE__);
    assertNonNegative(0.0, "value", __FILE__, __LINE__);
    assertNonzero(0.5, "value", __FILE__, __LINE__);
    require(fegetexcept() == 0, "Successful assertion leaked its mask");
    bool failed = false;
    try {
        assertRange(0.0, -0.5, 1.0, "value", __FILE__, __LINE__);
    } catch (std::runtime_error const&) {
        failed = true;
    }
    require(failed && fegetexcept() == 0, "Throwing assertion did not restore its mask");
}
#endif

} // namespace

int main() {
#if hasFpeEnableExcept
    std::fenv_t originalEnvironment;
    std::fegetenv(&originalEnvironment);
    try {
        testNestingAndUnwinding();
        testPendingFlags();
        testTrap(FE_INVALID);
        testTrap(FE_DIVBYZERO);
        testTrap(FE_OVERFLOW);
        testContracts();
        std::fesetenv(&originalEnvironment);
        std::cout << "FpeGuard tests passed\n";
        return EXIT_SUCCESS;
    } catch (std::exception const& error) {
        std::fesetenv(&originalEnvironment);
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
#else
    std::cout << "Floating-point exception masks are unavailable on this platform\n";
    return EXIT_SUCCESS;
#endif
}
