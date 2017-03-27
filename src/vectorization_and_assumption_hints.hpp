///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015-7 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

// "Portable" Bryce-to-vectorizer communication facilities.
// I never leave home without 'em!

#if !defined(BOOST_00009837_445B_4D65_BEA8_5CF085F9F9ED)
#define BOOST_00009837_445B_4D65_BEA8_5CF085F9F9ED

// BOOST_DETAIL_PP_STRINGIZE(expr) - Return expr as a string literal.
#define BOOST_DETAIL_PP_STRINGIZE_(expr) #expr
#define BOOST_DETAIL_PP_STRINGIZE(expr) BOOST_DETAIL_PP_STRINGIZE_(expr)

// BOOST_PP_PRAGMA(args) - Emits a pragma.
#define BOOST_PRAGMA(args) _Pragma(BOOST_DETAIL_PP_STRINGIZE(args))

// BOOST_DEMAND_VECTORIZATION - Insist that the compiler disregard loop-carried
// dependency analysis and cost modelling and vectorize the loop directly
// following the macro. Using this incorrectly can silently cause bogus codegen
// that blows up in unexpected ways. Usage:
//
// BOOST_DEMAND_VECTORIZATION for (/* ... */) { /* ... */ }
//
// NOTE: Unlike Clang and Intel, GCC doesn't have a stronger hint than ivdep,
// so this is the best we can do. It is not clear if this overrides GCC's cost
// modeling.
#if   defined(__INTEL_COMPILER)
    #define BOOST_DEMAND_VECTORIZATION                                         \
        BOOST_PRAGMA(simd)                                                     \
        /**/
#elif defined(__clang__)
    #define BOOST_DEMAND_VECTORIZATION                                         \
        BOOST_PRAGMA(clang loop vectorize(enable) interleave(enable))          \
        /**/
#else
    #define BOOST_DEMAND_VECTORIZATION                                         \
        BOOST_PRAGMA(GCC ivdep)                                                \
        /**/
#endif

// Sometimes it is nice to check that our brash and bold claims are, in fact,
// correct. Defining BOOST_CHECK_ASSUMPTIONS does that (e.g. assumption will be
// asserted before they are assumed).
#if defined(BOOST_CHECK_ASSUMPTIONS)
    #include <cassert>
    #include <stdint>
    #define BOOST_ASSERT_ASSUMPTION(expr) assert(expr)
#else
    #define BOOST_ASSERT_ASSUMPTION(expr)
#endif

// BOOST_ASSUME(expr) - Tell the compiler to assume that expr is true.
// Useful for telling the compiler that the trip count for a loop is division
// by a unrolling/vectorizing-friendly number:
//
//   BOOST_ASSUME((N % 32) == 0); for (auto i = 0; i != N; ++i) /* ... */
//
// BOOST_ASSUME_ALIGNED(ptr, align) - Tell the compiler to
// assume that ptr is aligned to align bytes. ptr must be an lvalue non-const
// pointer.
//
// NOTE: These used to have ridiculous exponential-in-number-of-uses
// compile-time costs with Clang/LLVM. For example, a 10k line project with
// ~100 BOOST_ASSUME/BOOST_ASSUME_ALIGNED usages would take ~20
// seconds to build with ICPC and ~5-10 minutes with Clang/LLVM. I believe the
// issue has now been fixed, but you'll run into it with older versions.
//
// NOTE: To the best of my knowledge, ICPC's __assume_aligned() is an
// assumption about the first argument, while Clang/GCC's
// __builtin_assume_aligned() is an assumption about the return value of the
// intrinsic.
#if   defined(__INTEL_COMPILER)
    #define BOOST_ASSUME(expr)                                                 \
        BOOST_ASSERT_ASSUMPTION(expr)                                          \
        __assume(expr)                                                         \
        /**/
    #define BOOST_ASSUME_ALIGNED(ptr, align)                                   \
        BOOST_ASSERT_ASSUMPTION(0 == (std::uintptr_t(ptr) % alignment))        \
        __assume_aligned(ptr, align)                                           \
        /**/
#elif defined(__clang__)
    #define BOOST_ASSUME(expr)                                                 \
        BOOST_ASSERT_ASSUMPTION(expr)                                          \
        __builtin_assume(expr)                                                 \
        /**/
    #define BOOST_ASSUME_ALIGNED(ptr, align)                                   \
        BOOST_ASSERT_ASSUMPTION(0 == (std::uintptr_t(ptr) % alignment))        \
        {                                                                      \
            ptr = reinterpret_cast<decltype(ptr)>(                             \
                __builtin_assume_aligned(ptr, align)                           \
            );                                                                 \
        }                                                                      \
        /**/
#else // GCC
    #define BOOST_ASSUME(expr)                                                 \
        BOOST_ASSERT_ASSUMPTION(expr)                                          \
        do { if (!(expr)) __builtin_unreachable(); } while (0)                 \
        /**/
    #define BOOST_ASSUME_ALIGNED(ptr, align)                                   \
        BOOST_ASSERT_ASSUMPTION(0 == (std::uintptr_t(ptr) % alignment))        \
        {                                                                      \
            ptr = reinterpret_cast<decltype(ptr)>(                             \
                __builtin_assume_aligned(ptr, align)                           \
            );                                                                 \
        }                                                                      \
        /**/
#endif

#endif // BOOST_00009837_445B_4D65_BEA8_5CF085F9F9ED

