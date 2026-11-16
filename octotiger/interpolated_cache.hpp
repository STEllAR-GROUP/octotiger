// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace octotiger {

// Lazily sampled four-point cubic interpolation on an anchored uniform grid.
// Queries outside the cached range extend it by whole blocks before evaluation;
// no query extrapolates. Coordinates and ordinates may have either sign. This is
// an ordinary cubic, so it does not impose monotonicity or positivity.
//
// One deterministic, thread-safe sampler is bound for the object's lifetime.
// Concurrent misses may duplicate sampling, but immutable published tables and
// globally anchored nodes make results independent of query order. Choose the
// spacing for the smoothness/accuracy of the function; no error is estimated.
//
// Log-log example for the positive function f(x) = 0.01 * (1 + x)^2, x > 0:
//   InterpolatedCache table([](long double const z) {
//       return std::log(0.01L) + 2 * std::log1p(std::exp(z));
//   }, 1.0L / 128, 128);
//   long double const value = std::exp(table.evaluate(std::log(x)));
// Here negative log ordinates are valid. Domain checks and transformations stay
// with the caller. Stencil sampling reaches one node beyond each range endpoint.
class InterpolatedCache
{
public:
    using Sampler = std::function<long double(long double)>;

    struct Statistics
    {
        size_t intervals = 0;
        size_t samples = 0;
        size_t sampleEvaluations = 0;
        long double lower = 0.0L;
        long double upper = 0.0L; // Exclusive upper bound, in input coordinates.
    };

private:
    using Index = std::int64_t;
    struct Table
    {
        Index lower;
        Index upper;
        std::vector<long double> values;
        std::vector<std::array<long double, 4>> coefficients;
    };

    Sampler const sampler_;
    long double const spacing_;
    Index const blockSize_;
    mutable std::shared_ptr<Table const> table_;
    mutable std::atomic<size_t> sampleEvaluations_{0};
    static constexpr Index indexLimit = std::numeric_limits<Index>::max() / 4;

    Index interval_index(long double const coordinate) const {
        if (!std::isfinite(coordinate)) {
            throw std::invalid_argument("Nonfinite interpolation coordinate");
        }
        long double const index = std::floor(coordinate / spacing_);
        if (std::abs(index) > static_cast<long double>(indexLimit)) {
            throw std::out_of_range("Interpolation coordinate exceeds index range");
        }
        return static_cast<Index>(index);
    }

    static bool contains(Table const& table, Index const index) {
        return index >= table.lower && index <= table.upper;
    }

    long double interpolate(Table const& table, Index const index,
        long double const coordinate) const {
        auto const& c = table.coefficients[static_cast<size_t>(index - table.lower)];
        long double const fraction = (coordinate - index * spacing_) / spacing_;
        long double const value = ((c[3] * fraction + c[2]) * fraction + c[1]) * fraction + c[0];
        if (!std::isfinite(value)) {
            throw std::overflow_error("Nonfinite cubic interpolation result");
        }
        return value;
    }

    std::shared_ptr<Table const> extend(std::shared_ptr<Table const> const& old,
        Index const requested) const {
        Index block = requested / blockSize_;
        if (requested < 0 && requested % blockSize_ != 0) {
            --block; // Integer floor division anchors negative blocks as well.
        }
        Index const blockLower = block * blockSize_;
        Index const blockUpper = blockLower + blockSize_ - 1;
        Index const lower = old ? std::min(old->lower, blockLower) : blockLower;
        Index const upper = old ? std::max(old->upper, blockUpper) : blockUpper;
        auto next = std::make_shared<Table>();
        next->lower = lower;
        next->upper = upper;
        size_t const count = static_cast<size_t>(upper - lower + 1);
        next->values.resize(count + 3);
        next->coefficients.resize(count);
        for (Index node = lower - 1; node <= upper + 2; ++node) {
            long double value;
            if (old && node >= old->lower - 1 && node <= old->upper + 2) {
                value = old->values[static_cast<size_t>(node - old->lower + 1)];
            } else {
                long double const coordinate = node * spacing_;
                if (!std::isfinite(coordinate)) {
                    throw std::overflow_error("Nonfinite interpolation stencil coordinate");
                }
                value = sampler_(coordinate);
                sampleEvaluations_.fetch_add(1, std::memory_order_relaxed);
                if (!std::isfinite(value)) {
                    throw std::runtime_error("Nonfinite interpolation sample");
                }
            }
            next->values[static_cast<size_t>(node - lower + 1)] = value;
        }
        for (Index index = lower; index <= upper; ++index) {
            size_t const slot = static_cast<size_t>(index - lower);
            if (old && contains(*old, index)) {
                next->coefficients[slot] = old->coefficients[static_cast<size_t>(index - old->lower)];
                continue;
            }
            long double const center = next->values[slot + 1];
            long double const left = next->values[slot] - center;
            long double const right = next->values[slot + 2] - center;
            long double const farRight = next->values[slot + 3] - center;
            // The four-point formula follows CosmicTiger's interpolation helper.
            auto& c = next->coefficients[slot];
            c = {{center, -left / 3.0L + right - farRight / 6.0L,
                0.5L * (left + right), -left / 6.0L - 0.5L * right + farRight / 6.0L}};
            for (long double const coefficient : c) {
                if (!std::isfinite(coefficient)) {
                    throw std::overflow_error("Nonfinite cubic interpolation coefficient");
                }
            }
        }
        return next;
    }

public:
    explicit InterpolatedCache(Sampler sampler, long double const spacing,
        Index const extensionBlockSize = 128)
      : sampler_(std::move(sampler)), spacing_(spacing), blockSize_(extensionBlockSize) {
        if (!sampler_) {
            throw std::invalid_argument("Interpolation requires a bound sampler");
        }
        if (!(spacing_ > 0.0L) || !std::isfinite(spacing_)) {
            throw std::invalid_argument("Interpolation spacing must be positive and finite");
        }
        if (blockSize_ <= 0 || blockSize_ > indexLimit) {
            throw std::invalid_argument("Invalid interpolation extension block size");
        }
    }

    long double evaluate(long double const coordinate) const {
        Index const index = interval_index(coordinate);
        auto snapshot = std::atomic_load_explicit(&table_, std::memory_order_acquire);
        for (;;) {
            if (snapshot && contains(*snapshot, index)) {
                return interpolate(*snapshot, index, coordinate);
            }
            // Sampling and allocation are outside publication. Racing extensions
            // may duplicate cold work; readers safely retain immutable snapshots.
            auto next = extend(snapshot, index);
            if (std::atomic_compare_exchange_strong_explicit(&table_, &snapshot, next,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return interpolate(*next, index, coordinate);
            }
        }
    }

    Statistics statistics() const {
        Statistics result;
        auto const snapshot = std::atomic_load_explicit(&table_, std::memory_order_acquire);
        result.sampleEvaluations = sampleEvaluations_.load(std::memory_order_relaxed);
        if (snapshot) {
            result.intervals = snapshot->coefficients.size();
            result.samples = snapshot->values.size();
            result.lower = snapshot->lower * spacing_;
            result.upper = (snapshot->upper + 1) * spacing_;
        }
        return result;
    }
};

} // namespace octotiger
