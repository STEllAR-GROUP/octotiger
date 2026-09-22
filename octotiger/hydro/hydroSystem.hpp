// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/math/Vector.hpp"
#include "octotiger/mesh.hpp"
#include "octotiger/physics/finiteVolume.hpp"

#include <array>

namespace octotiger::hydro {

class ConservedState : public Vector<Real, 5> {
public:
    using Vector<Real, 5>::Vector;
    constexpr ConservedState() = default;
    constexpr ConservedState(Vector<Real, 5> const& state) :
        Vector<Real, 5>(state) {
    }

    constexpr Real& density() {
        return (*this)[0];
    }
    constexpr Real density() const {
        return (*this)[0];
    }
    constexpr Real& momentum(int axis) {
        return (*this)[axis + 1];
    }
    constexpr Real momentum(int axis) const {
        return (*this)[axis + 1];
    }
    constexpr Real& totalEnergy() {
        return (*this)[4];
    }
    constexpr Real totalEnergy() const {
        return (*this)[4];
    }
};

class PrimitiveState : public Vector<Real, 5> {
public:
    using Vector<Real, 5>::Vector;
    constexpr PrimitiveState() = default;
    constexpr PrimitiveState(Vector<Real, 5> const& state) :
        Vector<Real, 5>(state) {
    }

    constexpr Real& density() {
        return (*this)[0];
    }
    constexpr Real density() const {
        return (*this)[0];
    }
    constexpr Real& velocity(int axis) {
        return (*this)[axis + 1];
    }
    constexpr Real velocity(int axis) const {
        return (*this)[axis + 1];
    }
    constexpr Real& pressure() {
        return (*this)[4];
    }
    constexpr Real pressure() const {
        return (*this)[4];
    }
};

class HydroSystem {
public:
    using State = ConservedState;
    using Reconstruction = PrimitiveState;

    explicit HydroSystem(Real adiabaticIndex = Real(5) / 3,
        Real densityFloor = Real(1e-14), Real pressureFloor = Real(1e-14));

    Real adiabaticIndex() const;
    [[nodiscard]] PrimitiveState reconstructionVariables(ConservedState const& state) const;
    [[nodiscard]] ConservedState conservedState(PrimitiveState const& state) const;
    ConservedState physicalFlux(ConservedState const& state, int normal) const;
    ConservedState riemann(ConservedState const& left,
        ConservedState const& right, int normal) const;
    [[nodiscard]] ConservedState reflected(ConservedState state, int normal) const;
    Real maximumSignalSpeed(ConservedState const& state, int normal) const;
    bool admissible(ConservedState const& state) const;
    [[nodiscard]] ConservedState correctRoundoff(ConservedState state, Real updateScale) const;
    [[nodiscard]] ConservedState limitFlux(ConservedState const& left,
        ConservedState const& right, ConservedState const& highOrderFlux,
        int normal, Real stepOverCellWidth, int dimensionCount) const;

private:
    Real adiabaticIndex_;
    Real densityFloor_;
    Real pressureFloor_;

    Real pressure(ConservedState const& state) const;
    ConservedState hll(ConservedState const& left,
        ConservedState const& right, int normal) const;
};

template <int dimensionCount>
using Solver = physics::MusclHancock<HydroSystem, dimensionCount>;

using Fields = mesh::PatchData<ConservedState>;

} // namespace octotiger::hydro
