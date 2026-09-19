// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/math/Debug.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/math/Vector.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <utility>

// Skinner & Ostriker (2013), ApJS 206:21, https://arxiv.org/abs/1306.0010.
// Equation numbers below refer to S&O. Storage/checkpoints contain physical F;
// numerical states use Q=F/c (a constant change of units, NOT F/c_hat).
// Reconstruct (E,Q) directly. There are no Hanawa variables or eigensystems.
template <typename Type = Real, int dimensionCount = 3>
class RadiationM1 {
    static_assert(dimensionCount > 0);
public:
    RadiationM1() = delete;
    using Scalar = Type;
    using SpatialVector = Vector<Type, dimensionCount>;
    using StateVector = Vector<Type, dimensionCount + 1>;
    static constexpr int fieldCount = dimensionCount + 1;
    static constexpr Type roundoff = Type(4096) * std::numeric_limits<Type>::epsilon();

    struct Closure {
        Type chi, isotropic, directed;
        SpatialVector reducedFlux;
        // P_ij = isotropic delta_ij + directed f_i f_j, f=Q/E.
    };
    struct FluxState {
        StateVector flux;
        Type minus, plus;
    };
    struct Energies { Type gas, radiation; };

    class ConservedState : public StateVector {
    public:
        using StateVector::StateVector;
        constexpr ConservedState() = default;
        constexpr ConservedState(StateVector const& u) : StateVector(u) {}
        constexpr Type& energyDensity() { return (*this)[0]; }
        constexpr Type energyDensity() const { return (*this)[0]; }
        constexpr SpatialVector normalizedFlux() const { return this->template split<1>().second; }
        void checkState(char const* context = "") const {
            if (!admissible(*this)) {
                std::ostringstream message;
                message.precision(17);
                message << context << ": M1 state requires finite E >= 0 and |F| <= c E; (E,F/c)=(";
                for (int f=0; f<fieldCount; ++f) message << (f ? "," : "") << (*this)[f];
                throw std::runtime_error(message.str()+")");
            }
        }
        Closure closure() const {
            FpeGuard guard;
            checkState("closure");
            Closure m{};
            m.chi = Type(1)/3;
            if ((*this)[0] == 0) return m;
            m.reducedFlux = normalizedFlux() / (*this)[0];
            Type f2 = m.reducedFlux.dot(m.reducedFlux);
            if (f2 > 1) { // roundoff only; the state was checked above
                m.reducedFlux /= std::sqrt(f2);
                f2 = 1;
            }
            Type const s = std::sqrt(Type(4)-Type(3)*f2);
            // Equations (14)--(15), evaluated without dividing by |F| at vacuum.
            m.chi = (Type(3)+Type(4)*f2)/(Type(5)+Type(2)*s);
            m.isotropic = (*this)[0] * (Type(1)-f2)/(s+Type(1));
            m.directed = (*this)[0] * Type(3)/(s+Type(2));
            return m;
        }
        FluxState physicalFlux(int normal, Type chat, Type gridVelocity = 0) const {
            FpeGuard guard;
            if (!(chat > 0) || normal < 0 || normal >= dimensionCount)
                throw std::runtime_error("Invalid M1 flux speed or normal");
            auto const m = closure();
            auto const f2 = std::min(Type(1), m.reducedFlux.dot(m.reducedFlux));
            auto const fn = m.reducedFlux[normal];
            auto const mu2 = f2 > 0 ? std::min(Type(1), fn*fn/f2) : Type(0);
            auto const s = std::sqrt(Type(4)-Type(3)*f2);
            // Eq. (41a): the radical is rewritten as
            // 2(1-f^2)[s+mu^2(s-2)]/(s+1), finite at f=0 and f=1.
            auto const radical = std::sqrt(std::max(Type(0),
                Type(2)*(Type(1)-f2)*(s+mu2*(s-Type(2)))/(s+Type(1))));
            FluxState out;
            out.minus = chat*(fn-radical)/s-gridVelocity;
            out.plus = chat*(fn+radical)/s-gridVelocity;
            // Eq. (21), rescaled to (E,Q): flux = c_hat (Q_n, P_nj).
            out.flux[0] = chat*(*this)[0]*fn-gridVelocity*(*this)[0];
            for (int d=0; d<dimensionCount; ++d) {
                Type const pressure = m.directed*fn*m.reducedFlux[d]
                    + (normal==d ? m.isotropic : Type(0));
                out.flux[d+1] = chat*pressure-gridVelocity*(*this)[d+1];
            }
            return out;
        }
    };
    using ConservedFlux = StateVector;

    static bool admissible(StateVector const& u) {
        if (!(u[0] >= 0) || !std::isfinite(u[0])) return false;
        Type norm = 0;
        for (int d=1; d<fieldCount; ++d) {
            if (!std::isfinite(u[d])) return false;
            norm = std::hypot(norm, u[d]);
        }
        return norm <= u[0] || (u[0] > 0 && norm-u[0] <= roundoff*u[0]);
    }

    // Resolve floating-point cancellation at the boundary of the cone. The
    // tolerance is measured against the local update, never an absolute floor.
    // The caller records any full-step correction in its conservation ledger.
    static ConservedState roundoffState(ConservedState u, Type updateScale) {
        Type magnitude=0;
        for (int f=1; f<fieldCount; ++f) magnitude=std::hypot(magnitude,u[f]);
        Type const tolerance=roundoff*updateScale;
        if (u[0]<0 && -u[0]<=tolerance && magnitude<=tolerance) return {};
        if (u[0]>=0 && magnitude>u[0] && magnitude-u[0]<=tolerance)
            for (int f=1; f<fieldCount; ++f) u[f]*=u[0]/magnitude;
        u.checkState("roundoff correction");
        return u;
    }

    // Athena's primitive PLM limiter (lr_states_prim2.c, steps 1--3).
    // For same-sign differences its limited slope is the harmonic mean.
    // Scaling avoids overflow in the usual 2*dl*dr/(dl+dr) expression.
    static Type plmSlope(Type dl, Type dr) {
        if (!((dl>0 && dr>0) || (dl<0 && dr<0))) return 0;
        Type const small = std::min(std::abs(dl), std::abs(dr));
        Type const large = std::max(std::abs(dl), std::abs(dr));
        return std::copysign(small * (Type(2)/(Type(1)+small/large)), dl);
    }
    static std::pair<ConservedState, ConservedState> reconstruct(
        ConservedState const& left, ConservedState const& center, ConservedState const& right) {
        FpeGuard guard;
        center.checkState("PLM center");
        StateVector slope;
        for (int f=0; f<fieldCount; ++f)
            slope[f] = Type(.5)*plmSlope(center[f]-left[f], right[f]-center[f]);
        // M1-specific addition to component PLM: scale ALL slopes together
        // so both faces remain in the convex realizability cone. The cell
        // average and the existing component monotonicity are preserved.
        auto valid = [&](Type a) { return admissible(center-a*slope) && admissible(center+a*slope); };
        Type a = 1;
        if (!valid(a)) {
            Type lo=0, hi=1;
            for (int it=0; it<56; ++it) {
                Type const mid=Type(.5)*(lo+hi);
                if (valid(mid)) lo=mid; else hi=mid;
            }
            a=lo;
        }
        return {ConservedState(center-a*slope), ConservedState(center+a*slope)};
    }

    // Equations (38)--(39): upwind HLL with the extreme speeds from BOTH states.
    static ConservedFlux hll(ConservedState const& left, ConservedState const& right,
        int normal, Type chat, Type gridVelocity = 0) {
        FpeGuard guard;
        auto canonical=[](ConservedState u) {
            u.checkState("HLL input");
            Type magnitude=0;
            for (int d=1; d<fieldCount; ++d) magnitude=std::hypot(magnitude,u[d]);
            if (magnitude>u[0])
                for (int d=1; d<fieldCount; ++d) u[d]*=u[0]/magnitude;
            return u;
        };
        auto const ul=canonical(left),ur=canonical(right);
        auto const l=ul.physicalFlux(normal,chat,gridVelocity);
        auto const r=ur.physicalFlux(normal,chat,gridVelocity);
        auto const sm=std::min(l.minus,r.minus), sp=std::max(l.plus,r.plus);
        if (sm>=0) return l.flux;
        if (sp<=0) return r.flux;
        auto const wr=sp/(sp-sm), wl=-sm/(sp-sm);
        // Group each one-sided wave contribution BEFORE weighting. This
        // cancels a streaming wave against vacuum exactly, rather than leaving
        // a tiny negative photon density from separately rounded large terms.
        return wr*(l.flux-sm*ul)+wl*(r.flux-sp*ur);
    }

    // Conservative realizability limiter for the final transport flux. Each
    // face gives two admissible one-face updates; the full unsplit FV update
    // is their convex average. Both neighbors compute the SAME face flux from
    // the t^n halo. No cell clipping or midstep boundary exchange is involved.
    static ConservedFlux limitFlux(ConservedState const& left, ConservedState const& right,
        ConservedFlux const& high, int normal, Type chat, Type gridVelocity, Type dtOverDx) {
        if (dtOverDx==0) return high;
        auto const l=left.physicalFlux(normal,chat,gridVelocity).flux;
        auto const r=right.physicalFlux(normal,chat,gridVelocity).flux;
        Type const factor=Type(2*dimensionCount)*dtOverDx;
        Type const tolerance=roundoff*(left[0]+right[0]);
        auto validState=[&](StateVector const& u) {
            if (admissible(u)) return true;
            if (!std::isfinite(u[0]) || u[0]<-tolerance) return false;
            Type norm=0;
            for (int f=1; f<fieldCount; ++f) {
                if (!std::isfinite(u[f])) return false;
                norm=std::hypot(norm,u[f]);
            }
            return norm-std::max(Type(0),u[0])<=tolerance;
        };
        auto valid=[&](StateVector const& flux) {
            return validState(left-factor*(flux-l)) && validState(right+factor*(flux-r));
        };
        if (valid(high)) return high;
        // Global-wave-speed HLL (local Lax-Friedrichs) is the realizable
        // first-order fallback. Roundoff padding handles exactly streaming F.
        Type const a=(chat+std::abs(gridVelocity))*(Type(1)+roundoff);
        ConservedFlux const low=Type(.5)*(l+r-a*(right-left));
        if (!valid(low)) throw std::runtime_error("M1 first-order flux violates realizability; reduce rad_cfl");
        Type lo=0,hi=1;
        for (int it=0; it<56; ++it) {
            Type const mid=Type(.5)*(lo+hi);
            if (valid(low+mid*(high-low))) lo=mid; else hi=mid;
        }
        return low+lo*(high-low);
    }

    // Sum-of-direction CFL for VL and its convex limiter; cfl is in (0,0.5].
    static Type transportTimestep(Type dx, Type chat, Type cfl, Type gridSpeed = 0) {
        if (!(dx>0 && chat>0 && cfl>0 && cfl<=Type(.5) && gridSpeed>=0))
            throw std::runtime_error("Invalid radiation CFL parameters");
        return cfl*dx/(Type(dimensionCount)*(chat+gridSpeed));
    }

    // Eq. (25), using the isotropic O(beta tau) simplification (18).
    // chiA != chiT extends the work term using (5a), preserving the existing
    // opacity model; equal means (or rad_opacity >= 0) recover S&O exactly.
    static StateVector explicitSource(ConservedState const& u, SpatialVector const& velocity,
        Type chiA, Type chiT, Type c, Type ratio, bool velocityTerms = true) {
        StateVector source{};
        if (!velocityTerms) return source;
        source[0]=ratio*(Type(2)*chiA-chiT)*velocity.dot(u.normalizedFlux());
        for (int d=0; d<dimensionCount; ++d)
            source[d+1]=ratio*chiT*(Type(4)/3)*u[0]*velocity[d];
        (void)c; // velocity is physical; ratio=c_hat/c supplies the v/c factor
        return source;
    }

    // Equation (43), with an optional explicit transport/source increment.
    // A theta<1 update that would reverse a damped quantity uses BE instead.
    static Type damp(Type old, Type increment, Type eta, Type theta) {
        if (!(eta>=0 && theta>=Type(.5) && theta<=1))
            throw std::runtime_error("Invalid radiation damping parameters");
        if ((Type(1)-theta)*eta>1) theta=1;
        Type const inv=Type(1)/(Type(1)+theta*eta);
        return (Type(1)-(Type(1)-theta)*eta)*inv*old+inv*increment;
    }

    // Equations (44)--(49). eta=c*chiA*dt uses PHYSICAL c and ratio=c_hat/c.
    // The invariant is e+E/ratio, not e+E. logAlpha=log(a_R[(gamma-1)mu/rho/kB]^4).
    // Solve the monotone scaled quartic with safeguarded Newton/bisection.
    // Fall back from theta<1 to BE if no nonnegative solution exists (S&O 3.4).
    static Energies thermalExchange(Type e, Type E, Type eta, Type logAlpha,
        bool marshak = false, Type ratio = 1, Type theta = 1) {
        FpeGuard guard;
        if (!(e>=0 && E>=0 && eta>=0 && ratio>0 && ratio<=1 && theta>=Type(.5) && theta<=1))
            throw std::runtime_error("Invalid radiation thermal-exchange parameters");
        if (eta==0) return {e,E};
        for (int attempt=0; attempt<2; ++attempt) {
            Type const inv=Type(1)/(Type(1)+theta*ratio*eta);
            Type const w=theta*eta*inv;
            Type oldEmission=0;
            if (theta<1 && e>0) {
                Type const logEmission=logAlpha+Type(4)*std::log(e);
                if (!marshak && logEmission>std::log(std::numeric_limits<Type>::max())) {
                    theta=1; continue;
                }
                oldEmission=marshak ? e : std::exp(logEmission);
            }
            Type const explicitPart=(Type(1)-theta)*eta*inv*oldEmission;
            Type const rhs=e+eta*inv*E-explicitPart;
            if (!(rhs>=0)) { theta=1; continue; }
            Type gas=0, emissionTerm=0;
            if (marshak) {
                gas=rhs/(Type(1)+w);
                emissionTerm=w*gas;
            } else if (rhs>0) {
                Type const logRhs=std::log(rhs);
                Type const logCoefficient=std::log(w)+logAlpha;
                Type const quarticLogScale=(logRhs-logCoefficient)/Type(4);
                bool const linearBound=logRhs<=quarticLogScale;
                Type const scale=linearBound ? rhs : std::exp(quarticLogScale);
                Type const linear=linearBound ? Type(1) : scale/rhs;
                Type const quartic=linearBound ?
                    std::exp(std::min(Type(0),logCoefficient+Type(3)*logRhs)) : Type(1);
                Type lo=0, hi=1, y=1;
                bool solved=false;
                for (int it=0; it<100; ++it) {
                    Type const y2=y*y;
                    Type const residual=linear*y+quartic*y2*y2-Type(1);
                    if (std::abs(residual)<=Type(8)*std::numeric_limits<Type>::epsilon()) {
                        gas=scale*y;
                        emissionTerm=rhs*quartic*y2*y2;
                        solved=true; break;
                    }
                    if (residual>0) hi=y; else lo=y;
                    Type const next=y-residual/(linear+Type(4)*quartic*y2*y);
                    y=(next>lo && next<hi) ? next : Type(.5)*(lo+hi);
                }
                if (!solved) throw std::runtime_error("Radiation thermal exchange did not converge");
            }
            // Direct radiation update avoids subtraction of two gas energies
            // when radiation is weak; the quartic enforces the same invariant.
            Type const nextE=(Type(1)-(Type(1)-theta)*ratio*eta)*inv*E
                +ratio*(explicitPart+emissionTerm);
            if (nextE>=0 && std::isfinite(nextE)) return {gas,nextE};
            theta=1;
        }
        throw std::runtime_error("Radiation thermal exchange has no physical solution");
    }
};
