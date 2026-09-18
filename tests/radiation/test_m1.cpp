// Standalone numerical regression tests for the production radiation kernels.
// Run from the source root: g++ -std=c++23 -O2 -I. tests/radiation/test_m1.cpp -o /tmp/test_m1 && /tmp/test_m1
#include "octotiger/radiation/m1.hpp"
#include <iostream>
#include <random>
#include <vector>

namespace rm = radiation_m1;
using rm::state;
using vec = rm::vector;

static int checks = 0;
void require(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
void near(Real a, Real b, Real tolerance, const char* message) {
    require(std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= tolerance * std::max({std::abs(a), std::abs(b), 1e-280}), message);
}
void admissible(const state& u, Real c) {
    require(std::isfinite(u[0]) && u[0] >= 0, "energy admissibility");
    require(std::hypot(u[1]/c, u[2]/c, u[3]/c) <= (1 + 1e-12)*u[0], "flux admissibility");
}

void closure_and_hll() {
    for (Real c : {0.25, 7.0, 2.99792458e10}) {
        const auto iso = rm::physical_flux({3,0,0,0}, 0, c);
        near(iso.minus, -c/std::sqrt(3.), 1e-14, "isotropic minus speed");
        near(iso.plus, c/std::sqrt(3.), 1e-14, "isotropic plus speed");
        near(iso.flux[1], c*c, 1e-14, "pressure flux must have c squared");
        for (int d=0; d<3; ++d) {
            state ul{2,0,0,0}, ur{5,0,0,0};
            ul[d+1] = 2*c; ur[d+1] = 5*c;
            const auto fp = rm::hll(ul,ur,d,c);
            near(fp[0],2*c,1e-14,"positive streaming is upwind");
            near(fp[d+1],2*c*c,1e-14,"physical streaming momentum flux");
            ul[d+1] *= -1; ur[d+1] *= -1;
            const auto fm = rm::hll(ul,ur,d,c);
            near(fm[0],-5*c,1e-14,"negative streaming is upwind");
            const auto perpendicular = rm::hll(ul,ur,(d+1)%3,c);
            for (Real f : perpendicular) require(f == 0,"zero-speed degeneracy");
        }
        const state oblique{1, c*0.6, c*0.8,0};
        const auto waves = rm::physical_flux(oblique,0,c,0.2*c);
        near(waves.plus,0.4*c,1e-7,"oblique moving-face wave speed");
        near(waves.minus,0.4*c,1e-7,"oblique coalesced wave speed");
        const auto vacuum = rm::hll({},{},0,c);
        for (Real f : vacuum) require(f == 0,"vacuum flux");
    }
    // Independent Levermore pressure-tensor form from S&O equations (14), (17).
    std::mt19937_64 random(8128);
    std::uniform_real_distribution<Real> dist(-1,1);
    for (int n=0; n<2000; ++n) {
        vec f{dist(random),dist(random),dist(random)};
        const Real norm=std::hypot(f[0],f[1],f[2]);
        const Real magnitude=std::abs(dist(random));
        for (auto& x:f) x *= magnitude/norm;
        const Real E=1+std::abs(dist(random)), c=19;
        const state u{E,c*E*f[0],c*E*f[1],c*E*f[2]};
        const Real chi=(3+4*magnitude*magnitude)/(5+2*std::sqrt(4-3*magnitude*magnitude));
        for (int d=0;d<3;++d) {
            const auto h=rm::physical_flux(u,d,c);
            require(h.minus >= -c*(1+1e-14) && h.plus <= c*(1+1e-14),"causal wave speeds");
            for (int j=0;j<3;++j) {
                Real P=E*0.5*(3*chi-1)*f[d]*f[j]/(magnitude*magnitude);
                if(d==j) P+=E*0.5*(1-chi);
                require(std::abs(h.flux[j+1]-c*c*P)<1e-11*c*c*E,"Hanawa closure matches Levermore");
            }
        }
    }
}

void reconstruction() {
    require(rm::minmod_theta(1,2)==1.3,"limiter theta 1.3");
    require(rm::minmod_theta(-1,2)==0,"limiter at extremum");
    const auto [m,p] = rm::reconstruct({2,0.1,0,0},{3,0.2,0,0},{4,0.3,0,0},10);
    near(m[0],2.5,1e-14,"linear energy minus face");
    near(p[0],3.5,1e-14,"linear energy plus face");
    near(m[1],10*2.5*0.15,1e-14,"reconstructed reduced flux converted back");
    near(p[1],10*3.5*0.25,1e-14,"plus reduced flux converted back");
    std::mt19937_64 random(12345);
    std::uniform_real_distribution<Real> dist(-1,1);
    auto q = [&] {
        state v{std::exp(4*dist(random)),dist(random),dist(random),dist(random)};
        const Real norm=std::hypot(v[1],v[2],v[3]);
        const Real mag=0.999999999999999;
        for(int f=1;f<4;++f) v[f] *= mag/norm;
        return v;
    };
    for(int n=0;n<4000;++n) {
        const auto [l,r] = rm::reconstruct(q(),q(),q(),13);
        admissible(l,13); admissible(r,13);
    }
}

void thermal_sources() {
    for (Real e : {1e-20, 1e-6, 1.0, 1e12, 1e30}) {
        for (Real ratio : {1e-20, 1e-4, 1.0, 1e4, 1e20}) {
            for (Real eta : {1e-14, 1e-3, 1.0, 1e8, 1e20}) {
                const Real E=e*ratio;
                const Real log_alpha=std::log(2*E)-4*std::log(e);
                const auto next=rm::thermal_exchange(e,E,eta,log_alpha);
                require(next.gas>0 && next.radiation>0,"positive thermal exchange");
                near(next.gas+next.radiation,e+E,2e-12,"thermal total energy");
                // Check the BE equation in its bounded form, independently in long double.
                const long double w=(long double)eta/(1+(long double)eta);
                const long double B=std::exp((long double)log_alpha+4*std::log((long double)next.gas));
                const long double residual=next.radiation-((long double)E/(1+(long double)eta)+w*B);
                require(std::abs(residual)<2e-12L*(e+E),"backward Euler thermal residual");
            }
        }
    }
    const auto zero = rm::thermal_exchange(2,3,0,0);
    require(zero.gas==2 && zero.radiation==3,"zero-opacity thermal identity");
    const auto lte=rm::thermal_exchange(2,16,1e18,0);
    near(lte.gas,2,1e-13,"stiff LTE gas unchanged");
    near(lte.radiation,16,1e-13,"stiff LTE radiation unchanged");
    const auto marshak=rm::thermal_exchange(1,3,10,0,true);
    near(marshak.gas,41./21,1e-14,"Marshak linear heat capacity");
    near(marshak.radiation,43./21,1e-14,"Marshak energy balance");
}

void coupled_sources() {
    const Real c=17, rho=10, e=100;
    const state u{4,20,-10,5};
    const vec momentum{1,-2,0.5};
    const Real K=rm::dot(momentum,momentum)/(2*rho);
    const auto zero=rm::couple(u,momentum,e+K,e,rho,0,0,1,c,0);
    require(zero.radiation==u && zero.momentum==momentum && zero.gas_energy==e+K,"zero-coupling identity");
    for (const auto& opac : {std::pair{0.,0.7}, std::pair{0.7,0.7}, std::pair{0.2,0.7}}) {
        const Real dt=0.01;
        const auto out=rm::couple(u,momentum,e+K,e,rho,opac.first,opac.second,dt,c,std::log(4.)-4*std::log(e));
        near(out.radiation[0]+out.gas_energy,u[0]+e+K,2e-13,"coupled total energy conserved");
        for(int d=0;d<3;++d) near(out.momentum[d]+out.radiation[d+1]/c/c,momentum[d]+u[d+1]/c/c,2e-13,"coupled total momentum conserved");
        const Real kinetic=rm::dot(out.momentum,out.momentum)/(2*rho);
        near(out.gas_energy-kinetic,out.internal_energy,2e-13,"thermal energy includes kinetic change");
        admissible(out.radiation,c);
    }
    const Real chi=2,dt=0.03;
    const auto static_gas=rm::couple(u,{0,0,0},e,e,rho,chi,chi,dt,c,std::log(4.)-4*std::log(e));
    for(int d=0;d<3;++d) {
        near(static_gas.radiation[d+1],u[d+1]/(1+dt*c*chi),1e-14,"implicit flux damping");
        near(static_gas.momentum[d],(u[d+1]-static_gas.radiation[d+1])/c/c,1e-14,"physical momentum conversion c squared");
    }
    const auto stiff=rm::couple({1,0,0,0},{0,0,0},100,100,1,1e15,1e15,1,c,-4*std::log(100.));
    near(stiff.radiation[0]+stiff.gas_energy,101,2e-13,"stiff coupled energy conserved");
    admissible(stiff.radiation,c);
}

void periodic_transport() {
    // A beam at non-unit c must translate exactly one box length per crossing time.
    constexpr int N=64;
    const Real c=17,dx=1./N;
    std::vector<state> u(N),next(N),minus(N),plus(N),flux(N);
    for(int i=0;i<N;++i) {
        Real E=1+0.2*std::sin(2*pi_R*(i+0.5)*dx);
        u[i]={E,c*E,0,0};
    }
    const auto initial=u;
    const int steps=static_cast<int>(std::ceil((1/c)/rm::transport_timestep(dx,c,0.4)));
    const Real dt=1/(c*steps);
    for(int step=0;step<steps;++step) {
        for(int i=0;i<N;++i) {
            const auto faces=rm::reconstruct(rm::primitive(u[(i+N-1)%N],c),rm::primitive(u[i],c),rm::primitive(u[(i+1)%N],c),c);
            minus[i]=faces.first; plus[i]=faces.second;
        }
        for(int i=0;i<N;++i) flux[i]=rm::hll(plus[(i+N-1)%N],minus[i],0,c);
        for(int i=0;i<N;++i) {
            for(int f=0;f<4;++f) next[i][f]=u[i][f]-(dt/dx)*(flux[(i+1)%N][f]-flux[i][f]);
            admissible(next[i],c);
        }
        u.swap(next);
    }
    Real sum=0,error=0;
    for(int i=0;i<N;++i) { sum+=u[i][0];error+=std::abs(u[i][0]-initial[i][0])/N; }
    near(sum,Real(N),2e-13,"periodic transport conservation");
    require(error<0.025,"beam crossing speed and profile");
    near(rm::transport_timestep(2,10,0.4),1./60,1e-14,"3D light CFL bound");
    near(rm::transport_timestep(2,10,0.4,10),1./120,1e-14,"moving-grid CFL bound");
}

void debug_guards() {
#ifndef NDEBUG
    const Real nan=std::numeric_limits<Real>::quiet_NaN();
    bool caught=false;
    try { (void)expectPositive(nan); } catch(const std::runtime_error&) {caught=true;}
    require(caught,"expectPositive catches NaN");
    caught=false;
    try { (void)expectRange(0.,nan,1.); } catch(const std::runtime_error&) {caught=true;}
    require(caught,"expectRange catches NaN");
    caught=false;
    try { (void)expectFinite(std::numeric_limits<Real>::infinity()); } catch(const std::runtime_error&) {caught=true;}
    require(caught,"expectFinite catches infinity");
#endif
}

int main() {
    try {
        closure_and_hll(); reconstruction(); thermal_sources(); coupled_sources(); periodic_transport(); debug_guards();
        std::cout << checks << " radiation numerical checks passed\n";
    } catch(const std::exception& e) { std::cerr << "FAILED after " << checks << " checks: " << e.what() << '\n'; return 1; }
}
