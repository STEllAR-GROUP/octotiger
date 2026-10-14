// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include <octotiger/ssp_history_amr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "SSP history AMR test failed: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}
bool close(double a, double b) {
    return std::abs(a-b) <= 8e-14 * std::max({1.0, std::abs(a), std::abs(b)});
}
double average(double x, double y, double z, double h, bool affine) {
    return 2 + 0.4*x + 0.3*y + 0.2*z + (affine ? 0.0 :
        0.1*(x*x+h*h/12) + 0.2*x*y + 0.15*(z*z+h*h/12));
}
double reconstruction_error(int n, bool affine) {
    const double h=1.0/n;
    std::vector<double> p(size_t(n)*n*n), k(p.size());
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            for (int l=0; l<n; ++l) {
                const int index=(i*n+j)*n+l;
                p[index]=average((i+0.5)*h,(j+0.5)*h,(l+0.5)*h,h,affine);
                k[index]=3*p[index]-5;
            }
        }
    }
    double error=0;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            for (int l=0; l<n; ++l) {
                const int index=(i*n+j)*n+l;
                const auto fine=ssp_multistep::prolong_history_cell(p,k,n,{i,j,l});
                double p_mean=0,k_mean=0;
                for (int child=0; child<8; ++child) {
                    const double x=(i+0.5+((child&1)?0.25:-0.25))*h;
                    const double y=(j+0.5+((child&2)?0.25:-0.25))*h;
                    const double z=(l+0.5+((child&4)?0.25:-0.25))*h;
                    const double exact=average(x,y,z,h/2,affine);
                    error+=std::abs(fine.state[child]-exact)/(8.0*p.size());
                    require(close(fine.rhs[child],3*fine.state[child]-5),
                        "state and derivative use a common linear mapping");
                    if (affine) {
                        require(close(fine.state[child],exact),
                            "affine exactness including block edges and corners");
                    }
                    p_mean+=fine.state[child]/8;
                    k_mean+=fine.rhs[child]/8;
                }
                require(close(p_mean,p[index]),"history state octet conservation");
                require(close(k_mean,k[index]),"history RHS octet conservation");
            }
        }
    }
    return error;
}
void constant_state_variable_rhs() {
    constexpr int n=8;
    std::vector<double> p(n*n*n,2),k(p.size());
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            for (int l=0; l<n; ++l) {
                k[(i*n+j)*n+l]=1+i+2*j+3*l;
            }
        }
    }
    for (const auto cell : {std::array<int,3>{0,0,0},{3,4,5},{7,7,7}}) {
        const auto fine=ssp_multistep::prolong_history_cell(p,k,n,cell);
        const double coarse=1+cell[0]+2*cell[1]+3*cell[2];
        for (int child=0; child<8; ++child) {
            const double exact=coarse+((child&1)?0.25:-0.25)
                +2*((child&2)?0.25:-0.25)+3*((child&4)?0.25:-0.25);
            require(close(fine.state[child],2),"constant state preserved");
            require(close(fine.rhs[child],exact),"affine RHS preserved with constant state");
        }
    }
    for (const auto cell : {std::array<int,3>{-1,0,0},{0,n,0},{0,0,-1}}) {
        bool rejected=false;
        try {
            ssp_multistep::prolong_history_cell(p,k,n,cell);
        } catch (const std::runtime_error&) {
            rejected=true;
        }
        require(rejected,"invalid cell rejected before indexing");
    }
}
void positivity_cases() {
    constexpr int n=3;
    std::vector<double> p(n*n*n), k(p.size());
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            for (int l=0; l<n; ++l) {
                p[(i*n+j)*n+l]=1+5*i;
            }
        }
    }
    auto fine=ssp_multistep::prolong_history_cell(p,k,n,{0,1,1});
    require(*std::min_element(fine.state.begin(),fine.state.end())<0,
        "one-sided boundary counterexample is exercised");
    double theta=ssp_multistep::history_pair_positivity_factor(fine,1,0,1);
    require(theta>0 && theta<1,"boundary state activates conservative limiter");
    double mean=0;
    for (int child=0; child<8; ++child) {
        const double value=1+theta*(fine.state[child]-1);
        require(value>=0,"boundary counterexample repaired");
        mean+=value/8;
    }
    require(close(mean,1),"boundary limiting preserves coarse mean");
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            for (int l=0; l<n; ++l) {
                const int index=(i*n+j)*n+l;
                p[index]=1+i;
                k[index]=(i==0 ? 100.0 : 0.0)-p[index];
            }
        }
    }
    fine=ssp_multistep::prolong_history_cell(p,k,n,{1,1,1});
    theta=ssp_multistep::history_pair_positivity_factor(fine,2,-2,1);
    require(theta==0,"zero Euler mean selects constant conservative pair");
    double p_mean=0,k_mean=0;
    for (int child=0; child<8; ++child) {
        const double state=2+theta*(fine.state[child]-2);
        const double rhs=-2+theta*(fine.rhs[child]+2);
        require(state>=0 && state+rhs>=0,"historical Euler endpoint repaired");
        p_mean+=state/8;
        k_mean+=rhs/8;
    }
    require(close(p_mean,2) && close(k_mean,-2),"paired limiting preserves both means");
}
}
int main() {
    using ssp_multistep::ideal_gas_history_admissible;
    require(!ideal_gas_history_admissible(1e200,1e200,2e200,0,0),
        "thermal cone check avoids inf >= inf");
    require(ideal_gas_history_admissible(1e200,1e200,1e200,0,0),
        "large finite positive thermal energy accepted");
    require(!ideal_gas_history_admissible(1e300,1e-300,2,0,0),
        "thermal cone check handles disparate scales");
    require(ideal_gas_history_admissible(1e300,1e-300,1,0,0),
        "disparate-scale positive thermal energy accepted");
    constant_state_variable_rhs();
    positivity_cases();
    reconstruction_error(8,true);
    const double a=reconstruction_error(8,false);
    const double b=reconstruction_error(16,false);
    const double c=reconstruction_error(32,false);
    require(a/b>3.5 && b/c>3.5,"second-order smooth reconstruction across all cells");
    bool rejected=false;
    try {
        std::vector<double> bad(4);
        ssp_multistep::prolong_history_cell(bad,bad,2,{0,0,0});
    } catch (const std::runtime_error&) {
        rejected=true;
    }
    require(rejected,"invalid shape rejected");
    std::printf("PASS SSP history AMR: affine exactness, common state/RHS stencil, conservation, pair positivity; smooth ratios %.6f %.6f\n",a/b,b/c);
}
