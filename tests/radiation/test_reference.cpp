// Independent contracts for profiles, modal evolution, binary data and FFT units.
#include "octotiger/test_problems/radiation/reference.hpp"
#include <complex>
#include <iostream>
#include <sstream>
#include <filesystem>
using namespace radiation_tests;
void require(bool yes,char const* text) { if(!yes) throw std::runtime_error(text); }
void near(double a,double b,double tol,char const* text) { require(std::isfinite(a)&&std::abs(a-b)<=tol,text); }
int main(int argc,char** argv) {
	try {
		require(argc==2,"Pass a generated reference file");
		auto const ref=reference_data::read(argv[1]); auto const& p=ref.p;
		double const h=p.length/ref.n;
		// Exact mode limits: k=0; chi=0; critical damping; huge chi with finite slow decay.
		auto m=telegraph_mode(0,.7,3,2); near(m[0],1,0,"zero mode");
		m=telegraph_mode(12,.2,1,0); near(m[0],std::cos(.4),2e-15,"undamped mode");
		m=telegraph_mode(3,.7,1,2); near(m[0],1.7*std::exp(-.7),2e-15,"critical damping");
		m=telegraph_mode(1,1,1,1e12); near(m[0],1-1./3e12,2e-15,"overdamped stability");
		for (int d=0;d<3;++d) {
			point x{.12,-.43,.2}, y=x; y[d]+=p.length;
			auto a=streaming_wave(x,.37,h,p), b=streaming_wave(y,.37,h,p);
			for (int f=0;f<4;++f) near(a[f],b[f],2e-14,"oblique wave periodicity");
		}
		parameters units=p; units.c=17;
		auto beam=streaming_front({0,0,0},0,h,units);
		near(beam[1],17*beam[0],1e-14,"physical front flux");
		near(beam[0],.5+.5e-10,1e-14,"front cell average");
		auto center=equilibrium_sphere({0,0,0},p);
		for(int d=1;d<4;++d) near(center[d],0,0,"finite bulb center");
		// Independent finite differences check div F=q and dE/dr=-3 chi Fr/c.
		point x{.21,.17,-.13}; double div=0; double const eps=1e-5;
		for(int d=0;d<3;++d) {
			point lo=x,hi=x; lo[d]-=eps;hi[d]+=eps;
			auto a=equilibrium_sphere(lo,p), b=equilibrium_sphere(hi,p);
			div+=(b[d+1]-a[d+1])/(2*eps);
			near((b[0]-a[0])/(2*eps),-3*p.chi/p.c*equilibrium_sphere(x,p)[d+1],1e-9,"diffusion gradient");
		}
		double const r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
		near(div,p.luminosity*std::exp(-r2/(p.width*p.width))/std::pow(std::sqrt(pi)*p.width,3),1e-9,"matching bulb source");
		long double e0=0,e1=0;
		for(std::size_t r=0;r<ref.cells();++r) {
			e0+=ref.values[r]; e1+=ref.values[4*ref.cells()+r];
			for(int d=1;d<4;++d) near(ref.values[d*ref.cells()+r],0,0,"initial physical flux is zero");
		}
		near(double(e0/ref.cells()),p.background+p.amplitude*std::pow(std::sqrt(pi)*p.width/p.length,3),2e-14,"Gaussian normalization");
		near(double((e1-e0)/ref.cells()),0,2e-14,"mean energy conservation");
		// Direct DFT of selected modes, independent of FFTW and its normalization.
		// Evolve E,F with 4-variable RK4 rather than reusing telegraph_mode.
		using complex=std::complex<long double>; complex const I(0,1);
		for (point mode:{point{1,0,0},point{1,2,0},point{2,1,3}}) {
			std::array<complex,4> initial{},final{};
			for(unsigned i=0;i<ref.n;++i) for(unsigned j=0;j<ref.n;++j) for(unsigned k=0;k<ref.n;++k) {
				complex const phase=std::exp(-I*(2.L*std::numbers::pi_v<long double>/ref.n)*static_cast<long double>(mode[0]*i+mode[1]*j+mode[2]*k));
				for(int f=0;f<4;++f) {
					initial[f]+=phase*(long double)(ref.values[ref.index(0,f,i,j,k)]-(f==0?p.background:0));
					final[f]+=phase*(long double)(ref.values[ref.index(1,f,i,j,k)]-(f==0?p.background:0));
				}
			}
			point wave;for(int d=0;d<3;++d) wave[d]=2*pi*mode[d]/p.length;
			auto rhs=[&](std::array<complex,4> const& u) {
				std::array<complex,4> v{};
				for(int d=0;d<3;++d) {
					v[0]-=I*(long double)wave[d]*u[d+1];
					v[d+1]=-I*(long double)(p.c*p.c/3*wave[d])*u[0]-(long double)(p.c*p.chi)*u[d+1];
				}
				return v;
			};
			auto u=initial; long double const dt=p.time/2000;
			for(int step=0;step<2000;++step) {
				auto a=rhs(u), v=u;
				for(int f=0;f<4;++f) v[f]=u[f]+dt/2*a[f];
				auto b=rhs(v);
				for(int f=0;f<4;++f) v[f]=u[f]+dt/2*b[f];
				auto c=rhs(v);
				for(int f=0;f<4;++f) v[f]=u[f]+dt*c[f];
				auto d=rhs(v);
				for(int f=0;f<4;++f) u[f]+=dt/6*(a[f]+2.L*b[f]+2.L*c[f]+d[f]);
			}
			for(int f=0;f<4;++f) require(std::abs(final[f]-u[f])<1e-10*std::max(1.L,std::abs(initial[0])),"FFT telegraph mode disagrees with independent ODE");
		}
		// Coarse-grid initialization conservatively averages the finest binary cells.
		point pos{-p.length/2+h,-p.length/2+h,-p.length/2+h};
		auto coarse=ref.average(pos,2*h,0); state sum{};
		for(int i=0;i<2;++i) for(int j=0;j<2;++j) for(int k=0;k<2;++k)
			for(int f=0;f<4;++f) sum[f]+=ref.values[ref.index(0,f,i,j,k)]/8;
		for(int f=0;f<4;++f) near(coarse[f],sum[f],1e-14,"coarse initialization average");
		bool rejected=false;try {ref.average(pos,2*h,p.time/2);} catch(std::runtime_error const&){rejected=true;}
		require(rejected,"reject wrong snapshot time");
		std::string const bad=std::string(argv[1])+".truncated";
		{std::ofstream out(bad,std::ios::binary);out<<"OTRAD001";}
		rejected=false;try {reference_data::read(bad);} catch(std::runtime_error const&){rejected=true;}
		std::filesystem::remove(bad);require(rejected,"reject truncated file");
		std::cout<<"Reference, Fourier, profile and binary contracts passed\n";
	} catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
