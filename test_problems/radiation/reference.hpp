// Copyright (c) 2026 AUTHORS. Distributed under the Boost Software License, Version 1.0.
#pragma once
#include "profiles.hpp"
#include "octotiger/math/Debug.hpp"
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace radiationTests {
// Binary format OTRAD001: uint32 N, 8 binary64 parameters in declaration order,
// then 2 snapshots x 4 fields x N^3 binary64 cell averages (z fastest).
// All integers and IEEE doubles are little endian; no native struct padding.
inline void putInteger(std::ostream& out, std::uint64_t value, int bytes) {
	for (int b=0;b<bytes;++b) out.put(char((value>>(8*b))&255));
}
inline std::uint64_t getInteger(std::istream& in, int bytes) {
	std::uint64_t value=0;
	for (int b=0;b<bytes;++b) {
		int const ch=in.get();
		if (ch==EOF) throw std::runtime_error("Truncated radiation reference file");
		value|=std::uint64_t(ch)<<(8*b);
	}
	return value;
}
inline void putDouble(std::ostream& out, double v) { putInteger(out,std::bit_cast<std::uint64_t>(v),8); }
inline double getDouble(std::istream& in) {
	double const v=std::bit_cast<double>(getInteger(in,8));
	if (!std::isfinite(v)) throw std::runtime_error("Nonfinite radiation reference value");
	return v;
}
struct ReferenceData {
	std::uint32_t n=0;
	Parameters p;
	std::vector<double> values;
	std::size_t cells() const { return std::size_t(n)*n*n; }
	std::size_t index(int snapshot,int field,int i,int j,int k) const {
		auto wrap=[this](int a){return (a%int(n)+int(n))%int(n);};
		return std::size_t(snapshot*4+field)*cells()+wrap(k)+std::size_t(n)*(wrap(j)+std::size_t(n)*wrap(i));
	}
	void write(std::string const& filename) const {
		static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
		p.validate();
		if (n<4 || n>512 || n%2 || values.size()!=8*cells()) throw std::runtime_error("Invalid reference shape");
		std::ofstream out(filename,std::ios::binary|std::ios::trunc);
		out.exceptions(std::ios::badbit|std::ios::failbit);
		out.write("OTRAD001",8); putInteger(out,n,4);
		for (double v:{p.length,p.c,p.chi,p.width,p.background,p.amplitude,p.time,p.luminosity}) putDouble(out,v);
		for (double v:values) {
			if (!std::isfinite(v)) throw std::runtime_error("Cannot write nonfinite reference");
			putDouble(out,v);
		}
		out.close();
	}
	static ReferenceData read(std::string const& filename) {
		std::ifstream in(filename,std::ios::binary);
		if (!in) throw std::runtime_error("Cannot open radiation reference: "+filename+"; run gen_radiation_reference first");
		char magic[8]; in.read(magic,8);
		if (!in || std::string(magic,8)!="OTRAD001") throw std::runtime_error("Invalid radiation reference version");
		ReferenceData r; r.n=getInteger(in,4);
		if (r.n<4 || r.n>512 || r.n%2) throw std::runtime_error("Invalid radiation reference dimensions");
		for (double* v:{&r.p.length,&r.p.c,&r.p.chi,&r.p.width,&r.p.background,&r.p.amplitude,&r.p.time,&r.p.luminosity}) *v=getDouble(in);
		r.p.validate();
		in.seekg(0,std::ios::end);
		if (in.tellg()!=std::streamoff(76+8*8*r.cells())) throw std::runtime_error("Radiation reference file size mismatch");
		in.seekg(76); r.values.resize(8*r.cells());
		for (double& v:r.values) v=getDouble(in);
		return r;
	}
	State average(Point x,double dx,double t) const {
		FpeGuard fpeGuard{};
		double const tol=256*std::numeric_limits<double>::epsilon();
		int snapshot;
		if (std::abs(t)<=tol*p.time) snapshot=0;
		else if (std::abs(t-p.time)<=tol*p.time) snapshot=1;
		else throw std::runtime_error("Reference contains only t=0 and its configured final time");
		double const h=p.length/n;
		int const count=std::lround(dx/h);
		if (count<1 || count>int(n) || std::abs(dx/h-count)>tol*n) throw std::runtime_error("Reference resolution does not cover this cell");
		std::array<int,3> lo;
		for (int d=0;d<3;++d) {
			double const q=(x[d]+p.length/2-dx/2)/h;
			lo[d]=std::lround(q);
			if (std::abs(q-lo[d])>tol*n) throw std::runtime_error("Reference and simulation grids are misaligned");
		}
		State u{};
		for (int f=0;f<4;++f) for (int i=0;i<count;++i) for (int j=0;j<count;++j) for (int k=0;k<count;++k)
			u[f]+=values[index(snapshot,f,lo[0]+i,lo[1]+j,lo[2]+k)];
		for (double& v:u) v/=double(count)*count*count;
		return u;
	}
};
} // namespace radiationTests
