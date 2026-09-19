"""Exercise real rad_grid transport and conservation methods without HPX.

Only the options, mesh geometry and distributed infrastructure are fixtures.
The evolution, M1 closure, test sources, face budgets and volume reductions are
compiled from production sources. Run with unittest discovery or this file.
These serial tests do not replace the distributed/AMR application tests.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verification_results" / "radiation"))
from results import CONSERVATION_FILE, read_conservation


def extract_method(source, signature):
    begin = source.index(signature)
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
        end += 1
    return source[begin:end]


FIXTURE = r'''
#include "octotiger/math/Debug.hpp"
#include "octotiger/radiation/m1.hpp"
#include "octotiger/radiation/conservation.hpp"
#include "octotiger/test_problems/radiation/profiles.hpp"
#include <atomic>
#include <functional>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#define PROFILE()
using integer = long long;
constexpr int INX=8, NRF=4, NDIM=3, NCHILD=8, H_BW=3, RAD_BW=3;
using M1=RadiationM1<Real,NDIM>;
constexpr int RAD_NX=INX+2*RAD_BW, RAD_N3=RAD_NX*RAD_NX*RAD_NX;
constexpr int H_NX=RAD_NX, H_N3=RAD_N3;
constexpr int HS_NX=INX/2+2*H_BW, HS_N3=HS_NX*HS_NX*HS_NX;
constexpr int XDIM=0, YDIM=1, ZDIM=2, RADIATION_EQUILIBRIUM_SPHERE=3;
integer rindex(integer i,integer j,integer k) { return k+RAD_NX*(j+RAD_NX*i); }
integer hindex(integer i,integer j,integer k) { return rindex(i,j,k); }
struct options_fixture { Real cfl=.25; int problem=0; };
options_fixture& opts() { static options_fixture o; return o; }
struct constants_fixture { Real c=1; };
constants_fixture& physcon() { static constants_fixture p; return p; }
radiationTests::Parameters parameters;
radiationTests::Parameters radiationTestParameters() { return parameters; }
struct silo_var_t {};
namespace geo {
constexpr int MINUS=0, PLUS=1;
struct direction {}; struct dimension {}; struct octant {};
struct face {
    int index;
    face(int value): index(value) {}
    int get_dimension() const { return index/2; }
    int get_side() const { return index%2; }
    operator int() const { return index; }
};
}
#define private public
'''


CHECKS = r'''
#undef private
using radiationConservation::Totals;
using radiationConservation::Ledger;
using radiationConservation::Moments;
using state = radiationTests::State;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void near(long double actual, long double expected, long double scale,
          const std::string& label, long double tolerance=3e-12L) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>tolerance*std::abs(scale)) {
        std::ostringstream message;
        message << std::setprecision(20) << label << ": actual=" << actual
                << " expected=" << expected << " scale=" << scale;
        throw std::runtime_error(message.str());
    }
}
long double volume(double dx) {
    return static_cast<long double>(INX)*INX*INX*dx*dx*dx;
}
template<class Function> void interior(Function function) {
    for(int i=RAD_BW;i<RAD_BW+INX;++i)
        for(int j=RAD_BW;j<RAD_BW+INX;++j)
            for(int k=RAD_BW;k<RAD_BW+INX;++k) function(rindex(i,j,k));
}
void initialize(rad_grid& g, double dx, const std::function<state(radiationTests::Point)>& profile,
                double xshift=0) {
    std::vector<std::vector<Real>> x(3,std::vector<Real>(H_N3));
    for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k) {
        auto r=rindex(i,j,k);
        radiationTests::Point point{(i-RAD_BW+.5-INX/2.)*dx+xshift,
                                    (j-RAD_BW+.5-INX/2.)*dx,
                                    (k-RAD_BW+.5-INX/2.)*dx};
        auto u=profile(point);
        for(int d=0;d<3;++d) x[d][r]=point[d];
        for(int f=0;f<4;++f) g.U[f][r]=u[f];
    }
    g.set_dx(dx); g.set_X(x);
}
void periodic(rad_grid& g) {
    auto wrap=[](int i) { return RAD_BW+(i-RAD_BW+INX)%INX; };
    for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
        for(int f=0;f<4;++f)
            g.U[f][rindex(i,j,k)]=g.U[f][rindex(wrap(i),wrap(j),wrap(k))];
}
void account_all_faces(rad_grid& g, double dt) {
    for(int face=0;face<6;++face) g.accountBoundaryFlux(dt,geo::face(face));
}
void balance(const Totals& initial, const Totals& final, double c,
             const std::string& label) {
    for(int f=0;f<4;++f) {
        long double residual=final.value[f]-initial.value[f]
            +final.boundary[f]-initial.boundary[f]-final.source[f]+initial.source[f];
        long double scale=std::abs(initial.value[0])*(f==0?1:c)
            +std::abs(final.boundary[f])+std::abs(final.source[f]);
        near(residual,0,scale,label+" field "+std::to_string(f));
    }
}

void volume_and_faces() {
    double const dx=.37, dt=.013;
    physcon().c=7;
    rad_grid g(dx);
    initialize(g,dx,[](auto){ return state{2,.3,-.2,0}; });
    // Ghost cells must not enter the global volume or state sums.
    for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
        if(i<RAD_BW||j<RAD_BW||k<RAD_BW||i>=RAD_BW+INX||j>=RAD_BW+INX||k>=RAD_BW+INX)
            for(int f=0;f<4;++f) g.U[f][rindex(i,j,k)]=1e20;
    auto total=g.takeConservation();
    near(total.volume,volume(dx),volume(dx),"interior volume");
    state const initial{2,.3,-.2,0};
    for(int f=0;f<4;++f) near(total.value[f],initial[f]*volume(dx),volume(dx),"interior sum");

    // An affine flux encodes all coordinates, catching wrong axes, face
    // offsets, ghost-cell inclusion, signs, area and timestep factors.
    for(int d=0;d<3;++d)for(int f=0;f<4;++f)
        for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
            g.flux[d][f][rindex(i,j,k)]=1000*(d+1)+100*(f+1)+i+10*j+100*k;
    for(int face=0;face<6;++face) {
        g.accountBoundaryFlux(dt,geo::face(face));
        auto total=g.takeConservation();
        int const d=face/2, side=face%2;
        double midpoint=RAD_BW+(INX-1)/2.;
        std::array<double,3> q{midpoint,midpoint,midpoint};
        q[d]=RAD_BW+side*INX;
        for(int f=0;f<4;++f) {
            long double expected=(side?1:-1)*static_cast<long double>(dt)*dx*dx*INX*INX
                *(1000*(d+1)+100*(f+1)+q[0]+10*q[1]+100*q[2]);
            near(total.boundary[f],expected,expected,"face "+std::to_string(face));
            near(total.source[f],0,1,"face has no source");
        }
        auto drained=g.takeConservation();
        for(int f=0;f<4;++f) near(drained.boundary[f],0,1,"boundary drained once");
    }
}

void streaming(bool oblique) {
    double const dx=.37, c=7;
    physcon().c=c;
    rad_grid g(dx);
    radiationTests::Point mode=oblique?radiationTests::Point{2,-3,1}:radiationTests::Point{1,0,0};
    double norm=std::sqrt(radiationTests::dot(mode,mode));
    initialize(g,dx,[&](auto x) {
        double E=1+.25*std::cos(2*std::numbers::pi*radiationTests::dot(mode,x)/(INX*dx));
        return state{E,c*E*mode[0]/norm,c*E*mode[1]/norm,c*E*mode[2]/norm};
    });
    Ledger history;
    auto initial=history.consume(g.takeConservation());
    auto final=initial;
    for(int step=0;step<32;++step) {
        periodic(g);
        double dt=g.maxTimestep(0)*.71;
        g.compute_flux(0);
        g.advance(dt,0);
        final=history.consume(g.takeConservation());
        balance(initial,final,c,"periodic streaming");
    }
    for(int f=0;f<4;++f) {
        near(final.boundary[f],0,1,"periodic boundary budget");
        near(final.source[f],0,1,"streaming source budget");
    }
    if(!oblique) for(int f=2;f<4;++f)
        near(final.value[f],0,c*initial.value[0],"initially zero component");

    // A state corruption outside the update must appear in the residual;
    // defining the boundary budget by global state differences would hide it.
    double const corruption=.125;
    g.U[0][rindex(RAD_BW+2,RAD_BW+1,RAD_BW+3)]+=corruption;
    auto damaged=history.consume(g.takeConservation());
    long double residual=damaged.value[0]-initial.value[0]+damaged.boundary[0]-damaged.source[0];
    near(residual,static_cast<long double>(corruption)*dx*dx*dx,
         static_cast<long double>(corruption)*dx*dx*dx,"injected drift detected",3e-10L);
    require(std::abs(residual)>1e-5L*initial.value[0],"injected drift was hidden");
}

void damping_and_injection() {
    double const dx=.37, dt=.013, c=7;
    parameters={}; parameters.c=c; parameters.chi=2.3;
    parameters.length=INX*dx; parameters.width=.31; parameters.luminosity=.27;
    physcon().c=c; opts().problem=2;
    rad_grid g(dx);
    state const u{2,.8*c,-.3*c,.2*c};
    initialize(g,dx,[&](auto){ return u; });
    Ledger history;
    auto initial=history.consume(g.takeConservation());
    g.applyRegressionSource(dt);
    auto damped=history.consume(g.takeConservation());
    double factor=1/(1+dt*c*parameters.chi);
    near(damped.source[0],0,initial.value[0],"scattering preserves energy");
    for(int f=1;f<4;++f) {
        near(damped.value[f],initial.value[f]*factor,std::abs(initial.value[f]),"physical F damping");
        near(damped.source[f],initial.value[f]*(factor-1),std::abs(initial.value[f]),"damping source units");
    }
    balance(initial,damped,c,"scattering budget");
    opts().problem=RADIATION_EQUILIBRIUM_SPHERE;
    g.applyRegressionSource(dt);
    auto forced=history.consume(g.takeConservation());
    // Integrate the prescribed Gaussian analytically over the entire cube.
    long double added=dt*parameters.luminosity*std::pow(std::erf(INX*dx/(2*parameters.width)),3);
    near(forced.source[0],added,added,"Gaussian luminosity integral",3e-10L);
    balance(initial,forced,c,"injection plus damping budget");
    auto repeated=history.consume(g.takeConservation());
    for(int f=0;f<4;++f) near(repeated.source[f],forced.source[f],std::abs(initial.value[0])*c,"source drained once");
}

void sphere(bool cgs) {
    double const dx=cgs?2.5e11:.37;
    parameters={}; parameters.length=INX*dx;
    parameters.c=cgs?2.99792458e10:7;
    parameters.width=cgs?3e11:.31;
    parameters.chi=cgs?1e-10:2.3;
    parameters.luminosity=cgs?1e38:.27;
    parameters.background=cgs?1e-4:1;
    physcon().c=parameters.c;
    opts().problem=RADIATION_EQUILIBRIUM_SPHERE;
    rad_grid g(dx);
    initialize(g,dx,[&](auto x){ return radiationTests::sphereAverage(x,dx,parameters); });
    Ledger history;
    auto initial=history.consume(g.takeConservation());
    auto final=initial;
    double elapsed=0;
    for(int step=0;step<8;++step) {
        double dt=g.maxTimestep(0)*.71;
        g.compute_flux(0);
        account_all_faces(g,dt);
        g.advance(dt,0);
        g.applyRegressionSource(dt);
        elapsed+=dt;
        final=history.consume(g.takeConservation());
        balance(initial,final,parameters.c,"open sphere");
    }
    require(final.boundary[0]>0,"sphere must transport energy outwards");
    require(final.source[0]>0,"sphere must inject energy");
    long double expected=elapsed*parameters.luminosity
        *std::pow(std::erf(INX*dx/(2*parameters.width)),3);
    near(final.source[0],expected,expected,"sphere source dimensions",3e-10L);
}

void rotation() {
    double const dx=.37, c=7, dt=.013, omega=.71;
    physcon().c=c;
    rad_grid g(dx);
    state const u{2,.8*c,-.3*c,.2*c};
    initialize(g,dx,[&](auto){ return u; });
    auto initial=g.takeConservation();
    g.compute_flux(omega);
    account_all_faces(g,dt);
    g.advance(dt,omega);
    auto final=g.takeConservation();
    double const cs=std::cos(omega*dt), sn=std::sin(omega*dt);
    near(final.source[0],0,initial.value[0],"rotation energy source");
    near(final.source[1],(cs-1)*initial.value[1]+sn*initial.value[2],c*initial.value[0],"rotation Fx source");
    near(final.source[2],(cs-1)*initial.value[2]-sn*initial.value[1],c*initial.value[0],"rotation Fy source");
    near(final.source[3],0,c*initial.value[0],"rotation Fz source");
    balance(initial,final,c,"rotating basis");
}

void two_blocks() {
    double const dx=.37, c=7;
    physcon().c=c;
    rad_grid left(dx), right(dx);
    auto profile=[&](auto x) {
        double E=2+.01*x[0]+.02*x[1]-.03*x[2];
        return state{E,.2*c*E,-.1*c*E,.3*c*E};
    };
    initialize(left,dx,profile,-INX*dx/2);
    initialize(right,dx,profile,INX*dx/2);
    auto initial=left.takeConservation(); initial+=right.takeConservation();
    double dt=std::min(left.maxTimestep(0),right.maxTimestep(0))*.71;
    left.compute_flux(0); right.compute_flux(0);
    for(int j=RAD_BW;j<RAD_BW+INX;++j)for(int k=RAD_BW;k<RAD_BW+INX;++k)
        for(int f=0;f<4;++f)
            near(left.flux[0][f][rindex(RAD_BW+INX,j,k)],right.flux[0][f][rindex(RAD_BW,j,k)],
                 c*c*2,"shared interface flux");
    // Only the exterior faces contribute to the global boundary budget.
    left.accountBoundaryFlux(dt,geo::face(0));
    right.accountBoundaryFlux(dt,geo::face(1));
    for(int face=2;face<6;++face) {
        left.accountBoundaryFlux(dt,geo::face(face));
        right.accountBoundaryFlux(dt,geo::face(face));
    }
    left.advance(dt,0); right.advance(dt,0);
    auto final=left.takeConservation(); final+=right.takeConservation();
    near(final.volume,2*volume(dx),2*volume(dx),"two block volume");
    balance(initial,final,c,"internal interface cancels");
}

void writeConservationFixture() {
    double const dx=.37, dt=.013, c=7;
    parameters={}; parameters.c=c; parameters.chi=2.3;
    parameters.length=INX*dx; parameters.width=.31;
    physcon().c=c; opts().problem=2;
    rad_grid g(dx);
    initialize(g,dx,[&](auto){ return state{2,.8*c,-.3*c,.2*c}; });
    radiationConservation::Ledger history;
    radiationConservation::writeCsvHeader(std::cout);
    radiationConservation::writeCsvRow(std::cout,0,history.consume(g.takeConservation()));
    for(int step=1;step<=2;++step) {
        g.applyRegressionSource(dt);
        radiationConservation::writeCsvRow(std::cout,step*dt,history.consume(g.takeConservation()));
    }
}

int main(int argc,char** argv) {
    try {
        require(argc==2,"expected test name");
        std::string name=argv[1];
        if(name=="csv") { writeConservationFixture(); return 0; }
        if(name=="faces") volume_and_faces();
        else if(name=="streaming") streaming(false);
        else if(name=="oblique") streaming(true);
        else if(name=="sources") damping_and_injection();
        else if(name=="sphere") sphere(false);
        else if(name=="cgs") sphere(true);
        else if(name=="rotation") rotation();
        else if(name=="blocks") two_blocks();
        else throw std::runtime_error("unknown test "+name);
        std::cout << name << " passed\n";
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
'''


class ConservationSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "g++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("A C++23 compiler is required for production solver tests")
        cls.folder = tempfile.TemporaryDirectory(prefix="radiation-conservation-")
        cls.addClassCleanup(cls.folder.cleanup)
        cls.executable = Path(cls.folder.name) / "conservation"
        cpp = (ROOT / "src/radiation/rad_grid.cpp").read_text()
        header = "\n".join(
            line for line in (ROOT / "octotiger/radiation/rad_grid.hpp").read_text().splitlines()
            if not line.startswith("#include")
        )
        methods = [
            "void rad_grid::allocate()", "void rad_grid::set_dx(",
            "void rad_grid::set_X(", "Real rad_grid::maxTimestep(",
            "void rad_grid::compute_flux(", "void rad_grid::advance(",
            "void rad_grid::sanity_check()", "void rad_grid::applyRegressionSource(",
            "rad_grid::rad_grid(Real _dx)", "rad_grid::rad_grid()",
            "radiationConservation::Totals rad_grid::takeConservation()",
            "void rad_grid::accountBoundaryFlux(",
        ]
        source = Path(cls.folder.name) / "conservation.cpp"
        source.write_text(FIXTURE + header + "\n" + "\n".join(
            extract_method(cpp, signature) for signature in methods
        ) + CHECKS)
        command = compiler + ["-std=c++23", "-O2", "-I" + str(ROOT), str(source), "-o", str(cls.executable)]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise AssertionError("Production conservation harness failed to compile:\n" + result.stderr)

    def run_case(self, name):
        result = subprocess.run([str(self.executable), name], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_production_csv_roundtrip_and_physical_flux_units(self):
        result = subprocess.run([str(self.executable), "csv"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        folder = Path(self.folder.name) / "csv-roundtrip"
        folder.mkdir()
        (folder / CONSERVATION_FILE).write_text(result.stdout)
        diagnostics = read_conservation(folder, dict(time=.026, length=8*.37, c=7))
        self.assertEqual(len(diagnostics["history"]), 3)
        for record in diagnostics["summary"]:
            self.assertLess(record["max_normalized_error"], 3e-12)
        flux = diagnostics["summary"][1]
        self.assertAlmostEqual(flux["initial"], .8*7*(8*.37)**3)
        self.assertLess(flux["source"], 0)
        self.assertEqual(flux["boundary"], 0)

    def test_interior_totals_and_all_six_face_signs(self):
        self.run_case("faces")

    def test_periodic_streaming_zero_components_and_injected_drift(self):
        self.run_case("streaming")

    def test_periodic_oblique_streaming_and_injected_drift(self):
        self.run_case("oblique")

    def test_damping_and_gaussian_source_units(self):
        self.run_case("sources")

    def test_open_sphere_boundary_and_source_balance(self):
        self.run_case("sphere")

    def test_open_sphere_in_cgs_units(self):
        self.run_case("cgs")

    def test_rotating_basis_source_balance(self):
        self.run_case("rotation")

    def test_two_block_internal_interface_cancellation(self):
        self.run_case("blocks")


if __name__ == "__main__":
    unittest.main()
