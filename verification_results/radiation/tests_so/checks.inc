void require(bool yes,std::string const& label) { if(!yes)throw std::runtime_error(label); }
void near(long double a,long double b,long double scale,std::string const& label,long double tol=3e-12L) {
    if(!std::isfinite(a) || std::abs(a-b)>tol*std::max(std::abs(scale),1e-30L)) {
        std::cerr<<std::setprecision(18)<<label<<": "<<a<<" vs "<<b<<" scale="<<scale<<'\n';
        throw std::runtime_error(label);
    }
}
void reset() {opts()=options_fixture{};physcon()=constants_fixture{};}
template<class Fn>void interiors(Fn fn) {
    for(int i=RAD_BW;i<RAD_BW+INX;++i)for(int j=RAD_BW;j<RAD_BW+INX;++j)
      for(int k=RAD_BW;k<RAD_BW+INX;++k) fn(rindex(i,j,k));
}
void initialize(rad_grid& r,grid& g,Real E=1,Real q=.1,Real e=100,Real v=.02) {
    for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k) {
        auto n=rindex(i,j,k);
        g.X[0][n]=(i-RAD_BW+.5)*g.dx;g.X[1][n]=(j-RAD_BW+.5)*g.dx;g.X[2][n]=(k-RAD_BW+.5)*g.dx;
        r.U[0][n]=E;r.U[1][n]=physcon().c*q;r.U[2][n]=r.U[3][n]=0;
        g.U[rho_i][n]=g.U[spc_i][n]=1;g.U[egas_i][n]=e+.5*v*v;
        g.U[tau_i][n]=std::pow(e,1/grid::get_fgamma());g.U[sx_i][n]=v;
    }
    r.set_dx(g.dx);r.set_X(g.X);
}
void kernel_checks() {
    reset();
    for(Real chat:{.07,7.}) {
        auto iso=M1::ConservedState{1,0,0,0}.physicalFlux(0,chat);
        near(iso.plus,chat/std::sqrt(3.),chat,"isotropic wave");
        near(iso.minus,-chat/std::sqrt(3.),chat,"isotropic wave");
        near(iso.flux[1],chat/3,chat,"isotropic pressure");
        auto beam=M1::ConservedState{2,1.2,1.6,0}.physicalFlux(0,chat);
        near(beam.plus,.6*chat,chat,"oblique beam");near(beam.minus,.6*chat,chat,"oblique beam");
        auto transverse=M1::ConservedState{2,1.2,1.6,0}.physicalFlux(2,chat);
        near(transverse.plus,0,chat,"transverse beam");
        auto vacuum=M1::ConservedState{}.physicalFlux(0,chat);
        for(auto x:static_cast<std::array<Real,4>>(vacuum.flux))near(x,0,1,"vacuum flux");
    }
    // General closure trace and independent Eq. (41a), including near-streaming states.
    for(int n=1;n<100;++n) {
        Real f=Real(n)/100,mu=.37,E=2;
        M1::ConservedState u{E,E*f*mu,E*f*std::sqrt(1-mu*mu),0};
        Real trace=0;
        for(int d=0;d<3;++d)trace+=u.physicalFlux(d,1).flux[d+1];
        near(trace,E,E,"pressure trace");
        Real s=std::sqrt(4-3*f*f);
        Real root=std::sqrt((2./3)*(4-3*f*f-s)+2*mu*mu*(2-f*f-s));
        near(u.physicalFlux(0,1).plus,(mu*f+root)/s,1,"S&O eigenvalue",1e-10);
    }
    auto pair=M1::reconstruct({1,1,0,0},{1,0,1,0},{1,-1,0,0});
    pair.first.checkState();pair.second.checkState();
    for(int f=0;f<4;++f)near(.5*(pair.first[f]+pair.second[f]),M1::ConservedState{1,0,1,0}[f],1,"PLM average");
    near(M1::plmSlope(1,2),4./3,1,"Athena PLM slope");
    near(M1::plmSlope(1,-1),0,1,"PLM extremum");
    std::cout<<"PASS closure, Eq. 41 wave speeds, vacuum and PLM realizability\n";
}
void thermal_checks() {
    for(Real ratio:{1.,.1,.001})for(Real eta:{1e-9,.03,2.,1e6})for(Real theta:{.5,.51,1.}) {
        auto e=M1::thermalExchange(.7,.4,eta,std::log(.3),false,ratio,theta);
        near(e.gas+e.radiation/ratio,.7+.4/ratio,.7+.4/ratio,"RSLA thermal conservation");
        require(e.gas>=0 && e.radiation>=0,"thermal positivity");
    }
    auto be=M1::thermalExchange(1,0,1e6,0,false,.01,1);
    auto fallback=M1::thermalExchange(1,0,1e6,0,false,.01,.51);
    near(fallback.gas,be.gas,1,"stiff thermal fallback");
    auto solution=[](int n) {
        Real e=1,E=.2,ratio=.1,t=.4;
        for(int i=0;i<n;++i) {
            auto out=M1::thermalExchange(e,E,t/n,0,true,ratio,.5);e=out.gas;E=out.radiation;
        }
        Real const equilibrium=(.1*1+.2)/1.1;
        Real const exact=equilibrium+(1-equilibrium)*std::exp(-1.1*t);
        return std::abs(e-exact);
    };
    Real const reduction=solution(4)/solution(8);
    require(reduction>3.9 && reduction<4.1,"theta=.5 thermal order");
    near(M1::damp(1,0,1e8,.51),1/(1+1e8),1,"stiff flux fallback");
    std::cout<<"PASS thermal quartic, RSLA invariant, theta convergence and stiff fallback\n";
}
void driver_checks() {
    reset();opts().rad_log_subcycles=true;
    timestep_t gas,rad,combined;
    gas.dt=.004;gas.radiation_dt=.003;gas.x=1;
    rad.dt=.008;rad.radiation_dt=.001;rad.x=2;
    combined.reduce(gas);combined.reduce(rad);
    near(combined.dt,.004,1,"gas reduction");near(combined.radiation_dt,.001,1,"radiation reduction");
    near(combined.x,1,1,"winning diagnostic");
    radiation::SubcyclePlan nearInteger(.1*7,.1,true,7);
    require(nearInteger.count==7,"roundoff at subcycle budget");
    bool rejected=false;
    try {radiation::SubcyclePlan invalid(1,.1,false,100);}catch(std::runtime_error const&){rejected=true;}
    require(rejected,"disabled subcycling must enforce CFL");
    node_server leaf,parent;
    initialize(*leaf.rad_grid_ptr,*leaf.grid_ptr);
    initialize(*parent.rad_grid_ptr,*parent.grid_ptr);
    parent.is_refined=true;
    leaf.dt_=parent.dt_=combined;
    leaf.compute_radiation(combined.dt,0);parent.compute_radiation(combined.dt,0);
    require(leaf.boundaries==parent.boundaries && leaf.fluxKeys==parent.fluxKeys,"all-level substep sequence");
    require(leaf.boundaries.size()==5 && leaf.fluxKeys.size()==4,"one halo per substep plus final restriction");
    require(leaf.hcycle==13 && leaf.rcycle==52,"radiation independent of hydro counter");
    near(leaf.boundaries.back().second,leaf.current_time+combined.dt,1,"exact final boundary time");
    // Migration/regridding inherits rcycle, never starts flux epochs from zero.
    node_server migrated;initialize(*migrated.rad_grid_ptr,*migrated.grid_ptr);
    migrated.rcycle=leaf.rcycle;migrated.dt_=combined;
    migrated.compute_radiation(combined.dt,0);
    require(migrated.fluxKeys.front()>leaf.fluxKeys.back(),"epochs after migration");
    std::cout<<"PASS production subcycle driver, independent minima and boundary epochs\n";
}
void source_checks() {
    for(Real ratio:{1.,.03}) {
        reset();opts().rad_c_ratio=ratio;opts().rad_implicit=true;
        opts().rad_energy_mode="equilibrium";opts().rad_theta=.51;
        node_server node;initialize(*node.rad_grid_ptr,*node.grid_ptr);
        auto const h=hindex(RAD_BW,RAD_BW,RAD_BW),r=rindex(RAD_BW,RAD_BW,RAD_BW);
        Real const E0=node.grid_ptr->U[egas_i][h]+node.rad_grid_ptr->U[0][r]/ratio;
        Real const m0=node.grid_ptr->U[sx_i][h]+node.rad_grid_ptr->U[1][r]/(physcon().c*physcon().c*ratio);
        node.dt_.radiation_dt=node.rad_grid_ptr->maxTimestep(0);
        node.dt_.dt=3.2*node.dt_.radiation_dt;
        node.compute_radiation(node.dt_.dt,0);
        near(node.grid_ptr->U[egas_i][h]+node.rad_grid_ptr->U[0][r]/ratio,E0,E0,"coupled energy");
        near(node.grid_ptr->U[sx_i][h]+node.rad_grid_ptr->U[1][r]/(physcon().c*physcon().c*ratio),m0,1,"coupled momentum");
        require(node.hcycle==14,"one gas cache exchange, independent of subcycle count");
    }
    reset();opts().rad_implicit=true;opts().rad_c_ratio=.1;
    rad_grid r(1./INX);grid g;initialize(r,g,.4,0,.7,0);
    Real const e0=g.U[egas_i][rindex(3,3,3)]+r.U[0][rindex(3,3,3)]/.1;
    r.compute_mmw(g.U);r.rad_imp(g.U[egas_i],g.U[tau_i],g.U[sx_i],g.U[sy_i],g.U[sz_i],g.U[rho_i],.3);
    near(g.U[egas_i][rindex(3,3,3)]+r.U[0][rindex(3,3,3)]/.1,e0,e0,"production thermal invariant");
    // Radiation-only absorption reduces E and Q at the same rate, preserving a beam.
    opts().rad_energy_mode="absorption";opts().rad_theta=1;opts().rad_velocity_terms=false;
    initialize(r,g,1,1,1e5,0);r.prepareSources(g.U);
    Real const dt=r.maxTimestep(0);r.compute_flux(dt,0);r.advance(dt,0);
    near(r.U[0][rindex(3,3,3)],1/(1+dt*.7*.7),1,"absorption rate with c_hat");
    near(r.U[1][rindex(3,3,3)]/physcon().c,r.U[0][rindex(3,3,3)],1,"absorbed beam realizability");
    std::cout<<"PASS production gas feedback, thermal step and absorption-only source\n";
}

// A set of physical 3D blocks holding a wave along x. Fill just ONE t^n halo
// from the common old solution, then run every block's real transport methods.
std::vector<Real> wave(int blocks,Real ratio,bool discontinuity=false) {
    reset();opts().rad_c_ratio=ratio;
    Real const c=physcon().c,chat=c*ratio,dx=1./(blocks*INX),end=.15/chat;
    std::vector<std::unique_ptr<rad_grid>> grids;
    for(int b=0;b<blocks;++b) {
        grids.emplace_back(std::make_unique<rad_grid>(dx));
        grid g;g.dx=dx;initialize(*grids.back(),g);
        interiors([&](int r){
            int const i=r/(RAD_NX*RAD_NX)-RAD_BW;
            Real const x=(b*INX+i+.5)*dx;
            Real E=discontinuity ? (x<.5 ? 1 : 0) : 1+.2*std::cos(2*std::numbers::pi*x)*std::sin(std::numbers::pi*dx)/(std::numbers::pi*dx);
            grids.back()->U[0][r]=E;grids.back()->U[1][r]=c*E;
        });
    }
    Real t=0;
    while(t<end) {
        Real const dt=std::min(grids.front()->maxTimestep(0),end-t);
        for(int b=0;b<blocks;++b)for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k) {
            int const global=(b*INX+i-RAD_BW+blocks*INX)%(blocks*INX);
            int const other=global/INX, oi=global%INX+RAD_BW;
            int const oj=(j-RAD_BW+INX)%INX+RAD_BW,ok=(k-RAD_BW+INX)%INX+RAD_BW;
            for(int f=0;f<4;++f)grids[b]->U[f][rindex(i,j,k)]=grids[other]->U[f][rindex(oi,oj,ok)];
        }
        for(auto& g:grids)g->compute_flux(dt,0);
        for(auto& g:grids)g->advance(dt,0);
        t+=dt;
    }
    std::vector<Real> out;
    for(auto& g:grids)for(int i=RAD_BW;i<RAD_BW+INX;++i)out.push_back(g->U[0][rindex(i,RAD_BW,RAD_BW)]);
    return out;
}
void transport_checks() {
    std::array<Real,3> errors{};
    for(int level=0;level<3;++level) {
        int const blocks=1<<level,N=INX*blocks;
        auto u=wave(blocks,1);
        Real sum=0;
        for(int i=0;i<N;++i) {
            Real exact=1+.2*std::cos(2*std::numbers::pi*((i+.5)/N-.15))*std::sin(std::numbers::pi/N)/(std::numbers::pi/N);
            errors[level]+=std::abs(u[i]-exact)/N;sum+=u[i];
        }
        near(sum/N,1,1,"periodic transport conservation");
    }
    std::cout<<std::scientific<<std::setprecision(6)<<"VL L1 errors "<<errors[0]<<" "<<errors[1]<<" "<<errors[2]
             <<"; orders "<<std::log2(errors[0]/errors[1])<<" "<<std::log2(errors[1]/errors[2])<<'\n';
    require(errors[1]/errors[2]>3,"VL simultaneous space/time convergence");
    auto full=wave(2,1),reduced=wave(2,.01);
    for(std::size_t i=0;i<full.size();++i)near(full[i],reduced[i],1,"RSLA transport speed");
    std::cout<<"RSLA speed comparison passed; checking front\n";
    auto front=wave(2,.01,true);
    for(auto e:front)require(e>=0 && e<=1+1e-12,"front positivity/maximum principle");
    std::cout<<"PASS 3D production transport: smooth convergence, block halos, RSLA, front/vacuum\n";
}
void boundary_and_reflux_checks() {
    reset();opts().problem=MARSHAK;opts().periodic=false;
    rad_grid r(1./INX);grid g;initialize(r,g);
    std::array<bool,6> physical{};physical[0]=true;
    Real const time=.3,dt=r.maxTimestep(0);
    r.set_physical_boundaries(geo::face(0),time);
    r.compute_flux(dt,0,time,physical);
    near(r.Uhalf[0][rindex(2,3,3)],1+time+.5*dt,1,"half-time inflow energy");
    near(r.Uhalf[1][rindex(2,3,3)],.2+.1*(time+.5*dt),1,"half-time physical F normalization");
    // Fine-face area averaging and overwrite of the coarse conservative flux.
    rad_grid fine(.5/INX),coarse(1./INX);
    for(int f=0;f<4;++f)for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
        fine.flux[0][f][rindex(i,j,k)]=f+2*j+3*k;
    std::array<integer,3> lo{RAD_BW,RAD_BW,RAD_BW},hi{RAD_BW+1,RAD_BW+INX,RAD_BW+INX};
    auto values=fine.get_flux_restrict(lo,hi,geo::dimension(0));
    std::array<integer,3> chi{RAD_BW+1,RAD_BW+INX/2,RAD_BW+INX/2};
    coarse.set_flux_restrict(values,lo,chi,geo::dimension(0));
    for(int f=0;f<4;++f)near(coarse.flux[0][f][rindex(3,3,3)],f+2*3.5+3*3.5,30,"AMR face area average");
    std::cout<<"PASS physical half-time inflow and production AMR face restriction\n";
}
int main() {
    try {
        kernel_checks();thermal_checks();driver_checks();source_checks();transport_checks();boundary_and_reflux_checks();
        std::cout<<"All S&O serial production-method checks passed. HPX/MPI runtime is not exercised.\n";
    }catch(std::exception const& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
