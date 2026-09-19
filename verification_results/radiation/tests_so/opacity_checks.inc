// Tests the production rad_grid methods and opacity selectors, not a second solver.
void opacity_source_checks() {
    // Unit conversion, including constant extinction (Ensman-style cm^-1 inputs).
    radiation::GreyOpacity o;o.model="grey";o.absorption=2;o.scattering=3;o.transport_absorption=.5;
    o.validate(-1);
    auto k=radiation::greyCoefficients(o,4,5,2);
    near(k.absorption,10,10,"cgs mass absorption conversion");
    near(k.scattering,15,15,"cgs mass scattering conversion");
    near(k.transport,17.5,17.5,"independent Rosseland mean plus scattering");
    o.units="1/cm";k=radiation::greyCoefficients(o,4,5,2);
    near(k.absorption,4,4,"extinction absorption conversion");
    near(k.transport,7,7,"extinction transport conversion");
    near(radiation::greyCoefficients(o,400,5,2).transport,7,7,"constant extinction density independence");

    // Compare explicit S&O and equal-opacity grey to the historical constant mode,
    // using the whole production driver (thermal exchange and momentum feedback).
    auto solve=[](std::string model) {
        reset();opts().rad_implicit=true;opts().radiationOpacity.model=model;
        if(model=="grey") {opts().rad_opacity=-1;opts().radiationOpacity.absorption=.7;}
        opts().radiationOpacity.validate(opts().rad_opacity);
        node_server node;initialize(*node.rad_grid_ptr,*node.grid_ptr,1,0,.7,.002);
        node.dt_.radiation_dt=node.rad_grid_ptr->maxTimestep(0);
        node.dt_.dt=.7*node.dt_.radiation_dt;
        node.compute_radiation(node.dt_.dt,0);
        auto const r=rindex(3,3,3);
        std::vector<Real> values;
        for(auto const& f:node.rad_grid_ptr->U)values.push_back(f[r]);
        for(auto const& f:node.grid_ptr->U)values.push_back(f[r]);
        return values;
    };
    auto const legacy=solve("legacy");
    require(legacy==solve("skinner_ostriker"),"explicit S&O bitwise source compatibility");
    require(legacy==solve("grey"),"equal-opacity grey bitwise source compatibility");

    for(Real ratio:{1.,.03})for(Real sigma:{0.,1e-9,2.,1e6})for(Real v:{0.,.002}) {
        reset();opts().rad_implicit=true;opts().rad_c_ratio=ratio;opts().rad_opacity=-1;
        opts().radiationOpacity.model="grey";opts().radiationOpacity.scattering=sigma;
        // Static stiff scattering is implicit; moving stiff beta*tau is explicitly
        // outside this solver's regime. Test moving gas only in its resolved regime.
        if(v!=0 && sigma>1e3)continue;
        rad_grid r(1./INX);grid g;initialize(r,g,1,.1,10,v);r.compute_mmw(g.U);
        auto const h=rindex(3,3,3);Real const c=physcon().c;
        Real const E0=r.U[0][h],gas0=g.U[egas_i][h],p0=g.U[sx_i][h],F0=r.U[1][h];
        auto const beforeGas=g.U[egas_i];auto const beforeE=r.U[0];
        Real const dt=.2*r.maxTimestep(0);
        r.rad_imp(g.U[egas_i],g.U[tau_i],g.U[sx_i],g.U[sy_i],g.U[sz_i],g.U[rho_i],dt);
        require(beforeGas==g.U[egas_i] && beforeE==r.U[0],"scattering has exactly zero thermal exchange");
        r.prepareSources(g.U);r.compute_flux(dt,0);r.advance(dt,0);r.finishSources(g.U);
        near(g.U[egas_i][h]+r.U[0][h]/ratio,gas0+E0/ratio,gas0+E0/ratio,"scattering total energy invariant");
        near(g.U[sx_i][h]+r.U[1][h]/(c*c*ratio),p0+F0/(c*c*ratio),1,"scattering momentum invariant");
        if(v==0) {
            near(r.U[0][h],E0,E0,"static scattering radiation energy");
            near(g.U[egas_i][h],gas0,gas0,"static scattering gas total energy");
            near(r.U[1][h],F0/(1+dt*c*ratio*sigma),F0,"thin/thick scattering flux decay");
        } else if(sigma>1e-6) {
            require(r.U[0][h]<E0,"moving scattering performs mechanical work");
            require(g.U[egas_i][h]>gas0,"scattering work goes to gas");
        }
        require(r.U[0][h]>=0 && g.U[tau_i][h]>=0,"source positivity");
    }
    for(Real absorption:{0.,1e-9,2.,1e6})for(Real ratio:{1.,.03}) {
        reset();opts().rad_implicit=true;opts().rad_opacity=-1;opts().rad_c_ratio=ratio;
        opts().radiationOpacity.model="grey";opts().radiationOpacity.units="1/cm";
        opts().radiationOpacity.absorption=absorption;
        rad_grid r(1./INX);grid g;initialize(r,g,.4,0,.7,0);r.compute_mmw(g.U);
        r.rad_imp(g.U[egas_i],g.U[tau_i],g.U[sx_i],g.U[sy_i],g.U[sz_i],g.U[rho_i],.3);
        auto const h=rindex(3,3,3);
        auto const expected=M1::thermalExchange(.7,.4,.3*physcon().c*absorption,0,false,ratio,1);
        near(g.U[egas_i][h],expected.gas,1,"production pure absorption gas root");
        near(r.U[0][h],expected.radiation,1,"production pure absorption radiation root");
        near(g.U[egas_i][h]+r.U[0][h]/ratio,.7+.4/ratio,.7+.4/ratio,"pure absorption conservation");
        require(g.U[egas_i][h]>=0 && r.U[0][h]>=0,"pure absorption positivity");
    }
    // Absorption and scattering have independent roles in production caches.
    reset();opts().rad_implicit=true;opts().rad_opacity=-1;
    opts().radiationOpacity.model="grey";opts().radiationOpacity.absorption=2;
    opts().radiationOpacity.transport_absorption=.5;opts().radiationOpacity.scattering=3;
    rad_grid r(1./INX);grid g;initialize(r,g,1,0,.7,0);r.prepareSources(g.U);
    near(r.chiAbsorption[rindex(3,3,3)],2,2,"Planck/energy cache excludes scattering");
    near(r.chiTotal[rindex(3,3,3)],3.5,3.5,"Rosseland/transport cache includes scattering");
    Real const expectedSignal=std::sqrt((4./9)*(-std::expm1(-3.5*g.dx)));
    near(r.hydroSignalSpeed(g.U[egas_i],g.U[tau_i],g.U[sx_i],g.U[sy_i],g.U[sz_i],g.U[rho_i]),
         expectedSignal,1,"existing acoustic formula selects total transport opacity");

    // Semi-discrete isotropic flux equation: dQ/dt=-c_hat grad(E)/3-c_hat chi_T Q.
    // Its steady physical F must be -c grad(E)/(3 chi_T), independent of RSLA.
    // This checks source/pressure balance, not unresolved-cell AP accuracy.
    for(Real ratio:{1.,.03})for(Real chi:{10.,100.,1e4}) {
        Real const c=7,chat=c*ratio,grad=.02,dt=20/(chat*chi);
        Real Q=0;
        Real const pressure=M1::ConservedState{1,0,0,0}.physicalFlux(0,chat).flux[1];
        for(int n=0;n<15;++n)Q=M1::damp(Q,-dt*pressure*grad,dt*chat*chi,1);
        near(c*Q,-c*grad/(3*chi),c*grad/(3*chi),"diffusion flux inverse-opacity scaling");
    }
    std::cout<<"PASS grey opacity units, equal-mode regression, zero/thin/thick sources, scattering conservation and diffusion balance\n";
}
