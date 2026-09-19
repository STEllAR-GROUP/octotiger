#include <filesystem>
#include <fstream>
// Verification-only configuration. Numerical transport/source routines above are
// extracted verbatim from production. No hydro evolution is substituted.
int main(int argc,char** argv) {
 try {
    if(argc!=3)throw std::runtime_error("suite EXPLICIT_CONFIG OUTPUT_DIRECTORY");
    std::ifstream config(argv[1]);config.exceptions(std::ios::badbit|std::ios::failbit);
    std::string test;double L,end,chi,A,width,velocity,ratio,rho,e,E;int frames;
    config>>test>>L>>end>>chi>>A>>width>>velocity>>ratio>>rho>>e>>E>>frames;
    reset();physcon().c=2.99792458e10;physcon().sigma=5.670374419e-5;
    physcon().mh=1.6735575e-24;physcon().kb=1.380649e-16;
    opts().rad_c_ratio=ratio;opts().xscale=L/2;
    bool const diffusion=test=="static_diffusion" || test=="diffusion_thicker";
    bool const thermal=test=="thermal_relaxation";
    bool const moving=test=="moving_scattering";
    bool const coupled=diffusion||moving||test=="damped_wave";
    bool const counters=test=="boundary_subcycles";
    opts().rad_implicit=coupled||thermal||counters;
    opts().radiationOpacity.model="grey";opts().radiationOpacity.units="1/cm";
    opts().rad_opacity=-1;
    opts().radiationOpacity.absorption=(thermal||test=="damped_wave")?chi:0;
    opts().radiationOpacity.scattering=(diffusion||moving||counters)?chi:0;
    opts().rad_energy_mode=test=="damped_wave"?"absorption":thermal?"thermal":"equilibrium";
    opts().rad_velocity_terms=moving||counters;
    node_server node;auto& r=*node.rad_grid_ptr;auto& g=*node.grid_ptr;
    double const dx=L/INX,c=physcon().c,chat=c*ratio;
    g.dx=dx;initialize(r,g,E,0,e,0);
    for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k) {
        int h=rindex(i,j,k);double x=-L/2+(i-RAD_BW+.5)*dx;
        g.X[0][h]=x;g.X[1][h]=-L/2+(j-RAD_BW+.5)*dx;g.X[2][h]=-L/2+(k-RAD_BW+.5)*dx;
        g.U[rho_i][h]=g.U[spc_i][h]=rho;g.U[egas_i][h]=e+.5*rho*velocity*velocity;
        g.U[sx_i][h]=rho*velocity;g.U[tau_i][h]=std::pow(e,1/grid::get_fgamma());
        double u=E,q=0;
        if(test=="streaming_front_1d") {
            radiationTests::Parameters p;p.length=L;p.c=c;
            u=radiationTests::streamingFront({x,0,0},0,dx,p)[0];q=u;
        } else if(test=="thin_gaussian") {
            for(int image=-3;image<=3;++image)
                u+=A*.5*std::sqrt(std::numbers::pi)*width/dx*
                    (std::erf((x+dx/2+image*L)/width)-std::erf((x-dx/2+image*L)/width));
            q=u;
        } else if(!thermal && !moving && !counters) {
            u+=A*std::cos(2*std::numbers::pi*x/L)*radiationTests::sinc(std::numbers::pi*dx/L);
            q=diffusion?0:u;
        } else if(moving||counters) q=.1*E;
        r.U[0][h]=u;r.U[1][h]=c*q;r.U[2][h]=r.U[3][h]=0;
    }
    r.set_X(g.X);r.compute_mmw(g.U);node.current_time=0;
    std::filesystem::path out=argv[2];std::filesystem::create_directories(out);
    std::ofstream slices(out/"samples.csv"),history(out/"history.csv"),exchange(out/"exchanges.csv");
    for(auto f:{&slices,&history,&exchange}) {f->exceptions(std::ios::badbit|std::ios::failbit);*f<<std::setprecision(17);}
    slices<<"frame,t,x,E,Fx,gas_energy,momentum\n";
    history<<"t,mean_E,mean_gas,mean_momentum,mean_Fx,min_E,max_reduced_flux,gas_dt,rad_dt,subcycles,rcycle,hcycle,hydro_ghost_unchanged\n";
    exchange<<"gas_step,event,rcycle,hcycle,time,interior_E,interior_Fx,halo_valid,hydro_unchanged\n";
    auto sample=[&](int frame,double t,double gasdt,double raddt,int sub,bool ghostOK) {
        long double sumE=0,sumG=0,sumP=0,sumF=0;double minE=1e300,maxf=0;
        interiors([&](int h){sumE+=r.U[0][h];sumF+=r.U[1][h];sumG+=g.U[egas_i][h];sumP+=g.U[sx_i][h];
            minE=std::min(minE,r.U[0][h]);
            maxf=std::max(maxf,std::hypot(r.U[1][h],r.U[2][h],r.U[3][h])/(c*std::max(r.U[0][h],1e-300)));});
        long double n=static_cast<long double>(INX)*INX*INX;
        history<<t<<','<<sumE/n<<','<<sumG/n<<','<<sumP/n<<','<<sumF/n<<','<<minE<<','<<maxf<<','<<gasdt<<','<<raddt<<','<<sub<<','<<node.rcycle<<','<<node.hcycle<<','<<ghostOK<<'\n';
        if(frame>=0)for(int i=0;i<INX;++i){int h=rindex(i+RAD_BW,RAD_BW,RAD_BW);
            slices<<frame<<','<<t<<','<<g.X[0][h]<<','<<r.U[0][h]<<','<<r.U[1][h]<<','<<g.U[egas_i][h]<<','<<g.U[sx_i][h]<<'\n';}
    };
    double t=0;int step=0;sample(0,0,0,0,0,true);
    for(int frame=1;frame<frames;++frame) {
        double target=end*frame/(frames-1);
        while(t<target) {
            double limit=r.maxTimestep(0);
            double dt=std::min(target-t,thermal?end/(8*INX):counters?4*limit:limit);
            if(!(dt>0))throw std::runtime_error("nonpositive suite timestep");
            auto gasBefore=g.U;
            if(thermal)r.rad_imp(g.U[egas_i],g.U[tau_i],g.U[sx_i],g.U[sy_i],g.U[sz_i],g.U[rho_i],dt);
            else if(counters) {
                node.dt_.radiation_dt=limit;node.dt_.dt=dt;node.current_time=t;
                node.compute_radiation(dt,0);
                for(auto const& event:node.audit)
                    exchange<<step<<','<<event.kind<<','<<event.r<<','<<event.h<<','<<event.time<<','<<event.E<<','<<event.F<<','<<event.halo<<','<<event.gas<<'\n';
                node.audit.clear();
            } else {
                periodic(r);
                if(coupled)r.prepareSources(g.U);
                r.compute_flux(dt,0,t);r.advance(dt,0);
                if(coupled)r.finishSources(g.U);
            }
            bool ghostOK=true;
            for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
                if(i<RAD_BW||i>=RAD_BW+INX||j<RAD_BW||j>=RAD_BW+INX||k<RAD_BW||k>=RAD_BW+INX)
                    for(int f=0;f<7;++f)ghostOK=ghostOK && gasBefore[f][rindex(i,j,k)]==g.U[f][rindex(i,j,k)];
            if(!ghostOK)throw std::runtime_error("radiation overwrote gas ghost data");
            t=std::min(target,t+dt);++step;
            int sub=counters?int(radiation::SubcyclePlan(dt,limit,true,1024).count):thermal?0:1;
            sample(t==target?frame:-1,t,dt,thermal?0:dt/std::max(sub,1),sub,ghostOK);
        }
    }
    std::cout<<"completed "<<test<<" N="<<INX<<" t="<<t<<" steps="<<step<<" c="<<c<<"\n";
 }catch(std::exception const& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
