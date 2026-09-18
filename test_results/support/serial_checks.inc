
#undef private
int main(int argc,char** argv) {
 try {
    auto const ref=radiationTests::ReferenceData::read(argv[1]);
    if(ref.n!=INX) throw std::runtime_error("fixture/reference size mismatch");
    auto const& p=ref.p; physcon().c=p.c;
    opts().radReference=argv[1];opts().stop_time=p.time;opts().xscale=p.length/2;
    opts().radTestChi=p.chi;opts().radTestWidth=p.width;opts().radTestBackground=p.background;
    opts().radTestAmplitude=p.amplitude;opts().radTestLuminosity=p.luminosity;
    double const dx=p.length/INX;
    std::vector<std::vector<Real>> X(3,std::vector<Real>(H_N3));
    for(int i=0;i<H_NX;++i)for(int j=0;j<H_NX;++j)for(int k=0;k<H_NX;++k){
        auto r=hindex(i,j,k);X[0][r]=-p.length/2+(i-H_BW+.5)*dx;
        X[1][r]=-p.length/2+(j-H_BW+.5)*dx;X[2][r]=-p.length/2+(k-H_BW+.5)*dx;
    }
    const char* names[]{"STREAMING_WAVE","STREAMING_FRONT","GAUSSIAN_PULSE","EQUILIBRIUM_SPHERE"};
    for(int test=0;test<4;++test){
        opts().problem=test;opts().periodic=test!=3;opts().rad_implicit=test>=2;
        validateRadiationTest();rad_grid g(dx);g.set_X(X);
        auto analytic=[&](radiationTests::Point x,double t){
            auto const full=t==0?radiationRegressionInit(x[0],x[1],x[2],dx):
                radiationRegressionAnalytic(x[0],x[1],x[2],t);
            radiationTests::State u;
            for(int f=0;f<4;++f)u[f]=full[opts().n_fields+f];
            return u;
        };
        for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k){
            auto r=rindex(i,j,k);auto u=analytic({X[0][r],X[1][r],X[2][r]},0);
            for(int f=0;f<4;++f)g.U[f][r]=u[f];
        }
        std::filesystem::path out=std::filesystem::path(argv[2])/names[test];
        std::filesystem::create_directories(out/"radiation-slices");
        std::ofstream conservation(out/"radiation-conservation.csv");
        conservation.exceptions(std::ios::badbit|std::ios::failbit);
        radiationConservation::Ledger budget;
        radiationConservation::writeCsvHeader(conservation);
        radiationConservation::writeCsvRow(conservation,0,budget.consume(g.takeConservation()));
        double t=0;
        while(t<p.time){
            if(test==3){for(int face=0;face<6;++face)g.set_physical_boundaries(face,t);}
            else {
                auto wrap=[](int x){return RAD_BW+(x-RAD_BW+INX)%INX;};
                for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
                    for(int f=0;f<4;++f)g.U[f][rindex(i,j,k)]=g.U[f][rindex(wrap(i),wrap(j),wrap(k))];
            }
            double const dt=std::min(g.maxTimestep(0),p.time-t);
            g.compute_flux(0);
            if(test==3)for(int face=0;face<6;++face)g.accountBoundaryFlux(dt,face);
            g.advance(dt,0);
            if(test>=2)g.applyRegressionSource(dt);
            t+=dt;
            radiationConservation::writeCsvRow(conservation,t,budget.consume(g.takeConservation()));
        }
        conservation.close();
        radiationTests::SliceOutput slice(true,out.string(),t,dx,p.length);
        std::array<double,4> l1{},l2{},linf{};
        for(int i=RAD_BW;i<RAD_BW+INX;++i)for(int j=RAD_BW;j<RAD_BW+INX;++j)for(int k=RAD_BW;k<RAD_BW+INX;++k){
            auto r=rindex(i,j,k);auto u=analytic({X[0][r],X[1][r],X[2][r]},t);
            slice.capture(X[0][r],X[1][r],X[2][r],u,0,[&](int f){return g.U[f][r];});
            for(int f=0;f<4;++f){double e=std::abs(u[f]-g.U[f][r]);l1[f]+=e;l2[f]+=e*e;linf[f]=std::max(linf[f],e);}
        }
        slice.finish();
        std::ofstream logfile(out/"run.log");
        auto old_buffer=std::cout.rdbuf(logfile.rdbuf());
        std::cout<<"RADIATION_TEST_FINISHED RADIATION_"<<names[test]<<" t="<<std::setprecision(17)<<t<<'\n';
        const char* fields[]{"er","fx","fy","fz"};
        for(int f=0;f<4;++f)std::cout<<fields[f]<<" "<<std::scientific<<l1[f]/(INX*INX*INX)<<" "<<std::sqrt(l2[f]/(INX*INX*INX))<<" "<<linf[f]<<'\n';
        std::cout.rdbuf(old_buffer);
        const char* norms[]{"L1","L2","Linf"};
        int level=std::lround(std::log2(double(INX)/8));
        for(int norm=0;norm<3;++norm){
            std::ofstream file(out/(std::string(norms[norm])+".dat"));
            file<<std::scientific<<std::setprecision(6)<<dx<<" "<<level;
            for(int f=0;f<4;++f) file<<" "<<(norm==0?l1[f]/(INX*INX*INX):
                norm==1?std::sqrt(l2[f]/(INX*INX*INX)):linf[f]);
            file<<'\n';
        }
    }
 } catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
