void periodic(rad_grid& g) {
    auto wrap=[](int n) { return RAD_BW+(n-RAD_BW+INX)%INX; };
    for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
        for(int f=0;f<4;++f)g.U[f][rindex(i,j,k)]=g.U[f][rindex(wrap(i),wrap(j),wrap(k))];
}
// Execute the production driver with serial stand-ins only for HPX actions and
// node geometry. The actual predictor, corrector and source methods run below.
#define GET(x) (x).get()
struct future_fixture { void get() {} };
struct location_fixture {
    int depth=0;
    int level() const { return depth; }
    bool is_physical_boundary(geo::face) const { return false; }
};
class node_server {
public:
    std::shared_ptr<grid> grid_ptr=std::make_shared<grid>();
    std::shared_ptr<rad_grid> rad_grid_ptr=std::make_shared<rad_grid>(1./INX);
    timestep_t dt_;
    Real current_time=2;
    bool is_refined=false;
    location_fixture my_location;
    std::size_t rcycle=47,hcycle=13,number_hydro_exchange_promises=100;
    std::array<future_fixture,100> all_neighbors_got_hydro{};
    std::vector<std::pair<std::size_t,Real>> boundaries;
    std::vector<std::size_t> fluxKeys;
    struct event { const char* kind;std::size_t r,h;Real time,E,F;bool halo,gas; };
    std::vector<event> audit;
    void all_hydro_bounds() { ++hcycle; audit.push_back({"hydro",rcycle,hcycle,current_time,0,0,true,true}); }
    void all_rad_bounds(Real time);
    future_fixture exchange_interlevel_rad_data() { return {}; }
    void send_rad_amr_bounds() {}
    void collect_radiation_bounds(Real time) {
        auto before=grid_ptr->U;
        auto& r=*rad_grid_ptr;
        boundaries.emplace_back(rcycle,time);
        Real E=r.U[0][rindex(RAD_BW,RAD_BW,RAD_BW)],F=r.U[1][rindex(RAD_BW,RAD_BW,RAD_BW)];
        periodic(r);
        bool halo=true;
        auto wrap=[](int n){return RAD_BW+(n-RAD_BW+INX)%INX;};
        for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
            for(int f=0;f<4;++f)halo=halo && r.U[f][rindex(i,j,k)]==r.U[f][rindex(wrap(i),wrap(j),wrap(k))];
        audit.push_back({"radiation",rcycle,hcycle,time,E,F,halo,before==grid_ptr->U});
    }
    future_fixture exchange_rad_flux_corrections() { fluxKeys.push_back(rcycle);audit.push_back({"flux",rcycle,hcycle,boundaries.back().second,rad_grid_ptr->Uhalf[0][rindex(RAD_BW,RAD_BW,RAD_BW)],physcon().c*rad_grid_ptr->Uhalf[1][rindex(RAD_BW,RAD_BW,RAD_BW)],true,true});return {}; }
    void compute_radiation(Real,Real);
};
