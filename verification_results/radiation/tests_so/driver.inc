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
    void all_hydro_bounds() { ++hcycle; }
    void all_rad_bounds(Real time) { boundaries.emplace_back(rcycle,time);periodic(*rad_grid_ptr);++rcycle; }
    future_fixture exchange_rad_flux_corrections() { fluxKeys.push_back(rcycle);return {}; }
    void compute_radiation(Real,Real);
};
