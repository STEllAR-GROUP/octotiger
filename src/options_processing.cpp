//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"
#include "octotiger/options_compatibility.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/common_kernel/interaction_constants.hpp"

#include <boost/program_options.hpp>
#if HPX_VERSION_FULL > 0x010600
// Can't find hpx::find_all_localities() in newer HPX versions without this header
#include <hpx/modules/runtime_distributed.hpp>
#endif

#include <cmath>
#include <iosfwd>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <fstream>

#define IN_OPTIONS_CPP

constexpr Real mass_solar = 1.2969;
constexpr Real number_solar = 1.0994;
constexpr Real X_solar = 0.7068;
constexpr Real Z_solar = 0.0181;


inline std::string to_string(const std::string &str) {
	return str;
}

inline std::string to_string(const Real &num) {
	std::ostringstream strm;
	strm << std::scientific << num;
	return strm.str();
}

inline std::string to_string(const integer &num) {
	return std::to_string(num);
}

inline std::string to_string(const size_t &num) {
	return std::to_string(num);
}

inline std::string to_string(const bool &b) {
	return b ? "T" : "F";
}

bool options::process_options(int argc, char *argv[]) {
	namespace po = boost::program_options;
	using namespace octotiger::options_compatibility;
	code_to_s = code_to_g = code_to_cm = 1.0;

	po::options_description legacy_opts("Legacy options (deprecated; retained for compatibility)");

	legacy_opts.add_options() //
	("help", "produce help message")("xscale", po::value<Real>(&(opts().xscale))->default_value(1.0), "grid scale")           //
	("dt_max", po::value<Real>(&(opts().dt_max))->default_value(0.333333), "max allowed pct change for positive fields in a timestep")           //
	("cfl", po::value<Real>(&(opts().cfl))->default_value(0.4), "cfl factor")           //
	("omega", po::value<Real>(&(opts().omega))->default_value(0.0), "(initial) angular frequency")                          //
	("v1309", po::value<bool>(&(opts().v1309))->default_value(false), "V1309 subproblem of DWD")                   //
	("idle_rates", po::value<bool>(&(opts().idle_rates))->default_value(false), "show idle rates and locality info in SILO")                 //
	("eblast0", po::value<Real>(&(opts().eblast0))->default_value(1.0), "energy for blast wave")     //
	("rho_floor", po::value<Real>(&(opts().rho_floor))->default_value(0.0), "density floor")     //
	("tau_floor", po::value<Real>(&(opts().tau_floor))->default_value(0.0), "entropy tracer floor")     //
	("sod_rhol", po::value<Real>(&(opts().sod_rhol))->default_value(1.0), "density in the left part of the grid")     //
	("sod_rhor", po::value<Real>(&(opts().sod_rhor))->default_value(0.125), "density in the right part of the grid")     //
	("sod_pl", po::value<Real>(&(opts().sod_pl))->default_value(1.0), "pressure in the left part of the grid")     //
	("sod_pr", po::value<Real>(&(opts().sod_pr))->default_value(0.1), "pressure in the right part of the grid")     //
	("sod_theta", po::value<Real>(&(opts().sod_theta))->default_value(0.0), "angle made by diaphragm normal w/x-axis (deg)")     //
	("sod_phi", po::value<Real>(&(opts().sod_phi))->default_value(90.0), "angle made by diaphragm normal w/z-axis (deg)")     //
	("sod_gamma", po::value<Real>(&(opts().sod_gamma))->default_value(1.4), "ratio of specific heats for gas")     //
        ("solid_sphere_xcenter", po::value<Real>(&(opts().solid_sphere_xcenter))->default_value(0.25), "x-position of the sphere center")     //
        ("solid_sphere_ycenter", po::value<Real>(&(opts().solid_sphere_ycenter))->default_value(0.0), "y-position of the sphere center")     //
        ("solid_sphere_zcenter", po::value<Real>(&(opts().solid_sphere_zcenter))->default_value(0.0), "z-position of the sphere center")     //
        ("solid_sphere_radius", po::value<Real>(&(opts().solid_sphere_radius))->default_value(1.0 / 3.0), "radius of the sphere")     //
        ("solid_sphere_mass", po::value<Real>(&(opts().solid_sphere_mass))->default_value(1.0), "total mass enclosed inside the sphere")     //
        ("solid_sphere_rho_min", po::value<Real>(&(opts().solid_sphere_rho_min))->default_value(1.0e-12), "minimal density outside (and within) the sphere")     //
        ("star_xcenter", po::value<Real>(&(opts().star_xcenter))->default_value(0.0), "x-position of the star center")     //
        ("star_ycenter", po::value<Real>(&(opts().star_ycenter))->default_value(0.0), "y-position of the star center")     //
        ("star_zcenter", po::value<Real>(&(opts().star_zcenter))->default_value(0.0), "z-position of the star center")     //
        ("star_n", po::value<Real>(&(opts().star_n))->default_value(1.5), "polytropic index of the star")     //
        ("star_rmax", po::value<Real>(&(opts().star_rmax))->default_value(1.0 / 3.0), "maximal star radius")     //
        ("star_dr", po::value<Real>(&(opts().star_dr))->default_value(1.0 / (3.0 * 128.0)), "differential radius for solving the Lane-Emden equation")     //
        ("star_alpha", po::value<Real>(&(opts().star_alpha))->default_value(1.0 / (3.0 * 3.65375)), "scaling factor for the Lane-Emden equation") // for default n=3/2, ksi_1=3.65375 and alpha=rmax/ksi_1
        ("star_rho_center", po::value<Real>(&(opts().star_rho_center))->default_value(1.0), "density at the center of the star")     //
        ("star_rho_out", po::value<Real>(&(opts().star_rho_out))->default_value(1.0e-10), "density outside the star")     //
	("star_egas_out", po::value<Real>(&(opts().star_egas_out))->default_value(1.0e-10), "gas energy outside the star")     //
        ("moving_star_xvelocity", po::value<Real>(&(opts().moving_star_xvelocity))->default_value(1.0), "velocity of the star in the x-direction")     //
        ("moving_star_yvelocity", po::value<Real>(&(opts().moving_star_yvelocity))->default_value(1.0), "velocity of the star in the y-direction")     //
        ("moving_star_zvelocity", po::value<Real>(&(opts().moving_star_zvelocity))->default_value(1.0), "velocity of the star in the z-direction")     //
	("driving_rate", po::value<Real>(&(opts().driving_rate))->default_value(0.0), "angular momentum loss driving rate")     //
	("driving_time", po::value<Real>(&(opts().driving_time))->default_value(0.0), "A.M. driving rate time")                 //
	("entropy_driving_time", po::value<Real>(&(opts().entropy_driving_time))->default_value(0.0), "entropy driving rate time")                 //
	("entropy_driving_rate", po::value<Real>(&(opts().entropy_driving_rate))->default_value(0.0), "entropy loss driving rate")      //
	("future_wait_time", po::value<integer>(&(opts().future_wait_time))->default_value(-1), "")      //
	("silo_offset_x", po::value<integer>(&(opts().silo_offset_x))->default_value(0), "")      //
	("silo_offset_y", po::value<integer>(&(opts().silo_offset_y))->default_value(0), "")      //
	("silo_offset_z", po::value<integer>(&(opts().silo_offset_z))->default_value(0), "")      //
	("amrbnd_order", po::value<integer>(&(opts().amrbnd_order))->default_value(1), "amr boundary interpolation order")        //
	("scf_output_frequency", po::value<integer>(&(opts().scf_output_frequency))->default_value(25), "Frequency of SCF output")        //
	("scf_rho_floor", po::value<Real>(&(opts().scf_rho_floor))->default_value(1.0e-12), "scf density floor")     //
	("silo_num_groups", po::value<integer>(&(opts().silo_num_groups))->default_value(-1), "Number of SILO I/O groups")        //
	("core_refine", po::value<bool>(&(opts().core_refine))->default_value(false), "refine cores by one more level")           //
	("grad_rho_refine", po::value<Real>(&(opts().grad_rho_refine))->default_value(-1.0), "density gradient refinement criteria (-1=off)")           //
	("accretor_refine", po::value<integer>(&(opts().accretor_refine))->default_value(0), "number of extra levels for accretor") //
	("extra_regrid", po::value<integer>(&(opts().extra_regrid))->default_value(0), "number of extra regrids on startup") //
	("donor_refine", po::value<integer>(&(opts().donor_refine))->default_value(0), "number of extra levels for donor")      //
	("ngrids", po::value<integer>(&(opts().ngrids))->default_value(-1), "fix numbger of grids")                             //
	("refinement_floor", po::value<Real>(&(opts().refinement_floor))->default_value(1.0e-3), "density refinement floor")      //
	("theta", po::value<Real>(&(opts().theta))->default_value(0.5), "controls nearness determination for FMM, must be between 1/3 and 1/2")               //
	("eos", po::value<eos_type>(&(opts().eos))->default_value(IDEAL), "gas equation of state")                              //
        ("ipr_nr_tol", po::value<Real>(&(opts().ipr_nr_tol))->default_value(1.48e-08), "Newton-Raphson tolerance for solving ideal gas plus radiation eos")                              //
        ("ipr_nr_maxiter", po::value<integer>(&(opts().ipr_nr_maxiter))->default_value(50), "Newton-Raphson max iterations for solving ideal gas plus radiation eos")                              //
        ("ipr_test", po::value<bool>(&(opts().ipr_test))->default_value(false), "test consistency of the ideal gas plus radiation eos")                              //
        ("ipr_eint_floor", po::value<Real>(&(opts().ipr_eint_floor))->default_value(0.0), "floor thermal energy for ideal gas plus radiation eos")                              //
	("hydro", po::value<bool>(&(opts().hydro))->default_value(true), "hydro on/off")    //
	("periodic", po::value<bool>(&(opts().periodic))->default_value(false), "periodic hydro boundary conditions")    //
	("radiation", po::value<bool>(&(opts().radiation))->default_value(false), "radiation on/off")    //
	("correct_am_hydro", po::value<bool>(&(opts().correct_am_hydro))->default_value(false), "Angular momentum correction switch for hydro")    //
	("correct_am_grav", po::value<bool>(&(opts().correct_am_grav))->default_value(true), "Angular momentum correction switch for gravity")    //
	("rewrite_silo", po::value<bool>(&(opts().rewrite_silo))->default_value(false), "rewrite silo and exit")    //
	("rad_implicit", po::value<bool>(&(opts().rad_implicit))->default_value(true), "implicit radiation on/off")    //
	("rad_subcycling", po::value<bool>(&(opts().rad_subcycling))->default_value(true), "Enable radiation subcycles inside each gas timestep")
	("rad_c_ratio", po::value<Real>(&(opts().rad_c_ratio))->default_value(1.0), "Reduced light speed c_hat/c in (0,1]; physical c and stored F are unchanged")
	("rad_cfl", po::value<Real>(&(opts().rad_cfl))->default_value(0.4), "Radiation Courant number for the sum-of-direction speed bound, in (0,0.5]")
	("rad_max_subcycles", po::value<integer>(&(opts().rad_max_subcycles))->default_value(1024), "Maximum radiation steps per gas step; caps the gas timestep")
	("rad_theta", po::value<Real>(&(opts().rad_theta))->default_value(1.0), "Source theta in [0.5,1]; 1=backward Euler, 0.51=near trapezoidal; stiff fallback to 1")
	("rad_velocity_terms", po::value<bool>(&(opts().rad_velocity_terms))->default_value(true), "Include S&O O(v/c) work and O(beta tau) momentum sources")
	("rad_opacity", po::value<Real>(&(opts().rad_opacity))->default_value(-1.0), "Constant gray opacity per mass in code units; negative uses existing Planck/Rosseland opacities")
	("rad_energy_mode", po::value<std::string>(&(opts().rad_energy_mode))->default_value("thermal"), "Radiation energy exchange: thermal, absorption (energy escapes this band), equilibrium")
	("rad_log_subcycles", po::value<bool>(&(opts().rad_log_subcycles))->default_value(false), "Print global gas/radiation timesteps and radiation subcycle counts")
	("gravity", po::value<bool>(&(opts().gravity))->default_value(true), "gravity on/off")    //
	("bench", po::value<bool>(&(opts().bench))->default_value(false), "run benchmark") //
	("datadir", po::value<std::string>(&(opts().data_dir))->default_value("./"), "directory for output") //
	("output", po::value<std::string>(&(opts().output_filename))->default_value(""), "filename for output") //
	("odt", po::value<Real>(&(opts().output_dt))->default_value(1.0 / 100.0), "output frequency") //
	("dual_energy_sw1", po::value<Real>(&(opts().dual_energy_sw1))->default_value(0.001), "dual energy switch 1") //
	("dual_energy_sw2", po::value<Real>(&(opts().dual_energy_sw2))->default_value(0.1), "dual energy switch 2") //
	("hard_dt", po::value<Real>(&(opts().hard_dt))->default_value(-1), "timestep size") //
	("experiment", po::value<int>(&(opts().experiment))->default_value(0), "experiment") //
	("unigrid", po::value<bool>(&(opts().unigrid))->default_value(false), "unigrid") //
	("inflow_bc", po::value<bool>(&(opts().inflow_bc))->default_value(false), "Inflow Boundary Conditions") //
	("reflect_bc", po::value<bool>(&(opts().reflect_bc))->default_value(false), "Reflecting Boundary Conditions") //
	("cdisc_detect", po::value<bool>(&(opts().cdisc_detect))->default_value(true), "PPM contact discontinuity detection") //
	("disable_output", po::value<bool>(&(opts().disable_output))->default_value(false), "disable silo output") //
	("rad_reference", po::value<std::string>(&(opts().radReference))->default_value("gaussian_pulse.bin"), "Gaussian radiation reference generated by gen_radiation_reference")
	("rad_test_chi", po::value<Real>(&(opts().radTestChi))->default_value(5), "Regression extinction coefficient [inverse code length]")
	("rad_test_width", po::value<Real>(&(opts().radTestWidth))->default_value(.2), "Regression Gaussian width [code length]")
	("rad_test_background", po::value<Real>(&(opts().radTestBackground))->default_value(1), "Regression background radiation energy density")
	("rad_test_amplitude", po::value<Real>(&(opts().radTestAmplitude))->default_value(.01), "Regression Gaussian peak perturbation")
	("rad_test_luminosity", po::value<Real>(&(opts().radTestLuminosity))->default_value(.01), "Regression Gaussian bulb luminosity")
	("disable_analytic", po::value<bool>(&(opts().disable_analytic))->default_value(false), "disable analytic step") //
	("disable_diagnostics", po::value<bool>(&(opts().disable_diagnostics))->default_value(false), "disable diagnostics") //
	("problem", po::value<problem_type>(&(opts().problem))->default_value(NONE), "problem type")                            //
	("restart_filename", po::value<std::string>(&(opts().restart_filename))->default_value(""), "restart filename")         //
	("stop_time", po::value<Real>(&(opts().stop_time))->default_value(std::numeric_limits<Real>::max()), "time to end simulation") //
	("stop_step", po::value<integer>(&(opts().stop_step))->default_value(std::numeric_limits<integer>::max() - 1), "number of timesteps to run")          //
	("min_level", po::value<integer>(&(opts().min_level))->default_value(1), "minimum number of refinement levels")         //
	("max_level", po::value<integer>(&(opts().max_level))->default_value(1), "maximum number of refinement levels")         //
	("amr_boundary_kernel_type", po::value<amr_boundary_type>(&(opts().amr_boundary_kernel_type))->default_value(AMR_OPTIMIZED), "amr completion kernel type") //
#ifdef OCTOTIGER_HAVE_KOKKOS //Changing default kernel to kokkos
	("multipole_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().multipole_host_kernel_type))->default_value(KOKKOS), "Host kernel type for multipole interactions ") //
	("multipole_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().multipole_device_kernel_type))->default_value(OFF), "Device kernel type for multipole interactions ") //
	("monopole_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().monopole_host_kernel_type))->default_value(KOKKOS), "Host kernel type for monopole interactions ") //
	("monopole_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().monopole_device_kernel_type))->default_value(OFF), "Device kernel type for monopole interactions ") //
	("hydro_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().hydro_host_kernel_type))->default_value(KOKKOS), "Host kernel type for the hydro solver ") //
	("hydro_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().hydro_device_kernel_type))->default_value(OFF), "Device kernel type for the hydro solver ") //
#else 
	("multipole_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().multipole_host_kernel_type))->default_value(VC), "Host kernel type for multipole interactions ") //
	("multipole_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().multipole_device_kernel_type))->default_value(OFF), "Device kernel type for multipole interactions ") //
	("monopole_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().monopole_host_kernel_type))->default_value(VC), "Host kernel type for monopole interactions ") //
	("monopole_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().monopole_device_kernel_type))->default_value(OFF), "Device kernel type for monopole interactions ") //
	("hydro_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().hydro_host_kernel_type))->default_value(LEGACY), "Host kernel type for the hydro solver ") //
	("hydro_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().hydro_device_kernel_type))->default_value(OFF), "Device kernel type for the hydro solver ") //
#endif
	("number_gpus", po::value<size_t>(&(opts().number_gpus))->default_value(size_t(0)), "cuda streams per HPX locality") //
	("executors_per_gpu", po::value<size_t>(&(opts().executors_per_gpu))->default_value(size_t(0)), "cuda streams per GPU (per locality)") //
	("max_gpu_executor_queue_length", po::value<size_t>(&(opts().max_gpu_executor_queue_length))->default_value(size_t(5)), "How many launches should be buffered before using the CPU") //
("polling-threads", po::value<int>(&(opts().polling_threads))->default_value(0), "Enable dedicated HPX thread pool for cuda/network polling using N threads!") //
	("max_kernels_fused", po::value<size_t>(&(opts().max_kernels_fused))->default_value(size_t(1)), "Maximum numbers of kernels combined into one by the dynamic work aggegation") //
	("root_node_on_device", po::value<bool>(&(opts().root_node_on_device))->default_value(true), "Offload root node gravity kernels to the GPU? May degrade performance given weak GPUs") //
	("optimize_local_communication", po::value<bool>(&(opts().optimize_local_communication))->default_value(true), "Use pointers of neighbors in local subgrids directly") //
	("print_times_per_timestep", po::value<bool>(&(opts().print_times_per_timestep))->default_value(false), "Print times per timestep during cleanup") //
	("input_file", po::value<std::string>(&(opts().input_file))->default_value(""), "input file for test problems") //
	("config_file", po::value<std::string>(&(opts().config_file))->default_value(""), "configuration file") //
	("n_species", po::value<integer>(&(opts().n_species))->default_value(5), "number of mass species") //
	("atomic_mass", po::value<std::vector<Real>>(&(opts().atomic_mass))->multitoken(), "atomic masses") //
	("atomic_number", po::value<std::vector<Real>>(&(opts().atomic_number))->multitoken(), "atomic numbers") //
	("X", po::value<std::vector<Real>>(&(opts().X))->multitoken(), "X - hydrogen mass fraction") //
	("Z", po::value<std::vector<Real>>(&(opts().Z))->multitoken(), "Z - metallicity") //
	("code_to_g", po::value<Real>(&(opts().code_to_g))->default_value(1), "code units to grams") //
	("code_to_cm", po::value<Real>(&(opts().code_to_cm))->default_value(1), "code units to centimeters") //
	("code_to_s", po::value<Real>(&(opts().code_to_s))->default_value(1), "code units to seconds") //
	("rotating_star_amr", po::value<bool>(&(opts().rotating_star_amr))->default_value(false), "rotating star with AMR boundary in star") //
	("rotating_star_x", po::value<Real>(&(opts().rotating_star_x))->default_value(0.0), "x center of rotating_star") //
			;

    // Boost.Program_options option names are deliberately flat strings.  The
    // dots are our canonical namespace; they do not imply parser nesting.
    po::options_description canonical_opts("Canonical options");
    std::vector<migration> migrations;
#define CANONICAL(canonical, legacy, member)                                      \
    add_canonical_option(canonical_opts, migrations, canonical, legacy,           \
        &(opts().member))
#define CANONICAL_MULTI(canonical, legacy, member)                                \
    add_canonical_multitoken_option(canonical_opts, migrations, canonical, legacy,\
        &(opts().member))
    canonical_opts.add_options()("runtime.help", "produce help message");
    migrations.emplace_back("help", "runtime.help");
    CANONICAL("mesh.scale", "xscale", xscale);
    CANONICAL("timestep.max_change", "dt_max", dt_max);
    CANONICAL("hydro.cfl", "cfl", cfl);
    CANONICAL("gravity.angular_frequency", "omega", omega);
    CANONICAL("problem.dwd.v1309", "v1309", v1309);
    CANONICAL("output.idle_rates", "idle_rates", idle_rates);
    CANONICAL("blast.energy", "eblast0", eblast0);
    CANONICAL("hydro.density_floor", "rho_floor", rho_floor);
    CANONICAL("hydro.entropy_floor", "tau_floor", tau_floor);
    CANONICAL("problem.sod.density_left", "sod_rhol", sod_rhol);
    CANONICAL("problem.sod.density_right", "sod_rhor", sod_rhor);
    CANONICAL("problem.sod.pressure_left", "sod_pl", sod_pl);
    CANONICAL("problem.sod.pressure_right", "sod_pr", sod_pr);
    CANONICAL("problem.sod.theta", "sod_theta", sod_theta);
    CANONICAL("problem.sod.phi", "sod_phi", sod_phi);
    CANONICAL("hydro.gamma", "sod_gamma", sod_gamma);
    CANONICAL("problem.solid_sphere.center_x", "solid_sphere_xcenter", solid_sphere_xcenter);
    CANONICAL("problem.solid_sphere.center_y", "solid_sphere_ycenter", solid_sphere_ycenter);
    CANONICAL("problem.solid_sphere.center_z", "solid_sphere_zcenter", solid_sphere_zcenter);
    CANONICAL("problem.solid_sphere.radius", "solid_sphere_radius", solid_sphere_radius);
    CANONICAL("problem.solid_sphere.mass", "solid_sphere_mass", solid_sphere_mass);
    CANONICAL("problem.solid_sphere.minimum_density", "solid_sphere_rho_min", solid_sphere_rho_min);
    CANONICAL("problem.star.center_x", "star_xcenter", star_xcenter);
    CANONICAL("problem.star.center_y", "star_ycenter", star_ycenter);
    CANONICAL("problem.star.center_z", "star_zcenter", star_zcenter);
    CANONICAL("problem.star.polytropic_index", "star_n", star_n);
    CANONICAL("problem.star.maximum_radius", "star_rmax", star_rmax);
    CANONICAL("problem.star.radial_step", "star_dr", star_dr);
    CANONICAL("problem.star.alpha", "star_alpha", star_alpha);
    CANONICAL("problem.star.central_density", "star_rho_center", star_rho_center);
    CANONICAL("problem.star.external_density", "star_rho_out", star_rho_out);
    CANONICAL("problem.star.external_gas_energy", "star_egas_out", star_egas_out);
    CANONICAL("problem.moving_star.velocity_x", "moving_star_xvelocity", moving_star_xvelocity);
    CANONICAL("problem.moving_star.velocity_y", "moving_star_yvelocity", moving_star_yvelocity);
    CANONICAL("problem.moving_star.velocity_z", "moving_star_zvelocity", moving_star_zvelocity);
    CANONICAL("problem.driving.angular_momentum.rate", "driving_rate", driving_rate);
    CANONICAL("problem.driving.angular_momentum.duration", "driving_time", driving_time);
    CANONICAL("problem.driving.entropy.duration", "entropy_driving_time", entropy_driving_time);
    CANONICAL("problem.driving.entropy.rate", "entropy_driving_rate", entropy_driving_rate);
    CANONICAL("runtime.future_wait_time", "future_wait_time", future_wait_time);
    CANONICAL("output.silo.offset_x", "silo_offset_x", silo_offset_x);
    CANONICAL("output.silo.offset_y", "silo_offset_y", silo_offset_y);
    CANONICAL("output.silo.offset_z", "silo_offset_z", silo_offset_z);
    CANONICAL("mesh.amr.boundary_order", "amrbnd_order", amrbnd_order);
    CANONICAL("problem.scf.output_frequency", "scf_output_frequency", scf_output_frequency);
    CANONICAL("problem.scf.density_floor", "scf_rho_floor", scf_rho_floor);
    CANONICAL("output.silo.groups", "silo_num_groups", silo_num_groups);
    CANONICAL("mesh.refinement.core", "core_refine", core_refine);
    CANONICAL("mesh.refinement.density_gradient", "grad_rho_refine", grad_rho_refine);
    CANONICAL("mesh.refinement.accretor_levels", "accretor_refine", accretor_refine);
    CANONICAL("mesh.extra_initial_regrids", "extra_regrid", extra_regrid);
    CANONICAL("mesh.refinement.donor_levels", "donor_refine", donor_refine);
    CANONICAL("mesh.fixed_grid_count", "ngrids", ngrids);
    CANONICAL("mesh.refinement.density_floor", "refinement_floor", refinement_floor);
    CANONICAL("gravity.opening_angle", "theta", theta);
    CANONICAL("hydro.eos", "eos", eos);
    CANONICAL("hydro.ipr.newton_tolerance", "ipr_nr_tol", ipr_nr_tol);
    CANONICAL("hydro.ipr.newton_max_iterations", "ipr_nr_maxiter", ipr_nr_maxiter);
    CANONICAL("hydro.ipr.test", "ipr_test", ipr_test);
    CANONICAL("hydro.ipr.internal_energy_floor", "ipr_eint_floor", ipr_eint_floor);
    CANONICAL("hydro.enabled", "hydro", hydro);
    CANONICAL("hydro.boundary.periodic", "periodic", periodic);
    CANONICAL("radiation.enabled", "radiation", radiation);
    CANONICAL("hydro.angular_momentum_correction", "correct_am_hydro", correct_am_hydro);
    CANONICAL("gravity.angular_momentum_correction", "correct_am_grav", correct_am_grav);
    CANONICAL("output.rewrite_silo", "rewrite_silo", rewrite_silo);
    CANONICAL("radiation.implicit", "rad_implicit", rad_implicit);
    CANONICAL("radiation.subcycling", "rad_subcycling", rad_subcycling);
    CANONICAL("radiation.reduced_light_speed_ratio", "rad_c_ratio", rad_c_ratio);
    CANONICAL("radiation.cfl", "rad_cfl", rad_cfl);
    CANONICAL("radiation.max_subcycles", "rad_max_subcycles", rad_max_subcycles);
    CANONICAL("radiation.source_theta", "rad_theta", rad_theta);
    CANONICAL("radiation.velocity_terms", "rad_velocity_terms", rad_velocity_terms);
    CANONICAL("radiation.opacity.constant", "rad_opacity", rad_opacity);
    CANONICAL("radiation.energy_mode", "rad_energy_mode", rad_energy_mode);
    CANONICAL("radiation.log_subcycles", "rad_log_subcycles", rad_log_subcycles);
    CANONICAL("gravity.enabled", "gravity", gravity);
    CANONICAL("runtime.benchmark", "bench", bench);
    CANONICAL("output.directory", "datadir", data_dir);
    CANONICAL("output.filename", "output", output_filename);
    CANONICAL("output.interval", "odt", output_dt);
    CANONICAL("hydro.dual_energy.switch1", "dual_energy_sw1", dual_energy_sw1);
    CANONICAL("hydro.dual_energy.switch2", "dual_energy_sw2", dual_energy_sw2);
    CANONICAL("timestep.fixed", "hard_dt", hard_dt);
    CANONICAL("problem.experiment", "experiment", experiment);
    CANONICAL("mesh.unigrid", "unigrid", unigrid);
    CANONICAL("hydro.boundary.inflow", "inflow_bc", inflow_bc);
    CANONICAL("hydro.boundary.reflecting", "reflect_bc", reflect_bc);
    CANONICAL("hydro.contact_discontinuity_detection", "cdisc_detect", cdisc_detect);
    CANONICAL("output.disabled", "disable_output", disable_output);
    CANONICAL("radiation.test.reference", "rad_reference", radReference);
    CANONICAL("radiation.test.extinction", "rad_test_chi", radTestChi);
    CANONICAL("radiation.test.width", "rad_test_width", radTestWidth);
    CANONICAL("radiation.test.background", "rad_test_background", radTestBackground);
    CANONICAL("radiation.test.amplitude", "rad_test_amplitude", radTestAmplitude);
    CANONICAL("radiation.test.luminosity", "rad_test_luminosity", radTestLuminosity);
    CANONICAL("problem.disable_analytic", "disable_analytic", disable_analytic);
    CANONICAL("runtime.disable_diagnostics", "disable_diagnostics", disable_diagnostics);
    CANONICAL("problem.name", "problem", problem);
    CANONICAL("restart.filename", "restart_filename", restart_filename);
    CANONICAL("runtime.stop_time", "stop_time", stop_time);
    CANONICAL("runtime.stop_step", "stop_step", stop_step);
    CANONICAL("mesh.level.minimum", "min_level", min_level);
    CANONICAL("mesh.level.maximum", "max_level", max_level);
    CANONICAL("execution.kernel.amr_boundary", "amr_boundary_kernel_type", amr_boundary_kernel_type);
    CANONICAL("execution.kernel.multipole.host", "multipole_host_kernel_type", multipole_host_kernel_type);
    CANONICAL("execution.kernel.multipole.device", "multipole_device_kernel_type", multipole_device_kernel_type);
    CANONICAL("execution.kernel.monopole.host", "monopole_host_kernel_type", monopole_host_kernel_type);
    CANONICAL("execution.kernel.monopole.device", "monopole_device_kernel_type", monopole_device_kernel_type);
    CANONICAL("execution.kernel.hydro.host", "hydro_host_kernel_type", hydro_host_kernel_type);
    CANONICAL("execution.kernel.hydro.device", "hydro_device_kernel_type", hydro_device_kernel_type);
    CANONICAL("execution.gpu.count", "number_gpus", number_gpus);
    CANONICAL("execution.gpu.executors_per_gpu", "executors_per_gpu", executors_per_gpu);
    CANONICAL("execution.gpu.max_queue_length", "max_gpu_executor_queue_length", max_gpu_executor_queue_length);
    CANONICAL("execution.polling_threads", "polling-threads", polling_threads);
    CANONICAL("execution.max_kernels_fused", "max_kernels_fused", max_kernels_fused);
    CANONICAL("execution.root_node_on_device", "root_node_on_device", root_node_on_device);
    CANONICAL("execution.optimize_local_communication", "optimize_local_communication", optimize_local_communication);
    CANONICAL("runtime.print_times_per_timestep", "print_times_per_timestep", print_times_per_timestep);
    CANONICAL("problem.input_file", "input_file", input_file);
    CANONICAL("runtime.config_file", "config_file", config_file);
    CANONICAL("hydro.species.count", "n_species", n_species);
    CANONICAL_MULTI("hydro.species.atomic_mass", "atomic_mass", atomic_mass);
    CANONICAL_MULTI("hydro.species.atomic_number", "atomic_number", atomic_number);
    CANONICAL_MULTI("hydro.species.hydrogen_fraction", "X", X);
    CANONICAL_MULTI("hydro.species.metallicity", "Z", Z);
    CANONICAL("units.grams", "code_to_g", code_to_g);
    CANONICAL("units.centimeters", "code_to_cm", code_to_cm);
    CANONICAL("units.seconds", "code_to_s", code_to_s);
    CANONICAL("problem.rotating_star.amr", "rotating_star_amr", rotating_star_amr);
    CANONICAL("problem.rotating_star.center_x", "rotating_star_x", rotating_star_x);
#undef CANONICAL_MULTI
#undef CANONICAL

    po::options_description command_opts("All options");
    command_opts.add(canonical_opts).add(legacy_opts);

	boost::program_options::variables_map vm;
	std::set<std::string> supplied;
    auto const command_line =
        po::command_line_parser(argc, argv).options(command_opts).allow_unregistered().run();
    remember_supplied(supplied, command_line);
    po::store(command_line, vm);
	po::notify(vm);
	reapply_canonical_values(vm, canonical_opts, migrations);
	if (vm.count("help") || vm.count("runtime.help")) {
		if (!check_compatibility_spellings(supplied, migrations)) {
			return false;
		}
		warn_legacy_spellings(supplied, migrations);
		std::cout << canonical_opts << "\n\n" << legacy_opts << "\n";
		return false;
	}
	if (!config_file.empty()) {
		std::ifstream cfg_fs { config_file };
		if (cfg_fs) {
			auto const config = po::parse_config_file(cfg_fs, command_opts);
            remember_supplied(supplied, config);
            po::store(config, vm);
		} else {
			printf("Configuration file %s not found!\n", config_file.c_str());
			return false;
		}
	}
	if (!check_compatibility_spellings(supplied, migrations)) {
        return false;
    }
	po::notify(vm);
	reapply_canonical_values(vm, canonical_opts, migrations);
	warn_legacy_spellings(supplied, migrations);
	if (opts().silo_num_groups == -1) {
		opts().silo_num_groups = hpx::find_all_localities().size();

	}
	if (opts().problem == DWD) {
		opts().n_species = std::max(int(5), int(opts().n_species));
	}
        if (opts().problem == MOVING_STAR || opts().problem == ROTATING_STAR) {
                opts().n_species = std::max(int(2), int(opts().n_species));
	}
	n_fields = n_species + 10;
	if (!opts().restart_filename.empty()) {
		// Preserve the already-resolved config/CLI value across checkpoint loading.
		auto const explicit_setting = [&supplied](char const* legacy, char const* canonical) {
			return supplied.count(legacy) != 0 || supplied.count(canonical) != 0;
		};
		auto const explicit_rad_implicit = opts().rad_implicit;
		auto const explicit_rad_subcycling = opts().rad_subcycling;
		auto const explicit_rad_c_ratio = opts().rad_c_ratio;
		auto const explicit_rad_cfl = opts().rad_cfl;
		auto const explicit_rad_max_subcycles = opts().rad_max_subcycles;
		auto const explicit_rad_theta = opts().rad_theta;
		auto const explicit_rad_velocity_terms = opts().rad_velocity_terms;
		auto const explicit_rad_opacity = opts().rad_opacity;
		auto const explicit_rad_energy_mode = opts().rad_energy_mode;
		auto const explicit_rad_log_subcycles = opts().rad_log_subcycles;
		FILE *fp = fopen(opts().restart_filename.c_str(), "rb");
		if (fp == NULL) {
			printf("restart.silo does not exist or invalid permissions\n");
			sleep(10);
			abort();
		} else {
			fclose(fp);
		}
		load_options_from_silo(opts().restart_filename);
        // Explicit CLI/config values override radiation checkpoint defaults,
        // irrespective of whether the canonical or legacy spelling was used.
        if (explicit_setting("rad_implicit", "radiation.implicit")) opts().rad_implicit=explicit_rad_implicit;
        if (explicit_setting("rad_subcycling", "radiation.subcycling")) opts().rad_subcycling=explicit_rad_subcycling;
        if (explicit_setting("rad_c_ratio", "radiation.reduced_light_speed_ratio")) opts().rad_c_ratio=explicit_rad_c_ratio;
        if (explicit_setting("rad_cfl", "radiation.cfl")) opts().rad_cfl=explicit_rad_cfl;
        if (explicit_setting("rad_max_subcycles", "radiation.max_subcycles")) opts().rad_max_subcycles=explicit_rad_max_subcycles;
        if (explicit_setting("rad_theta", "radiation.source_theta")) opts().rad_theta=explicit_rad_theta;
        if (explicit_setting("rad_velocity_terms", "radiation.velocity_terms")) opts().rad_velocity_terms=explicit_rad_velocity_terms;
        if (explicit_setting("rad_opacity", "radiation.opacity.constant")) opts().rad_opacity=explicit_rad_opacity;
        if (explicit_setting("rad_energy_mode", "radiation.energy_mode")) opts().rad_energy_mode=explicit_rad_energy_mode;
        if (explicit_setting("rad_log_subcycles", "radiation.log_subcycles")) opts().rad_log_subcycles=explicit_rad_log_subcycles;

	}
    // Validate after restart metadata is loaded as well as after CLI/config parsing.
    if (!(std::isfinite(rad_c_ratio) && rad_c_ratio>0 && rad_c_ratio<=1) ||
        !(std::isfinite(rad_cfl) && rad_cfl>0 && rad_cfl<=.5) ||
        !(std::isfinite(rad_theta) && rad_theta>=.5 && rad_theta<=1) ||
        !std::isfinite(rad_opacity) || rad_max_subcycles<1 ||
        (rad_energy_mode!="thermal" && rad_energy_mode!="absorption" && rad_energy_mode!="equilibrium")) {
        std::cerr << "Invalid radiation options: require 0<rad_c_ratio<=1, 0<rad_cfl<=.5, "
                     ".5<=rad_theta<=1, finite rad_opacity, rad_max_subcycles>=1, "
                     "rad_energy_mode=thermal|absorption|equilibrium\n";
        return false;
    }
    if (opts().executors_per_gpu > 0 && opts().number_gpus == 0) {
        opts().number_gpus = 1;
	}
	if (opts().theta < octotiger::fmm::THETA_FLOOR) {
		std::cerr << "theta " << theta << " is too small since Octo-Tiger was compiled for a minimum of " << octotiger::fmm::THETA_FLOOR << std::endl;
		std::cerr << "Either increase theta or recompile with a new theta minimum using the cmake parameter OCTOTIGER_THETA_MINIMUM";
		abort();
	}
    if (opts().correct_am_hydro) {
		std::cerr << std::endl;
		std::cerr << "WARNING: correct_am_hydro=1 is obsolete, setting to 0" << std::endl;
		std::cerr << "(pausing for 10 seconds)" << std::endl;
		opts().correct_am_hydro = 0;
		hpx::this_thread::sleep_for(std::chrono::seconds(10));       
	}
    opts().detected_intel_compiler = false;

#ifdef __VERSION__
  std::string compiler_version = std::string(__VERSION__);
  std::cout << "Using compiler " << compiler_version << std::endl;
  if (compiler_version.find("intel") != std::string::npos) {
    std::cout << "Detected Intel compiler..." << '\n';
    opts().detected_intel_compiler=true;
  }
#endif
	{
#define SHOW( opt ) std::cout << std::string( #opt ) << " = " << to_string(opt) << '\n';
		std::cout << "atomic_number=";
		for (auto r : atomic_number) {
			std::cout << std::to_string(r) << ',';
		}
		std::cout << '\n';
		std::cout << "atomic_mass=";
		for (auto r : atomic_mass) {
			std::cout << std::to_string(r) << ',';
		}
		std::cout << '\n';
		std::cout << "X=";
		for (auto r : X) {
			std::cout << std::to_string(r) << ',';
		}
		std::cout << '\n';
		std::cout << "Z=";
		for (auto r : Z) {
			std::cout << std::to_string(r) << ',';
		}
		std::cout << '\n';
		const auto num_loc = hpx::find_all_localities().size();
		if (silo_num_groups > num_loc) {
			printf("Number of SILO file groups cannot be greater than number of localities. Setting silo_num_groupds to %li\n", num_loc);
			silo_num_groups = num_loc;
		}
		SHOW(accretor_refine);
		SHOW(amrbnd_order);
		SHOW(bench);
		SHOW(cdisc_detect);
		SHOW(cfl);
		SHOW(config_file);
		SHOW(core_refine);
		SHOW(correct_am_grav);
		SHOW(correct_am_hydro);
		SHOW(code_to_cm);
		SHOW(code_to_g);
		SHOW(code_to_s);
		SHOW(data_dir);
		SHOW(disable_output);
		SHOW(driving_rate);
		SHOW(driving_time);
		SHOW(dt_max);
		SHOW(donor_refine);
		SHOW(dual_energy_sw1);
		SHOW(dual_energy_sw2);
		SHOW(eblast0);
		SHOW(eos);
		SHOW(entropy_driving_rate);
		SHOW(entropy_driving_time);
		SHOW(future_wait_time);
		SHOW(grad_rho_refine);
		SHOW(hard_dt);
		SHOW(hydro);
		SHOW(inflow_bc);
		SHOW(input_file);
		SHOW(min_level);
		SHOW(max_level);
		SHOW(n_species);
		SHOW(ngrids);
		SHOW(omega);
		SHOW(output_dt);
		SHOW(output_filename);
		SHOW(problem);
		SHOW(rad_implicit);
		SHOW(rad_subcycling);
		SHOW(rad_c_ratio);
		SHOW(rad_cfl);
		SHOW(rad_max_subcycles);
		SHOW(rad_theta);
		SHOW(rad_velocity_terms);
		SHOW(rad_opacity);
		SHOW(rad_energy_mode);
		SHOW(rad_log_subcycles);

		SHOW(radReference);
		SHOW(radTestChi);
		SHOW(radTestWidth);
		SHOW(radTestBackground);
		SHOW(radTestAmplitude);
		SHOW(radTestLuminosity);
		SHOW(radiation);
		SHOW(refinement_floor);
		SHOW(reflect_bc);
		SHOW(restart_filename);
		SHOW(rotating_star_amr);
		SHOW(rotating_star_x);
		SHOW(scf_output_frequency);
		SHOW(silo_num_groups);
		SHOW(stop_step);
		SHOW(stop_time);
		SHOW(theta);
		SHOW(unigrid);
		SHOW(v1309);
		SHOW(idle_rates);
		SHOW(xscale);
		SHOW(number_gpus);
		SHOW(executors_per_gpu);
		SHOW(max_gpu_executor_queue_length);
		SHOW(max_kernels_fused);
		SHOW(amr_boundary_kernel_type);
		SHOW(root_node_on_device);
		SHOW(optimize_local_communication);
		SHOW(multipole_device_kernel_type);
		SHOW(multipole_host_kernel_type);
		SHOW(monopole_device_kernel_type);
		SHOW(monopole_host_kernel_type);
		SHOW(hydro_device_kernel_type);
		SHOW(hydro_host_kernel_type);

	}
	while (atomic_number.size() < opts().n_species) {
		atomic_number.push_back(number_solar);
	}
	while (atomic_mass.size() < opts().n_species) {
		atomic_mass.push_back(mass_solar);
	}
	while (X.size() < opts().n_species) {
		X.push_back(X_solar);
	}
	while (Z.size() < opts().n_species) {
		Z.push_back(Z_solar);
	}
	normalize_constants();
	if (opts().problem == DWD) {
		if (opts().restart_filename == "" && opts().disable_diagnostics) {
			printf("Diagnostics must be enabled for DWD\n");
			sleep(10);
			abort();
		}
	}
    // Check parameters if we hit any implementation limitation as in
    // unsupported kernel configurations
    if (opts().periodic && opts().gravity) {
        std::cerr << "ERROR! Periodic hydro boundaries require gravity to be disabled (--gravity=0)." << std::endl;
        abort();
    }
    if (opts().gravity) {
#ifdef OCTOTIGER_DISABLE_ILIST
        std::cerr << "ERROR! Gravity is turned on but Octo-Tiger was compiled without interaction list" << std::endl
	          << "Either run a scenario without gravity, or remove OCTOTIGER_DISABLE_ILIST from cmake and recompile!" << std::endl;
        abort();
#endif
        if (opts().multipole_device_kernel_type == interaction_device_kernel_type::CUDA &&
            opts().multipole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Due to a current implementation limitation in the load balancing, " 
            << " multipole cuda device kernels cannot be mixed with the respective kokkos host kernel!" << std::endl
            << " Please choose a different host kernel "
            << "(or move to kokkos device kernel with --multipole_device_kernel_type=KOKKOS_CUDA)" << std::endl;
            abort();
        }
        if (opts().monopole_device_kernel_type == interaction_device_kernel_type::CUDA &&
            opts().monopole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Due to a current implementation limitation in the load balancing, " 
            << " monopole cuda device kernels cannot be mixed with the respective kokkos host kernel!" << std::endl
            << " Please choose a different host kernel "
            << "(or move to kokkos device kernel with --monopole_device_kernel_type=KOKKOS_CUDA)" << std::endl;
            abort();
        }
        if (opts().multipole_device_kernel_type == interaction_device_kernel_type::HIP &&
            opts().multipole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Due to a current implementation limitation in the load balancing, " 
            << " multipole hip device kernels cannot be mixed with the respective kokkos host kernel!" << std::endl
            << " Please choose a different host kernel "
            << "(or move to kokkos device kernel with --multipole_device_kernel_type=KOKKOS_HIP)" << std::endl;
            abort();
        }
        if (opts().monopole_device_kernel_type == interaction_device_kernel_type::HIP &&
            opts().monopole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Due to a current implementation limitation in the load balancing, " 
            << " monopole hip device kernels cannot be mixed with the respective kokkos host kernel!" << std::endl
            << " Please choose a different host kernel "
            << "(or move to kokkos device kernel with --monopole_device_kernel_type=KOKKOS_HIP)" << std::endl;
            abort();
        }
#ifndef OCTOTIGER_HAVE_VC
        if (opts().monopole_host_kernel_type == interaction_host_kernel_type::VC) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Octotiger has been compiled without Vc support!" << 
            " Choose a different --monopole_host_kernel_type!" << std::endl;
            abort();
        }
        if (opts().multipole_host_kernel_type == interaction_host_kernel_type::VC) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Octotiger has been compiled without Vc support! " <<
            "Choose a different --multipole_host_kernel_type!" << std::endl;
            abort();
        }
#endif
#ifndef OCTOTIGER_HAVE_KOKKOS
        if (opts().monopole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Octotiger has been compiled without Kokkos support!" 
            << " Choose a different --monopole_host_kernel_type!" << std::endl;
            abort();
        }
        if (opts().multipole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Octotiger has been compiled without Kokkos support! " <<
            " Choose a different --multipole_host_kernel_type!" << std::endl;
            abort();
        }
#endif

#ifndef OCTOTIGER_HAVE_CUDA
        if (opts().monopole_device_kernel_type == interaction_device_kernel_type::CUDA) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Octotiger has been compiled without CUDA support!" 
            << " Choose a different --monopole_device_kernel_type!" << std::endl;
            abort();
        }
        if (opts().multipole_device_kernel_type == interaction_device_kernel_type::CUDA) {
            std::cerr << std::endl << "ERROR: "; 
            std::cerr << "Octotiger has been compiled without CUDA support! " <<
            " Choose a different --multipole_device_kernel_type!" << std::endl;
            abort();
        }
#endif

    }
    if (opts().hydro_host_kernel_type == interaction_host_kernel_type::VC) {
        std::cerr << std::endl
                  << "ERROR: hydro host VC kernel is experimental - use either LEGACY or "
                     "KOKKOS (or disable this error for a dev/test build!) ";
        abort();
    }
    if (opts().eos == IPR) {
        if ((opts().hydro_host_kernel_type != interaction_host_kernel_type::VC) && (opts().hydro_host_kernel_type != interaction_host_kernel_type::LEGACY)) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "The ideal gas plus radiation (ipr) eos is currently only supported with LEGACY host kernel types of the hydro solver!"  << std::endl
            << " Choose either a LEGACY for hydro host kernel type or use a different eos!" << std::endl;
            abort();
        }
        if (opts().hydro_device_kernel_type != OFF) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "The ideal gas plus radiation (ipr) eos is currently only supported on the host!"  << std::endl
            << " Choose OFF for hydro device kernel type or use a different eos!" << std::endl;
            abort();
        }
        if (opts().cdisc_detect) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "The ideal gas plus radiation (ipr) eos is currently not supported with discontiniuty detection!"  << std::endl
            << " Either set cdisc_detect to off or use a different eos!" << std::endl;
            abort();        
        }
        if (opts().radiation) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "The ideal gas plus radiation (ipr) eos is currently not supported together with radiation field on!"  << std::endl
            << " Either set radiation to off or use a different eos!" << std::endl;
            abort();
        }
    }
    if (opts().executors_per_gpu < 1 && (opts().monopole_device_kernel_type != OFF ||
          opts().multipole_device_kernel_type != OFF || opts().hydro_device_kernel_type != OFF)) {
        std::cerr << std::endl << "ERROR: "; 
        std::cerr << "You have chosen an GPU kernel, however, you did not specify --executors_per_gpu > 0" << std::endl
        << " Choose a different kernel or add at least one or more executors via --executors_per_gpu=X" << std::endl;
        abort();
    }
    if (opts().max_kernels_fused < 1 && (opts().monopole_device_kernel_type != OFF ||
          opts().multipole_device_kernel_type != OFF || opts().hydro_device_kernel_type != OFF)) {
        std::cerr << std::endl << "ERROR: "; 
        std::cerr << "minimum value for --max_kernels_fused is 1 when a GPU kernel is active!" << std::endl;
        abort();
    }
    if (opts().monopole_device_kernel_type == OFF && opts().monopole_host_kernel_type == DEVICE_ONLY ||
        opts().multipole_device_kernel_type == OFF && opts().multipole_host_kernel_type == DEVICE_ONLY ||
	opts().hydro_device_kernel_type == OFF && opts().hydro_host_kernel_type == DEVICE_ONLY) {
        std::cerr << std::endl << "ERROR: "; 
        std::cerr << "You have disabled both host kernel- and device kernel execution!" << std::endl
        << " Choose a different host or device kernel type!" << std::endl;
        abort();
    }
#ifdef OCTOTIGER_HAVE_FAST_FP
    if (opts().monopole_device_kernel_type != OFF && opts().monopole_host_kernel_type != DEVICE_ONLY ||
        opts().multipole_device_kernel_type != OFF && opts().multipole_host_kernel_type != DEVICE_ONLY ||
	opts().hydro_device_kernel_type != OFF && opts().hydro_host_kernel_type != DEVICE_ONLY) {
        std::cerr << std::endl << "ERROR: "; 
        std::cerr << std::endl << "Octotiger has been build with OCTOTIGER_WITH_FAST_FP_CONTRACT=ON "; 
        std::cerr << " - This build configuration only supports either the host or the device kernel active, not both of them at the same time!" << std::endl
        << "Disable either the device kernel (OFF) or the host kernel (DEVICE_ONLY)." << std::endl;
        abort();
    }
#ifdef OCTOTIGER_HAVE_HIP
   if (opts().monopole_host_kernel_type == DEVICE_ONLY) {
     std::cerr << "\nWARNING: Monopole DEVICE_ONLY is currently not fully supported in HIP builds!!" << std::endl;
     std::cerr << "p2m kernel always executed on the cpu in this build..." << std::endl << std::endl;
     sleep(1);
   }

#endif
#endif

    return true;
}

std::vector<hpx::id_type> options::all_localities = { };
