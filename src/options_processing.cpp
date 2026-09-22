//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/options.hpp"
#include "octotiger/options_compatibility.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/grey_opacity_options.hpp"

#include <boost/program_options.hpp>
#if HPX_VERSION_FULL > 0x010600
// Can't find hpx::find_all_localities() in newer HPX versions without this header
#include <hpx/modules/runtime_distributed.hpp>
#endif

#include <cmath>
#include <functional>
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

inline std::string to_string(const std::string& str) {
    return str;
}

inline std::string to_string(const Real& num) {
    std::ostringstream strm;
    strm << std::scientific << num;
    return strm.str();
}

inline std::string to_string(const integer& num) {
    return std::to_string(num);
}

inline std::string to_string(const int& num) {
    return std::to_string(num);
}

inline std::string to_string(const size_t& num) {
    return std::to_string(num);
}

inline std::string to_string(const bool& b) {
    return b ? "T" : "F";
}

template <typename T>
std::string to_string(std::vector<T> const& values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << to_string(values[index]);
    }
    output << ']';
    return output.str();
}

struct OptionDisplay {
    std::string canonical;
    std::string legacy;
    std::function<std::string()> value;
};

bool options::process_options(int argc, char* argv[]) {
    namespace po = boost::program_options;
    using namespace octotiger::optionsCompatibility;
    code_to_s = code_to_g = code_to_cm = 1.0;

    po::options_description legacyOptions("Legacy options (deprecated; retained for compatibility)");

    legacyOptions.add_options()                                                                                              //
        ("xscale", po::value<Real>(&(opts().xscale))->default_value(1.0),
            "Set the root-domain half-width in code-length units.")    //
        ("dt_max", po::value<Real>(&(opts().dt_max))->default_value(0.333333),
            "Limit the fractional change of positive fields during one timestep.")    //
        ("cfl", po::value<Real>(&(opts().cfl))->default_value(0.4),
            "Set the hydrodynamic Courant factor.")    //
        ("omega", po::value<Real>(&(opts().omega))->default_value(0.0),
            "Set the initial z-component of mesh angular velocity in inverse code-time units.")    //
        ("omega_x", po::value<Real>(&(opts().omegaX))->default_value(0.0),
            "Set the initial x-component of mesh angular velocity in inverse code-time units.")    //
        ("omega_y", po::value<Real>(&(opts().omegaY))->default_value(0.0),
            "Set the initial y-component of mesh angular velocity in inverse code-time units.")    //
        ("v1309", po::value<bool>(&(opts().v1309))->default_value(false),
            "Enable the V1309 Sco variant of the double-white-dwarf problem.")    //
        ("idle_rates", po::value<bool>(&(opts().idle_rates))->default_value(false),
            "Write HPX idle-rate and locality diagnostics to Silo output.")    //
        ("eblast0", po::value<Real>(&(opts().eblast0))->default_value(1.0),
            "Set the initial blast-wave energy in code-energy units.")    //
        ("rho_floor", po::value<Real>(&(opts().rho_floor))->default_value(0.0),
            "Set the minimum hydrodynamic density in code-density units.")    //
        ("tau_floor", po::value<Real>(&(opts().tau_floor))->default_value(0.0),
            "Set the minimum entropy-tracer value.")    //
        ("sod_rhol", po::value<Real>(&(opts().sod_rhol))->default_value(1.0),
            "Set the left-state density for the Sod problem.")    //
        ("sod_rhor", po::value<Real>(&(opts().sod_rhor))->default_value(0.125),
            "Set the right-state density for the Sod problem.")    //
        ("sod_pl", po::value<Real>(&(opts().sod_pl))->default_value(1.0),
            "Set the left-state pressure for the Sod problem.")    //
        ("sod_pr", po::value<Real>(&(opts().sod_pr))->default_value(0.1),
            "Set the right-state pressure for the Sod problem.")    //
        ("sod_theta", po::value<Real>(&(opts().sod_theta))->default_value(0.0),
            "Set the angle between the Sod diaphragm normal and x-axis, in degrees.")    //
        ("sod_phi", po::value<Real>(&(opts().sod_phi))->default_value(90.0),
            "Set the angle between the Sod diaphragm normal and z-axis, in degrees.")    //
        ("sod_gamma", po::value<Real>(&(opts().sod_gamma))->default_value(1.4),
            "Set the gas adiabatic index (ratio of specific heats).")    //
        ("solid_sphere_xcenter", po::value<Real>(&(opts().solid_sphere_xcenter))->default_value(0.25),
            "Set the solid-sphere center along x in code-length units.")    //
        ("solid_sphere_ycenter", po::value<Real>(&(opts().solid_sphere_ycenter))->default_value(0.0),
            "Set the solid-sphere center along y in code-length units.")    //
        ("solid_sphere_zcenter", po::value<Real>(&(opts().solid_sphere_zcenter))->default_value(0.0),
            "Set the solid-sphere center along z in code-length units.")    //
        ("solid_sphere_radius", po::value<Real>(&(opts().solid_sphere_radius))->default_value(1.0 / 3.0),
            "Set the solid-sphere radius in code-length units.")    //
        ("solid_sphere_mass", po::value<Real>(&(opts().solid_sphere_mass))->default_value(1.0),
            "Set the total mass of the solid-sphere problem.")    //
        ("solid_sphere_rho_min", po::value<Real>(&(opts().solid_sphere_rho_min))->default_value(1.0e-12),
            "Set the minimum density inside and outside the solid sphere.")    //
        ("star_xcenter", po::value<Real>(&(opts().star_xcenter))->default_value(0.0),
            "Set the stellar center along x in code-length units.")    //
        ("star_ycenter", po::value<Real>(&(opts().star_ycenter))->default_value(0.0),
            "Set the stellar center along y in code-length units.")    //
        ("star_zcenter", po::value<Real>(&(opts().star_zcenter))->default_value(0.0),
            "Set the stellar center along z in code-length units.")    //
        ("star_n", po::value<Real>(&(opts().star_n))->default_value(1.5),
            "Set the stellar polytropic index.")    //
        ("star_rmax", po::value<Real>(&(opts().star_rmax))->default_value(1.0 / 3.0),
            "Set the stellar outer radius in code-length units.")    //
        ("star_dr", po::value<Real>(&(opts().star_dr))->default_value(1.0 / (3.0 * 128.0)),
            "Set the radial integration step for the Lane-Emden equation.")    //
        ("star_alpha", po::value<Real>(&(opts().star_alpha))->default_value(1.0 / (3.0 * 3.65375)),
            "Set the Lane-Emden radial scale factor.")    // for default n=3/2, ksi_1=3.65375 and
                                                             // alpha=rmax/ksi_1
        ("star_rho_center", po::value<Real>(&(opts().star_rho_center))->default_value(1.0),
            "Set the stellar central density in code-density units.")    //
        ("star_rho_out", po::value<Real>(&(opts().star_rho_out))->default_value(1.0e-10),
            "Set the ambient density outside the star.")    //
        ("star_egas_out", po::value<Real>(&(opts().star_egas_out))->default_value(1.0e-10),
            "Set the ambient gas-energy density outside the star.")    //
        ("moving_star_xvelocity", po::value<Real>(&(opts().moving_star_xvelocity))->default_value(1.0),
            "Set the moving-star velocity along x in code-speed units.")    //
        ("moving_star_yvelocity", po::value<Real>(&(opts().moving_star_yvelocity))->default_value(1.0),
            "Set the moving-star velocity along y in code-speed units.")    //
        ("moving_star_zvelocity", po::value<Real>(&(opts().moving_star_zvelocity))->default_value(1.0),
            "Set the moving-star velocity along z in code-speed units.")    //
        ("driving_rate", po::value<Real>(&(opts().driving_rate))->default_value(0.0),
            "Set the imposed angular-momentum loss rate.")    //
        ("driving_time", po::value<Real>(&(opts().driving_time))->default_value(0.0),
            "Set the duration of angular-momentum driving in code-time units.")    //
        ("entropy_driving_time", po::value<Real>(&(opts().entropy_driving_time))->default_value(0.0),
            "Set the duration of entropy driving in code-time units.")    //
        ("entropy_driving_rate", po::value<Real>(&(opts().entropy_driving_rate))->default_value(0.0),
            "Set the imposed entropy-loss rate.")    //
        ("future_wait_time", po::value<integer>(&(opts().future_wait_time))->default_value(-1),
            "Set the future-wait timeout in seconds; negative disables the timeout.")    //
        ("silo_offset_x", po::value<integer>(&(opts().silo_offset_x))->default_value(0),
            "Offset Silo mesh indices in the x direction.")    //
        ("silo_offset_y", po::value<integer>(&(opts().silo_offset_y))->default_value(0),
            "Offset Silo mesh indices in the y direction.")    //
        ("silo_offset_z", po::value<integer>(&(opts().silo_offset_z))->default_value(0),
            "Offset Silo mesh indices in the z direction.")    //
        ("amrbnd_order", po::value<integer>(&(opts().amrbnd_order))->default_value(1),
            "Set the AMR boundary interpolation order.")    //
        ("scf_output_frequency", po::value<integer>(&(opts().scf_output_frequency))->default_value(25),
            "Write SCF output every N iterations.")    //
        ("scf_rho_floor", po::value<Real>(&(opts().scf_rho_floor))->default_value(1.0e-12),
            "Set the SCF density floor in code-density units.")    //
        ("silo_num_groups", po::value<integer>(&(opts().silo_num_groups))->default_value(-1),
            "Set the number of parallel Silo I/O groups; -1 uses one group per locality.")    //
        ("core_refine", po::value<bool>(&(opts().core_refine))->default_value(false),
            "Refine stellar cores by one additional AMR level.")    //
        ("grad_rho_refine", po::value<Real>(&(opts().grad_rho_refine))->default_value(-1.0),
            "Set the density-gradient refinement threshold; negative disables it.")    //
        ("accretor_refine", po::value<integer>(&(opts().accretor_refine))->default_value(0),
            "Add this many AMR levels around the accretor.")    //
        ("extra_regrid", po::value<integer>(&(opts().extra_regrid))->default_value(0),
            "Perform this many additional regrids during startup.")    //
        ("donor_refine", po::value<integer>(&(opts().donor_refine))->default_value(0),
            "Add this many AMR levels around the donor.")    //
        ("ndim", po::value<integer>(&(opts().dimensionCount))->default_value(3),
            "Set the mesh dimensionality (1, 2, or 3).")    //
        ("modular_hydro", po::value<bool>(&(opts().modularHydro))->default_value(false),
            "Use distributed modular ideal-gas hydro with fixed mesh and independent shadow.")
        ("modular_problem", po::value<std::string>(&(opts().modularProblem))->default_value("sod"),
            "Modular initial condition: sod, advection, or kelvinHelmholtz.")
        ("modular_transport", po::value<bool>(&(opts().modularTransport))->default_value(false),
            "Use dimension-aware distributed modular hydro/radiation transport.")
        ("modular_radiation_source_free", po::value<bool>(&(opts().modularRadiationSourceFree))->default_value(false),
            "Explicitly select source-free radiation; no gas-radiation coupling.")
        ("modular_radiation_problem", po::value<std::string>(&(opts().modularRadiationProblem))->default_value("streamingGaussian"),
            "Modular radiation initial condition: streamingGaussian or isotropicPulse.")
        ("ngrids", po::value<integer>(&(opts().ngrids))->default_value(-1),
            "Target this number of grids when adapting the refinement floor; negative disables the target.")    //
        ("refinement_floor", po::value<Real>(&(opts().refinement_floor))->default_value(1.0e-3),
            "Set the density floor used by the AMR refinement criterion.")    //
        ("theta", po::value<Real>(&(opts().theta))->default_value(0.5),
            "Set the FMM opening angle; valid values are between 1/3 and 1/2.")    //
        ("eos", po::value<eos_type>(&(opts().eos))->default_value(IDEAL),
            "Select the hydrodynamic equation of state.")    //
        ("ipr_nr_tol", po::value<Real>(&(opts().ipr_nr_tol))->default_value(1.48e-08),
            "Set the Newton-Raphson tolerance for the ideal-gas-plus-radiation EOS.")    //
        ("ipr_nr_maxiter", po::value<integer>(&(opts().ipr_nr_maxiter))->default_value(50),
            "Set the maximum Newton-Raphson iterations for the ideal-gas-plus-radiation EOS.")    //
        ("ipr_test", po::value<bool>(&(opts().ipr_test))->default_value(false),
            "Enable consistency checks for the ideal-gas-plus-radiation EOS.")    //
        ("ipr_eint_floor", po::value<Real>(&(opts().ipr_eint_floor))->default_value(0.0),
            "Set the thermal-energy floor for the ideal-gas-plus-radiation EOS.")    //
        ("hydro", po::value<bool>(&(opts().hydro))->default_value(true),
            "Enable hydrodynamics.")    //
        ("periodic", po::value<bool>(&(opts().periodic))->default_value(false),
            "Use periodic mesh boundary conditions.")    //
        ("radiation", po::value<bool>(&(opts().radiation))->default_value(false),
            "Enable radiation transport.")    //
        ("correct_am_hydro", po::value<bool>(&(opts().correct_am_hydro))->default_value(false),
            "Obsolete and unsupported hydrodynamic angular-momentum correction.")    //
        ("correct_am_grav", po::value<bool>(&(opts().correct_am_grav))->default_value(true),
            "Enable the gravity angular-momentum correction.")    //
        ("rewrite_silo", po::value<bool>(&(opts().rewrite_silo))->default_value(false),
            "Rewrite the selected Silo restart file and exit.")    //
        ("rad_implicit", po::value<bool>(&(opts().rad_implicit))->default_value(true),
            "Enable implicit radiation-matter source coupling.")    //
        ("rad_subcycling", po::value<bool>(&(opts().radSubcycling))->default_value(true),
            "Subcycle radiation transport within each gas timestep.")(
            "rad_c_ratio", po::value<Real>(&(opts().radCRatio))->default_value(1.0),
            "Set the reduced light speed ratio c-hat/c in (0, 1]; physical c and stored F are unchanged.")(
            "rad_cfl", po::value<Real>(&(opts().radCfl))->default_value(0.4),
            "Set the radiation Courant factor for the summed directional-speed bound in (0, 0.5].")(
            "rad_max_subcycles", po::value<integer>(&(opts().radMaxSubcycles))->default_value(1024),
            "Set the maximum radiation subcycles per gas step; this can cap the gas timestep.")(
            "rad_theta", po::value<Real>(&(opts().radTheta))->default_value(1.0),
            "Set the radiation-source theta in [0.5, 1]; 1 is backward Euler and stiff solves fall back to 1.")(
            "rad_velocity_terms", po::value<bool>(&(opts().radVelocityTerms))->default_value(true),
            "Include Skinner-Ostriker O(v/c) work and O(beta*tau) momentum-source terms.")(
            "rad_opacity", po::value<Real>(&(opts().radOpacity))->default_value(-1.0),
            "Set constant grey opacity per mass in code units; negative selects the existing Planck/Rosseland opacities.")(
            "rad_energy_mode", po::value<std::string>(&(opts().radEnergyMode))->default_value("thermal"),
            "Select radiation energy exchange: thermal, absorption, or equilibrium.")(
            "rad_log_subcycles", po::value<bool>(&(opts().radLogSubcycles))->default_value(false),
            "Print gas and radiation timesteps and the radiation subcycle count.")(
            "gravity", po::value<bool>(&(opts().gravity))->default_value(true),
            "Enable self-gravity.")    //
        ("bench", po::value<bool>(&(opts().bench))->default_value(false),
            "Run the built-in benchmark mode.")    //
        ("datadir", po::value<std::string>(&(opts().data_dir))->default_value("./"),
            "Set the directory for simulation output.")    //
        ("output", po::value<std::string>(&(opts().output_filename))->default_value(""),
            "Set the output filename stem.")    //
        ("detailed_log", po::value<std::string>(&(opts().detailedLogPath))->default_value(""),
            "Write root-level actions and numerical details to this text file.")    //
        ("results_file", po::value<std::string>(&(opts().resultsPath))->default_value(""),
            "Write stable machine-readable run results to this JSON file.")    //
        ("odt", po::value<Real>(&(opts().output_dt))->default_value(1.0 / 100.0),
            "Set the simulation-time interval between outputs.")    //
        ("dual_energy_sw1", po::value<Real>(&(opts().dual_energy_sw1))->default_value(0.001),
            "Set the first dual-energy threshold for recovering internal energy.")    //
        ("dual_energy_sw2", po::value<Real>(&(opts().dual_energy_sw2))->default_value(0.1),
            "Set the second dual-energy threshold for synchronizing the entropy tracer.")    //
        ("hard_dt", po::value<Real>(&(opts().hard_dt))->default_value(-1),
            "Use this fixed timestep in code-time units; negative selects adaptive timesteps.")    //
        ("experiment", po::value<int>(&(opts().experiment))->default_value(0),
            "Select an experimental hydrodynamics mode by numeric identifier.")    //
        ("unigrid", po::value<bool>(&(opts().unigrid))->default_value(false),
            "Disable adaptive refinement and use a uniform grid.")    //
        ("inflow_bc", po::value<bool>(&(opts().inflow_bc))->default_value(false),
            "Use inflow mesh boundary conditions.")    //
        ("reflect_bc", po::value<bool>(&(opts().reflect_bc))->default_value(false),
            "Use reflecting mesh boundary conditions.")    //
        ("cdisc_detect", po::value<bool>(&(opts().cdisc_detect))->default_value(true),
            "Enable PPM contact-discontinuity detection.")    //
        ("disable_output", po::value<bool>(&(opts().disable_output))->default_value(false),
            "Disable Silo output.")    //
        ("rad_reference", po::value<std::string>(&(opts().radReference))->default_value("gaussian_pulse.bin"),
            "Set the Gaussian-pulse reference file generated by gen_radiation_reference.")(
            "rad_test_chi", po::value<Real>(&(opts().radTestChi))->default_value(5),
            "Set the prescribed regression extinction coefficient in inverse code-length units.")(
            "rad_test_width", po::value<Real>(&(opts().radTestWidth))->default_value(.2),
            "Set the regression Gaussian width in code-length units.")(
            "rad_test_background", po::value<Real>(&(opts().radTestBackground))->default_value(1),
            "Set the regression background radiation energy density.")(
            "rad_test_amplitude", po::value<Real>(&(opts().radTestAmplitude))->default_value(.01),
            "Set the regression Gaussian peak perturbation.")(
            "rad_test_luminosity", po::value<Real>(&(opts().radTestLuminosity))->default_value(.01),
            "Set the regression Gaussian-source luminosity.")(
            "disable_analytic", po::value<bool>(&(opts().disable_analytic))->default_value(false),
            "Disable analytic updates for the selected problem.")    //
        ("disable_diagnostics", po::value<bool>(&(opts().disable_diagnostics))->default_value(false),
            "Disable runtime diagnostics.")    //
        ("problem", po::value<problem_type>(&(opts().problem))->default_value(NONE),
            "Select the initial-value problem.")    //
        ("restart_filename", po::value<std::string>(&(opts().restart_filename))->default_value(""),
            "Restart from this Silo file.")    //
        ("stop_time", po::value<Real>(&(opts().stop_time))->default_value(std::numeric_limits<Real>::max()),
            "Stop when simulation time reaches this value.")    //
        ("stop_step", po::value<integer>(&(opts().stop_step))->default_value(std::numeric_limits<integer>::max() - 1),
            "Stop after this many timesteps.")    //
        ("min_level", po::value<integer>(&(opts().min_level))->default_value(1),
            "Set the minimum mesh-refinement level.")    //
        ("max_level", po::value<integer>(&(opts().max_level))->default_value(1),
            "Set the maximum mesh-refinement level.")    //
        ("amr_boundary_kernel_type", po::value<amr_boundary_type>(&(opts().amr_boundary_kernel_type))->default_value(AMR_OPTIMIZED),
            "Select the AMR boundary-completion kernel implementation.")    //
#ifdef OCTOTIGER_HAVE_KOKKOS                 // Changing default kernel to kokkos
        ("multipole_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().multipole_host_kernel_type))->default_value(KOKKOS),
            "Select the host kernel for multipole interactions.")    //
        ("multipole_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().multipole_device_kernel_type))->default_value(OFF),
            "Select the device kernel for multipole interactions.")    //
        ("monopole_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().monopole_host_kernel_type))->default_value(KOKKOS),
            "Select the host kernel for monopole interactions.")    //
        ("monopole_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().monopole_device_kernel_type))->default_value(OFF),
            "Select the device kernel for monopole interactions.")    //
        ("hydro_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().hydro_host_kernel_type))->default_value(KOKKOS),
            "Select the host kernel for the hydrodynamics solver.")    //
        ("hydro_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().hydro_device_kernel_type))->default_value(OFF),
            "Select the device kernel for the hydrodynamics solver.")    //
#else
        ("multipole_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().multipole_host_kernel_type))->default_value(VC),
            "Select the host kernel for multipole interactions.")    //
        ("multipole_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().multipole_device_kernel_type))->default_value(OFF),
            "Select the device kernel for multipole interactions.")    //
        ("monopole_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().monopole_host_kernel_type))->default_value(VC),
            "Select the host kernel for monopole interactions.")    //
        ("monopole_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().monopole_device_kernel_type))->default_value(OFF),
            "Select the device kernel for monopole interactions.")    //
        ("hydro_host_kernel_type", po::value<interaction_host_kernel_type>(&(opts().hydro_host_kernel_type))->default_value(LEGACY),
            "Select the host kernel for the hydrodynamics solver.")    //
        ("hydro_device_kernel_type", po::value<interaction_device_kernel_type>(&(opts().hydro_device_kernel_type))->default_value(OFF),
            "Select the device kernel for the hydrodynamics solver.")    //
#endif
        ("number_gpus", po::value<size_t>(&(opts().number_gpus))->default_value(size_t(0)),
            "Set the number of GPUs available to each HPX locality.")    //
        ("executors_per_gpu", po::value<size_t>(&(opts().executors_per_gpu))->default_value(size_t(0)),
            "Set the number of executor streams per GPU and locality.")    //
        ("max_gpu_executor_queue_length", po::value<size_t>(&(opts().max_gpu_executor_queue_length))->default_value(size_t(5)),
            "Use the CPU when a GPU executor has this many queued launches.")    //
        ("polling-threads", po::value<int>(&(opts().polling_threads))->default_value(0),
            "Reserve this many processing units in a dedicated HPX pool for GPU and network polling; 0 uses the default pool.")    //
        ("max_kernels_fused", po::value<size_t>(&(opts().max_kernels_fused))->default_value(size_t(1)),
            "Set the maximum kernels combined by dynamic work aggregation.")    //
        ("root_node_on_device", po::value<bool>(&(opts().root_node_on_device))->default_value(true),
            "Offload root-node gravity kernels to a GPU.")    //
        ("optimize_local_communication", po::value<bool>(&(opts().optimize_local_communication))->default_value(true),
            "Use direct neighbor pointers for subgrids on the same locality.")    //
        ("print_times_per_timestep", po::value<bool>(&(opts().print_times_per_timestep))->default_value(false),
            "Print per-timestep timing data during cleanup.")    //
        ("input_file", po::value<std::string>(&(opts().input_file))->default_value(""),
            "Read problem-specific initial data from this file.")    //
        ("config_file", po::value<std::string>(&(opts().config_file))->default_value(""),
            "Read Octo-TIGER options from this configuration file.")    //
        ("n_species", po::value<integer>(&(opts().n_species))->default_value(5),
            "Set the number of advected mass species.")    //
        ("atomic_mass", po::value<std::vector<Real>>(&(opts().atomic_mass))->multitoken(),
            "Set the mean atomic mass for each species, in atomic mass units.")    //
        ("atomic_number", po::value<std::vector<Real>>(&(opts().atomic_number))->multitoken(),
            "Set the mean atomic number for each species.")    //
        ("X", po::value<std::vector<Real>>(&(opts().X))->multitoken(),
            "Obsolete and unsupported; specify composition with hydro.species.atomic_mass and hydro.species.atomic_number.")    //
        ("Z", po::value<std::vector<Real>>(&(opts().Z))->multitoken(),
            "Obsolete and unsupported; specify composition with hydro.species.atomic_mass and hydro.species.atomic_number.")    //
        ("code_to_g", po::value<Real>(&(opts().code_to_g))->default_value(1),
            "Set the grams represented by one code-mass unit.")    //
        ("code_to_cm", po::value<Real>(&(opts().code_to_cm))->default_value(1),
            "Set the centimeters represented by one code-length unit.")    //
        ("code_to_s", po::value<Real>(&(opts().code_to_s))->default_value(1),
            "Set the seconds represented by one code-time unit.")    //
        ("rotating_star_amr", po::value<bool>(&(opts().rotating_star_amr))->default_value(false),
            "Allow an AMR boundary to pass through the rotating star.")    //
        ("rotating_star_x", po::value<Real>(&(opts().rotating_star_x))->default_value(0.0),
            "Set the rotating-star center along x in code-length units.")    //
        ;

    // Boost.Program_options option names are deliberately flat strings.  The
    // dots are our canonical namespace; they do not imply parser nesting.
    po::options_description canonicalOptions("Canonical options");
    std::vector<migration> migrations;
    std::vector<OptionDisplay> displayedOptions;
#define canonicalOption(canonical, legacy, member)                                                                     \
    do {                                                                                                          \
        addCanonicalOption(canonicalOptions, legacyOptions, migrations, canonical, legacy, &(opts().member));     \
        displayedOptions.push_back(                                                                             \
            {canonical, legacy, []() { return to_string(opts().member); }});                                    \
    } while (false)
#define canonicalMultiOption(canonical, legacy, member)                                                               \
    do {                                                                                                          \
        addCanonicalMultitokenOption(                                                                         \
            canonicalOptions, legacyOptions, migrations, canonical, legacy, &(opts().member));                      \
        displayedOptions.push_back(                                                                             \
            {canonical, legacy, []() { return to_string(opts().member); }});                                    \
    } while (false)
    canonicalOptions.add_options()("help", "Display the command-line option reference and exit.");
    canonicalOption("execution.gpu.count", "number_gpus", number_gpus);
    canonicalOption("execution.gpu.executors_per_gpu", "executors_per_gpu", executors_per_gpu);
    canonicalOption("execution.gpu.max_queue_length", "max_gpu_executor_queue_length", max_gpu_executor_queue_length);
    canonicalOption("execution.kernel.amr_boundary", "amr_boundary_kernel_type", amr_boundary_kernel_type);
    canonicalOption("execution.kernel.hydro.device", "hydro_device_kernel_type", hydro_device_kernel_type);
    canonicalOption("execution.kernel.hydro.host", "hydro_host_kernel_type", hydro_host_kernel_type);
    canonicalOption("execution.kernel.monopole.device", "monopole_device_kernel_type", monopole_device_kernel_type);
    canonicalOption("execution.kernel.monopole.host", "monopole_host_kernel_type", monopole_host_kernel_type);
    canonicalOption("execution.kernel.multipole.device", "multipole_device_kernel_type", multipole_device_kernel_type);
    canonicalOption("execution.kernel.multipole.host", "multipole_host_kernel_type", multipole_host_kernel_type);
    canonicalOption("execution.max_kernels_fused", "max_kernels_fused", max_kernels_fused);
    canonicalOption("execution.optimize_local_communication", "optimize_local_communication", optimize_local_communication);
    canonicalOption("execution.polling_threads", "polling-threads", polling_threads);
    canonicalOption("execution.root_node_on_device", "root_node_on_device", root_node_on_device);
    canonicalOption("gravity.angular_momentum_correction", "correct_am_grav", correct_am_grav);
    canonicalOption("gravity.enabled", "gravity", gravity);
    canonicalOption("gravity.opening_angle", "theta", theta);
    canonicalOption("hydro.cfl", "cfl", cfl);
    canonicalOption("hydro.contact_discontinuity_detection", "cdisc_detect", cdisc_detect);
    canonicalOption("hydro.density_floor", "rho_floor", rho_floor);
    canonicalOption("hydro.dual_energy.switch1", "dual_energy_sw1", dual_energy_sw1);
    canonicalOption("hydro.dual_energy.switch2", "dual_energy_sw2", dual_energy_sw2);
    canonicalOption("hydro.enabled", "hydro", hydro);
    canonicalOption("hydro.entropy_floor", "tau_floor", tau_floor);
    canonicalOption("hydro.eos", "eos", eos);
    canonicalOption("hydro.gamma", "sod_gamma", sod_gamma);
    canonicalOption("hydro.ipr.internal_energy_floor", "ipr_eint_floor", ipr_eint_floor);
    canonicalOption("hydro.ipr.newton_max_iterations", "ipr_nr_maxiter", ipr_nr_maxiter);
    canonicalOption("hydro.ipr.newton_tolerance", "ipr_nr_tol", ipr_nr_tol);
    canonicalOption("hydro.ipr.test", "ipr_test", ipr_test);
    canonicalMultiOption("hydro.species.atomic_mass", "atomic_mass", atomic_mass);
    canonicalMultiOption("hydro.species.atomic_number", "atomic_number", atomic_number);
    canonicalOption("hydro.species.count", "n_species", n_species);
    canonicalOption("mesh.amr.boundary_order", "amrbnd_order", amrbnd_order);
    canonicalOption("mesh.boundary.inflow", "inflow_bc", inflow_bc);
    canonicalOption("mesh.boundary.periodic", "periodic", periodic);
    canonicalOption("mesh.boundary.reflecting", "reflect_bc", reflect_bc);
    canonicalOption("mesh.extra_initial_regrids", "extra_regrid", extra_regrid);
    canonicalOption("mesh.fixed_grid_count", "ngrids", ngrids);
    canonicalOption("mesh.level.maximum", "max_level", max_level);
    canonicalOption("mesh.level.minimum", "min_level", min_level);
    canonicalOption("mesh.ndim", "ndim", dimensionCount);
    canonicalOption("hydro.modular.enabled", "modular_hydro", modularHydro);
    canonicalOption("hydro.modular.problem", "modular_problem", modularProblem);
    canonicalOption("runtime.modular.enabled", "modular_transport", modularTransport);
    canonicalOption("radiation.modular.source_free", "modular_radiation_source_free", modularRadiationSourceFree);
    canonicalOption("radiation.modular.problem", "modular_radiation_problem", modularRadiationProblem);
    canonicalOption("mesh.omega_x", "omega_x", omegaX);
    canonicalOption("mesh.omega_y", "omega_y", omegaY);
    canonicalOption("mesh.omega_z", "omega", omega);
    canonicalOption("mesh.scale", "xscale", xscale);
    canonicalOption("mesh.unigrid", "unigrid", unigrid);
    canonicalOption("output.directory", "datadir", data_dir);
    canonicalOption("output.disabled", "disable_output", disable_output);
    canonicalOption("output.filename", "output", output_filename);
    canonicalOption("output.detailed_log", "detailed_log", detailedLogPath);
    canonicalOption("output.results_file", "results_file", resultsPath);
    canonicalOption("output.idle_rates", "idle_rates", idle_rates);
    canonicalOption("output.interval", "odt", output_dt);
    canonicalOption("output.rewrite_silo", "rewrite_silo", rewrite_silo);
    canonicalOption("output.silo.groups", "silo_num_groups", silo_num_groups);
    canonicalOption("output.silo.offset_x", "silo_offset_x", silo_offset_x);
    canonicalOption("output.silo.offset_y", "silo_offset_y", silo_offset_y);
    canonicalOption("output.silo.offset_z", "silo_offset_z", silo_offset_z);
    canonicalOption("problem.blast.energy", "eblast0", eblast0);
    canonicalOption("problem.disable_analytic", "disable_analytic", disable_analytic);
    canonicalOption("problem.driving.angular_momentum.duration", "driving_time", driving_time);
    canonicalOption("problem.driving.angular_momentum.rate", "driving_rate", driving_rate);
    canonicalOption("problem.driving.entropy.duration", "entropy_driving_time", entropy_driving_time);
    canonicalOption("problem.driving.entropy.rate", "entropy_driving_rate", entropy_driving_rate);
    canonicalOption("problem.dwd.v1309", "v1309", v1309);
    canonicalOption("problem.experiment", "experiment", experiment);
    canonicalOption("problem.input_file", "input_file", input_file);
    canonicalOption("problem.moving_star.velocity_x", "moving_star_xvelocity", moving_star_xvelocity);
    canonicalOption("problem.moving_star.velocity_y", "moving_star_yvelocity", moving_star_yvelocity);
    canonicalOption("problem.moving_star.velocity_z", "moving_star_zvelocity", moving_star_zvelocity);
    canonicalOption("problem.name", "problem", problem);
    canonicalOption("problem.rotating_star.amr", "rotating_star_amr", rotating_star_amr);
    canonicalOption("problem.rotating_star.center_x", "rotating_star_x", rotating_star_x);
    canonicalOption("problem.scf.density_floor", "scf_rho_floor", scf_rho_floor);
    canonicalOption("problem.scf.output_frequency", "scf_output_frequency", scf_output_frequency);
    canonicalOption("problem.sod.density_left", "sod_rhol", sod_rhol);
    canonicalOption("problem.sod.density_right", "sod_rhor", sod_rhor);
    canonicalOption("problem.sod.phi", "sod_phi", sod_phi);
    canonicalOption("problem.sod.pressure_left", "sod_pl", sod_pl);
    canonicalOption("problem.sod.pressure_right", "sod_pr", sod_pr);
    canonicalOption("problem.sod.theta", "sod_theta", sod_theta);
    canonicalOption("problem.solid_sphere.center_x", "solid_sphere_xcenter", solid_sphere_xcenter);
    canonicalOption("problem.solid_sphere.center_y", "solid_sphere_ycenter", solid_sphere_ycenter);
    canonicalOption("problem.solid_sphere.center_z", "solid_sphere_zcenter", solid_sphere_zcenter);
    canonicalOption("problem.solid_sphere.mass", "solid_sphere_mass", solid_sphere_mass);
    canonicalOption("problem.solid_sphere.minimum_density", "solid_sphere_rho_min", solid_sphere_rho_min);
    canonicalOption("problem.solid_sphere.radius", "solid_sphere_radius", solid_sphere_radius);
    canonicalOption("problem.star.alpha", "star_alpha", star_alpha);
    canonicalOption("problem.star.center_x", "star_xcenter", star_xcenter);
    canonicalOption("problem.star.center_y", "star_ycenter", star_ycenter);
    canonicalOption("problem.star.center_z", "star_zcenter", star_zcenter);
    canonicalOption("problem.star.central_density", "star_rho_center", star_rho_center);
    canonicalOption("problem.star.external_density", "star_rho_out", star_rho_out);
    canonicalOption("problem.star.external_gas_energy", "star_egas_out", star_egas_out);
    canonicalOption("problem.star.maximum_radius", "star_rmax", star_rmax);
    canonicalOption("problem.star.polytropic_index", "star_n", star_n);
    canonicalOption("problem.star.radial_step", "star_dr", star_dr);

    canonicalOption("radiation.cfl", "rad_cfl", radCfl);
    canonicalOption("radiation.enabled", "radiation", radiation);
    canonicalOption("radiation.energy_mode", "rad_energy_mode", radEnergyMode);
    canonicalOption("radiation.implicit", "rad_implicit", rad_implicit);
    canonicalOption("radiation.log_subcycles", "rad_log_subcycles", radLogSubcycles);
    canonicalOption("radiation.max_subcycles", "rad_max_subcycles", radMaxSubcycles);
    canonicalOption("radiation.opacity.constant", "rad_opacity", radOpacity);
    canonicalOption("radiation.reduced_light_speed_ratio", "rad_c_ratio", radCRatio);
    canonicalOption("radiation.source_theta", "rad_theta", radTheta);
    canonicalOption("radiation.subcycling", "rad_subcycling", radSubcycling);
    canonicalOption("radiation.test.amplitude", "rad_test_amplitude", radTestAmplitude);
    canonicalOption("radiation.test.background", "rad_test_background", radTestBackground);
    canonicalOption("radiation.test.extinction", "rad_test_chi", radTestChi);
    canonicalOption("radiation.test.luminosity", "rad_test_luminosity", radTestLuminosity);
    canonicalOption("radiation.test.reference", "rad_reference", radReference);
    canonicalOption("radiation.test.width", "rad_test_width", radTestWidth);
    canonicalOption("radiation.velocity_terms", "rad_velocity_terms", radVelocityTerms);

    canonicalOption("refinement.accretor_levels", "accretor_refine", accretor_refine);
    canonicalOption("refinement.core", "core_refine", core_refine);
    canonicalOption("refinement.density_floor", "refinement_floor", refinement_floor);
    canonicalOption("refinement.density_gradient", "grad_rho_refine", grad_rho_refine);
    canonicalOption("refinement.donor_levels", "donor_refine", donor_refine);

    canonicalOption("restart.filename", "restart_filename", restart_filename);

    canonicalOption("runtime.benchmark", "bench", bench);
    canonicalOption("runtime.config_file", "config_file", config_file);
    canonicalOption("runtime.disable_diagnostics", "disable_diagnostics", disable_diagnostics);
    canonicalOption("runtime.future_wait_time", "future_wait_time", future_wait_time);
    canonicalOption("runtime.print_times_per_timestep", "print_times_per_timestep", print_times_per_timestep);
    canonicalOption("runtime.stop_step", "stop_step", stop_step);
    canonicalOption("runtime.stop_time", "stop_time", stop_time);

    canonicalOption("timestep.fixed", "hard_dt", hard_dt);
    canonicalOption("timestep.max_change", "dt_max", dt_max);

    canonicalOption("units.centimeters", "code_to_cm", code_to_cm);
    canonicalOption("units.grams", "code_to_g", code_to_g);
    canonicalOption("units.seconds", "code_to_s", code_to_s);
#undef canonicalMultiOption
#undef canonicalOption

    radiation::addGreyOpacityOptions(canonicalOptions, radiationOpacity);
    displayedOptions.push_back(
        {"radiation.opacity.model", "", []() { return opts().radiationOpacity.model; }});
    displayedOptions.push_back(
        {"radiation.opacity.units", "", []() { return opts().radiationOpacity.units; }});
    displayedOptions.push_back(
        {"radiation.opacity.absorption", "", []() { return to_string(opts().radiationOpacity.absorption); }});
    displayedOptions.push_back(
        {"radiation.opacity.scattering", "", []() { return to_string(opts().radiationOpacity.scattering); }});
    displayedOptions.push_back({"radiation.opacity.transport_absorption", "",
        []() { return to_string(opts().radiationOpacity.transportAbsorption); }});

    po::options_description command_opts("All options");
    command_opts.add(canonicalOptions).add(legacyOptions);

    boost::program_options::variables_map vm;
    std::set<std::string> supplied;
    auto const optionWasExplicitlySupplied = [&supplied](OptionDisplay const& option) {
        return supplied.count(option.canonical) != 0 ||
            (!option.legacy.empty() && supplied.count(option.legacy) != 0);
    };
    auto const showEffectiveOptions = [&displayedOptions, &optionWasExplicitlySupplied]() {
        std::cout << "Effective canonical options ([specified] came from the command line or "
                     "configuration file):\n";
        for (auto const& option : displayedOptions) {
            std::cout << "  " << option.canonical << " = " << option.value() << " "
                      << (optionWasExplicitlySupplied(option) ? "[specified]" : "[default]")
                      << '\n';
        }
    };
    auto const rejectObsoleteCompositionOptions = [&supplied]() {
        bool valid = true;
        for (auto const* option : {"X", "Z"}) {
            if (supplied.count(option) != 0) {
                std::cerr << "ERROR: legacy option '" << option
                          << "' is obsolete and unsupported. Specify species composition with "
                             "'hydro.species.atomic_mass' and 'hydro.species.atomic_number'.\n";
                valid = false;
            }
        }
        return valid;
    };
    auto const commandLine = po::command_line_parser(argc, argv).options(command_opts).allow_unregistered().run();
    rememberSupplied(supplied, commandLine);
    po::store(commandLine, vm);
    po::notify(vm);
    reapplyCanonicalValues(vm, canonicalOptions, migrations);
    if (vm.count("help")) {
        warnLegacySpellings(supplied, migrations);
        if (!rejectObsoleteCompositionOptions()) {
            return false;
        }
        if (!checkCompatibilitySpellings(supplied, migrations)) {
            return false;
        }
        std::cout << canonicalOptions << "\n\n" << legacyOptions << "\n";
        return false;
    }
    if (!config_file.empty()) {
        std::ifstream cfg_fs{config_file};
        if (cfg_fs) {
            auto const config = po::parse_config_file(cfg_fs, command_opts);
            rememberSupplied(supplied, config);
            po::store(config, vm);
        } else {
            warnLegacySpellings(supplied, migrations);
            printf("Configuration file %s not found!\n", config_file.c_str());
            return false;
        }
    }
    warnLegacySpellings(supplied, migrations);
    if (!rejectObsoleteCompositionOptions()) {
        return false;
    }
    if (!checkCompatibilitySpellings(supplied, migrations)) {
        return false;
    }
    po::notify(vm);
    reapplyCanonicalValues(vm, canonicalOptions, migrations);
    if (opts().correct_am_hydro) {
        std::cerr << "ERROR: option 'correct_am_hydro' is obsolete and unsupported. "
                     "Octo-TIGER never implemented this correction.\n";
        return false;
    }
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
    if ((opts().modularHydro || opts().modularTransport) && !opts().restart_filename.empty()) {
        std::cerr << "ERROR: modular transport cannot read legacy restart checkpoints.\n";
        return false;
    }
    if (!opts().restart_filename.empty()) {
        // Preserve the already-resolved config/CLI value across checkpoint loading.
        auto const explicitSetting = [&supplied](
                                          char const* legacy, char const* canonical) { return supplied.count(legacy) != 0 || supplied.count(canonical) != 0; };
        auto const explicitRadImplicit = opts().rad_implicit;
        auto const explicitRadSubcycling = opts().radSubcycling;
        auto const explicitRadCRatio = opts().radCRatio;
        auto const explicitRadCfl = opts().radCfl;
        auto const explicitRadMaxSubcycles = opts().radMaxSubcycles;
        auto const explicitRadTheta = opts().radTheta;
        auto const explicitRadVelocityTerms = opts().radVelocityTerms;
        auto const explicitRadOpacity = opts().radOpacity;
        auto const explicitOpacityModel = radiationOpacity;
        auto const explicitRadEnergyMode = opts().radEnergyMode;
        auto const explicitRadLogSubcycles = opts().radLogSubcycles;
        FILE* fp = fopen(opts().restart_filename.c_str(), "rb");
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
        if (explicitSetting("rad_implicit", "radiation.implicit")) opts().rad_implicit = explicitRadImplicit;
        if (explicitSetting("rad_subcycling", "radiation.subcycling")) opts().radSubcycling = explicitRadSubcycling;
        if (explicitSetting("rad_c_ratio", "radiation.reduced_light_speed_ratio")) opts().radCRatio = explicitRadCRatio;
        if (explicitSetting("rad_cfl", "radiation.cfl")) opts().radCfl = explicitRadCfl;
        if (explicitSetting("rad_max_subcycles", "radiation.max_subcycles")) opts().radMaxSubcycles = explicitRadMaxSubcycles;
        if (explicitSetting("rad_theta", "radiation.source_theta")) opts().radTheta = explicitRadTheta;
        if (explicitSetting("rad_velocity_terms", "radiation.velocity_terms")) opts().radVelocityTerms = explicitRadVelocityTerms;
        if (explicitSetting("rad_opacity", "radiation.opacity.constant")) opts().radOpacity = explicitRadOpacity;
        radiation::restoreGreyOpacityOverrides(radiationOpacity, explicitOpacityModel, [&](char const* key) { return supplied.count(key) != 0; });

        if (explicitSetting("rad_energy_mode", "radiation.energy_mode")) opts().radEnergyMode = explicitRadEnergyMode;
        if (explicitSetting("rad_log_subcycles", "radiation.log_subcycles")) opts().radLogSubcycles = explicitRadLogSubcycles;
    }
    if (opts().dimensionCount < 1 || opts().dimensionCount > 3) {
        std::cerr << "ERROR: mesh.ndim must be 1, 2, or 3.\n";
        return false;
    }
    if (opts().dimensionCount < 3 && opts().gravity) {
        std::cerr << "ERROR: gravity is only supported with mesh.ndim=3; set gravity.enabled=off.\n";
        return false;
    }
    if (opts().dimensionCount < 3 && !opts().modularHydro && !opts().modularTransport) {
        std::cerr << "ERROR: mesh.ndim<3 requires runtime.modular.enabled=on; legacy topology is 3D.\n";
        return false;
    }
    try {
        radiationOpacity.validate(radOpacity);
        if (radiationOpacity.model == "grey" && (!(std::isfinite(code_to_g) && code_to_g > 0 && std::isfinite(code_to_cm) && code_to_cm > 0)))
            throw std::runtime_error("Grey opacity requires positive finite units.grams and units.centimeters");
        if (radiationOpacity.model != "legacy" &&
            (problem == RADIATION_STREAMING_WAVE || problem == RADIATION_STREAMING_FRONT || problem == RADIATION_GAUSSIAN_PULSE ||
                problem == RADIATION_EQUILIBRIUM_SPHERE))
            throw std::runtime_error("Prescribed radiation regression problems use radiation.test parameters; material "
                                     "opacity models require a coupled problem");
    } catch (std::exception const& error) {
        std::cerr << error.what() << "\n";
        return false;
    }
    // Validate after restart metadata is loaded as well as after CLI/config parsing.
    if (!(std::isfinite(radCRatio) && radCRatio > 0 && radCRatio <= 1) || !(std::isfinite(radCfl) && radCfl > 0 && radCfl <= .5) ||
        !(std::isfinite(radTheta) && radTheta >= .5 && radTheta <= 1) || !std::isfinite(radOpacity) || radMaxSubcycles < 1 ||
        (radEnergyMode != "thermal" && radEnergyMode != "absorption" && radEnergyMode != "equilibrium")) {
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
        std::cerr << "Either increase theta or recompile with a new theta minimum using the cmake "
                     "parameter OCTOTIGER_THETA_MINIMUM";
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
        opts().detected_intel_compiler = true;
    }
#endif
    const auto num_loc = hpx::find_all_localities().size();
    if (silo_num_groups > num_loc) {
        printf("Number of SILO file groups cannot be greater than number of localities. "
               "Setting silo_num_groupds to %li\n",
            num_loc);
        silo_num_groups = num_loc;
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
    showEffectiveOptions();
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
                  << "Either run a scenario without gravity, or remove OCTOTIGER_DISABLE_ILIST from "
                     "cmake and recompile!"
                  << std::endl;
        abort();
#endif
        if (opts().multipole_device_kernel_type == interaction_device_kernel_type::CUDA &&
            opts().multipole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "Due to a current implementation limitation in the load balancing, "
                      << " multipole cuda device kernels cannot be mixed with the respective "
                         "kokkos host kernel!"
                      << std::endl
                      << " Please choose a different host kernel "
                      << "(or move to kokkos device kernel with "
                         "--multipole_device_kernel_type=KOKKOS_CUDA)"
                      << std::endl;
            abort();
        }
        if (opts().monopole_device_kernel_type == interaction_device_kernel_type::CUDA &&
            opts().monopole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "Due to a current implementation limitation in the load balancing, "
                      << " monopole cuda device kernels cannot be mixed with the respective kokkos "
                         "host kernel!"
                      << std::endl
                      << " Please choose a different host kernel "
                      << "(or move to kokkos device kernel with "
                         "--monopole_device_kernel_type=KOKKOS_CUDA)"
                      << std::endl;
            abort();
        }
        if (opts().multipole_device_kernel_type == interaction_device_kernel_type::HIP &&
            opts().multipole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "Due to a current implementation limitation in the load balancing, "
                      << " multipole hip device kernels cannot be mixed with the respective kokkos "
                         "host kernel!"
                      << std::endl
                      << " Please choose a different host kernel "
                      << "(or move to kokkos device kernel with "
                         "--multipole_device_kernel_type=KOKKOS_HIP)"
                      << std::endl;
            abort();
        }
        if (opts().monopole_device_kernel_type == interaction_device_kernel_type::HIP &&
            opts().monopole_host_kernel_type == interaction_host_kernel_type::KOKKOS) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "Due to a current implementation limitation in the load balancing, "
                      << " monopole hip device kernels cannot be mixed with the respective kokkos host "
                         "kernel!"
                      << std::endl
                      << " Please choose a different host kernel "
                      << "(or move to kokkos device kernel with --monopole_device_kernel_type=KOKKOS_HIP)" << std::endl;
            abort();
        }
#ifndef OCTOTIGER_HAVE_VC
        if (opts().monopole_host_kernel_type == interaction_host_kernel_type::VC) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "Octotiger has been compiled without Vc support!"
                      << " Choose a different --monopole_host_kernel_type!" << std::endl;
            abort();
        }
        if (opts().multipole_host_kernel_type == interaction_host_kernel_type::VC) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "Octotiger has been compiled without Vc support! "
                      << "Choose a different --multipole_host_kernel_type!" << std::endl;
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
            std::cerr << "Octotiger has been compiled without Kokkos support! "
                      << " Choose a different --multipole_host_kernel_type!" << std::endl;
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
            std::cerr << "Octotiger has been compiled without CUDA support! "
                      << " Choose a different --multipole_device_kernel_type!" << std::endl;
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
            std::cerr << "The ideal gas plus radiation (ipr) eos is currently only supported with LEGACY "
                         "host kernel types of the hydro solver!"
                      << std::endl
                      << " Choose either a LEGACY for hydro host kernel type or use a different eos!" << std::endl;
            abort();
        }
        if (opts().hydro_device_kernel_type != OFF) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "The ideal gas plus radiation (ipr) eos is currently only supported on the host!" << std::endl
                      << " Choose OFF for hydro device kernel type or use a different eos!" << std::endl;
            abort();
        }
        if (opts().cdisc_detect) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "The ideal gas plus radiation (ipr) eos is currently not supported with "
                         "discontiniuty detection!"
                      << std::endl
                      << " Either set cdisc_detect to off or use a different eos!" << std::endl;
            abort();
        }
        if (opts().radiation) {
            std::cerr << std::endl << "ERROR: ";
            std::cerr << "The ideal gas plus radiation (ipr) eos is currently not supported "
                         "together with radiation field on!"
                      << std::endl
                      << " Either set radiation to off or use a different eos!" << std::endl;
            abort();
        }
    }
    if (opts().executors_per_gpu < 1 &&
        (opts().monopole_device_kernel_type != OFF || opts().multipole_device_kernel_type != OFF || opts().hydro_device_kernel_type != OFF)) {
        std::cerr << std::endl << "ERROR: ";
        std::cerr << "You have chosen an GPU kernel, however, you did not specify --executors_per_gpu > 0" << std::endl
                  << " Choose a different kernel or add at least one or more executors via "
                     "--executors_per_gpu=X"
                  << std::endl;
        abort();
    }
    if (opts().max_kernels_fused < 1 &&
        (opts().monopole_device_kernel_type != OFF || opts().multipole_device_kernel_type != OFF || opts().hydro_device_kernel_type != OFF)) {
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
        std::cerr << " - This build configuration only supports either the host or the device "
                     "kernel active, not both of them at the same time!"
                  << std::endl
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

std::vector<hpx::id_type> options::all_localities = {};
