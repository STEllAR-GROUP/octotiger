# Step 01 option compatibility inventory

This document records the compatibility contract introduced by Step 01. Boost.Program_options continues to parse flat strings: dots in canonical names are naming conventions, while `options` exposes hierarchical C++ views. The historical flat data members remain the serialized storage so HPX archive field order and Silo checkpoint keys are unchanged.

## Precedence and diagnostics

- Built-in defaults are established first by the historical `default_value` declarations.
- The configuration file is applied next.
- Command-line values win over configuration values because they are stored first in the Boost variables map; Boost does not replace an existing explicit value with a later store.
- Supplying both spellings of one setting anywhere across the command line and configuration file is an error.
- All supplied legacy spellings are reported together in one deprecation warning per process. The warning lists every legacy name used and its hierarchical replacement.
- Help lists canonical options before the clearly labeled legacy section.

`help` is a canonical flag and retains the conventional spelling. All other supported spellings accept the same value syntax on the command line and in configuration files. Multitoken vector behavior is retained for `atomic_mass` and `atomic_number`; the obsolete `X` and `Z` inputs are rejected with guidance to use those composition options.

## Complete migration and implementation inventory

| Legacy CLI/config | Canonical CLI/config | C++ type / storage | Built-in default | Serialization | Direct consumers outside parser/storage |
|---|---|---|---|---|---|
| `xscale` | `mesh.scale` | `Real` / `xscale` | `1.0` | HPX archive; Silo metadata | `frontend/init_methods.cpp:354`<br>`src/eos.cpp:285`<br>`src/node_server_actions_2.cpp:207,224`<br>`src/grid_scf.cpp:179`<br>`src/grid_amr.cpp:216,219,222`<br>`src/problem.cpp:238`<br>`src/grid.cpp:1766`<br>`src/radiation/rad_grid.cpp:755`<br>`src/io/silo_out.cpp:66,70,307,382`<br>`src/io/silo_in.cpp:127`<br>`src/test_problems/blast/sedov.cpp:41,45`<br>`src/test_problems/radiation/streamingFront.cpp:21`<br>`src/test_problems/radiation/streamingWave.cpp:21`<br>`src/test_problems/radiation/radiation.cpp:11,73`<br>`src/test_problems/radiation/gaussianPulse.cpp:44,96,182`<br>`src/test_problems/marshak/marshak.cpp:139`<br>`radiation_results/cpp/maintenance.cpp:63` |
| `dt_max` | `timestep.max_change` | `Real` / `dt_max` | `0.333333` | HPX archive | `src/grid.cpp:2014` |
| `cfl` | `hydro.cfl` | `Real` / `cfl` | `0.4` | HPX archive | `octotiger/util.hpp:37`<br>`src/node_server_actions_3.cpp:622,660` |
| `omega` | `mesh.omega_z` | `Real` / `omega` | `0.0` | HPX archive; Silo metadata | `frontend/init_methods.cpp:346`<br>`src/io/silo_in.cpp:107,149`<br>`src/test_problems/radiation/radiation.cpp:47` |
| `v1309` | `problem.dwd.v1309` | `bool` / `v1309` | `false` | HPX archive | `src/eos.cpp:96,193`<br>`src/grid_scf.cpp:289,408,539,565,620` |
| `idle_rates` | `output.idle_rates` | `bool` / `idle_rates` | `false` | HPX archive; Silo metadata | `src/grid.cpp:126,321`<br>`src/io/silo_out.cpp:95` |
| `eblast0` | `problem.blast.energy` | `Real` / `eblast0` | `1.0` | HPX archive | `src/test_problems/blast/sedov.cpp:67` |
| `rho_floor` | `hydro.density_floor` | `Real` / `rho_floor` | `0.0` | HPX archive | `src/grid.cpp:2453,2468,2469,2470` |
| `tau_floor` | `hydro.entropy_floor` | `Real` / `tau_floor` | `0.0` | HPX archive | `src/grid.cpp:2445,2446,2475,2476` |
| `sod_rhol` | `problem.sod.density_left` | `Real` / `sod_rhol` | `1.0` | HPX archive | `src/test_problems/sod/sod.cpp:79` |
| `sod_rhor` | `problem.sod.density_right` | `Real` / `sod_rhor` | `0.125` | HPX archive | `src/test_problems/sod/sod.cpp:80` |
| `sod_pl` | `problem.sod.pressure_left` | `Real` / `sod_pl` | `1.0` | HPX archive | `src/test_problems/sod/sod.cpp:81` |
| `sod_pr` | `problem.sod.pressure_right` | `Real` / `sod_pr` | `0.1` | HPX archive | `src/test_problems/sod/sod.cpp:82` |
| `sod_theta` | `problem.sod.theta` | `Real` / `sod_theta` | `0.0` | HPX archive | `src/test_problems/sod/sod.cpp:46` |
| `sod_phi` | `problem.sod.phi` | `Real` / `sod_phi` | `90.0` | HPX archive | `src/test_problems/sod/sod.cpp:49` |
| `sod_gamma` | `hydro.gamma` | `Real` / `sod_gamma` | `1.4` | HPX archive | `frontend/init_methods.cpp:372` |
| `solid_sphere_xcenter` | `problem.solid_sphere.center_x` | `Real` / `solid_sphere_xcenter` | `0.25` | HPX archive | `src/problem.cpp:373,410` |
| `solid_sphere_ycenter` | `problem.solid_sphere.center_y` | `Real` / `solid_sphere_ycenter` | `0.0` | HPX archive | `src/problem.cpp:374,411` |
| `solid_sphere_zcenter` | `problem.solid_sphere.center_z` | `Real` / `solid_sphere_zcenter` | `0.0` | HPX archive | `src/problem.cpp:375,412` |
| `solid_sphere_radius` | `problem.solid_sphere.radius` | `Real` / `solid_sphere_radius` | `1.0 / 3.0` | HPX archive | `src/problem.cpp:376,405` |
| `solid_sphere_mass` | `problem.solid_sphere.mass` | `Real` / `solid_sphere_mass` | `1.0` | HPX archive | `src/problem.cpp:377,406` |
| `solid_sphere_rho_min` | `problem.solid_sphere.minimum_density` | `Real` / `solid_sphere_rho_min` | `1.0e-12` | HPX archive | `src/problem.cpp:454` |
| `star_xcenter` | `problem.star.center_x` | `Real` / `star_xcenter` | `0.0` | HPX archive | `src/problem.cpp:477` |
| `star_ycenter` | `problem.star.center_y` | `Real` / `star_ycenter` | `0.0` | HPX archive | `src/problem.cpp:478` |
| `star_zcenter` | `problem.star.center_z` | `Real` / `star_zcenter` | `0.0` | HPX archive | `src/problem.cpp:479` |
| `star_n` | `problem.star.polytropic_index` | `Real` / `star_n` | `1.5` | HPX archive | `src/problem.cpp:484` |
| `star_rmax` | `problem.star.maximum_radius` | `Real` / `star_rmax` | `1.0 / 3.0` | HPX archive | `src/problem.cpp:482` |
| `star_dr` | `problem.star.radial_step` | `Real` / `star_dr` | `1.0 / (3.0 * 128.0)` | HPX archive | `src/problem.cpp:483` |
| `star_alpha` | `problem.star.alpha` | `Real` / `star_alpha` | `1.0 / (3.0 * 3.65375)` | HPX archive | `src/problem.cpp:481` |
| `star_rho_center` | `problem.star.central_density` | `Real` / `star_rho_center` | `1.0` | HPX archive | `src/problem.cpp:480` |
| `star_rho_out` | `problem.star.external_density` | `Real` / `star_rho_out` | `1.0e-10` | HPX archive | `src/problem.cpp:460,538` |
| `star_egas_out` | `problem.star.external_gas_energy` | `Real` / `star_egas_out` | `1.0e-10` | HPX archive | `src/problem.cpp:509,512,513`<br>`src/grid.cpp:1074` |
| `moving_star_xvelocity` | `problem.moving_star.velocity_x` | `Real` / `moving_star_xvelocity` | `1.0` | HPX archive | `src/problem.cpp:535,554` |
| `moving_star_yvelocity` | `problem.moving_star.velocity_y` | `Real` / `moving_star_yvelocity` | `1.0` | HPX archive | `src/problem.cpp:536,555` |
| `moving_star_zvelocity` | `problem.moving_star.velocity_z` | `Real` / `moving_star_zvelocity` | `1.0` | HPX archive | `src/problem.cpp:537,556` |
| `driving_rate` | `problem.driving.angular_momentum.rate` | `Real` / `driving_rate` | `0.0` | HPX archive | `src/grid.cpp:2211,2214,2215` |
| `driving_time` | `problem.driving.angular_momentum.duration` | `Real` / `driving_time` | `0.0` | HPX archive | `src/grid.cpp:2213` |
| `entropy_driving_time` | `problem.driving.entropy.duration` | `Real` / `entropy_driving_time` | `0.0` | HPX archive | `src/grid.cpp:2236` |
| `entropy_driving_rate` | `problem.driving.entropy.rate` | `Real` / `entropy_driving_rate` | `0.0` | HPX archive | `src/grid.cpp:2230,2241` |
| `future_wait_time` | `runtime.future_wait_time` | `integer` / `future_wait_time` | `-1` | HPX archive | `octotiger/future.hpp:73` |
| `silo_offset_x` | `output.silo.offset_x` | `integer` / `silo_offset_x` | `0` | HPX archive | `src/grid.cpp:267` |
| `silo_offset_y` | `output.silo.offset_y` | `integer` / `silo_offset_y` | `0` | HPX archive | `src/grid.cpp:268` |
| `silo_offset_z` | `output.silo.offset_z` | `integer` / `silo_offset_z` | `0` | HPX archive | `src/grid.cpp:269` |
| `amrbnd_order` | `mesh.amr.boundary_order` | `integer` / `amrbnd_order` | `1` | HPX archive | no direct `opts()` consumer |
| `scf_output_frequency` | `problem.scf.output_frequency` | `integer` / `scf_output_frequency` | `25` | HPX archive | `src/grid_scf.cpp:469` |
| `scf_rho_floor` | `problem.scf.density_floor` | `Real` / `scf_rho_floor` | `1.0e-12` | HPX archive | `src/grid_scf.cpp:315,720`<br>`src/grid.cpp:669,1596` |
| `silo_num_groups` | `output.silo.groups` | `integer` / `silo_num_groups` | `-1` | HPX archive; Silo metadata | `src/io/silo_out.cpp:565,582` |
| `core_refine` | `refinement.core` | `bool` / `core_refine` | `false` | HPX archive | `src/problem.cpp:246` |
| `grad_rho_refine` | `refinement.density_gradient` | `Real` / `grad_rho_refine` | `-1.0` | HPX archive | `src/problem.cpp:257` |
| `accretor_refine` | `refinement.accretor_levels` | `integer` / `accretor_refine` | `0` | HPX archive | `src/problem.cpp:255` |
| `extra_regrid` | `mesh.extra_initial_regrids` | `integer` / `extra_regrid` | `0` | HPX archive | `frontend/frontend-helper.cpp:206` |
| `donor_refine` | `refinement.donor_levels` | `integer` / `donor_refine` | `0` | HPX archive | `src/problem.cpp:252` |
| `ngrids` | `mesh.fixed_grid_count` | `integer` / `ngrids` | `-1` | HPX archive | `src/node_server_actions_3.cpp:515,516` |
| `refinement_floor` | `refinement.density_floor` | `Real` / `refinement_floor` | `1.0e-3` | HPX archive; Silo metadata | `src/node_server_actions_2.cpp:44`<br>`src/node_server_actions_3.cpp:514,517`<br>`src/problem.cpp:261,286`<br>`src/io/silo_out.cpp:379`<br>`src/io/silo_in.cpp:126` |
| `theta` | `gravity.opening_angle` | `Real` / `theta` | `0.5` | HPX archive | `src/grid_fmm.cpp:1116`<br>`src/monopole_interactions/monopole_kernel_interface.cpp:129,134,184`<br>`src/multipole_interactions/multipole_kernel_interface.cpp:100,151`<br>`src/monopole_interactions/legacy/p2m_cpu_kernel.cpp:34,692`<br>`src/monopole_interactions/legacy/p2p_cpu_kernel.cpp:27`<br>`src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:88`<br>`src/monopole_interactions/util/calculate_stencil.cpp:23`<br>`src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:54`<br>`src/multipole_interactions/legacy/multipole_cpu_kernel.cpp:27`<br>`src/multipole_interactions/util/calculate_stencil.cpp:27` |
| `eos` | `hydro.eos` | `eos_type` / `eos` | `IDEAL` | runtime only; Silo metadata | `src/eos.cpp:58,146,175,213,356,475,483,491,510,524`<br>`src/grid_scf.cpp:278,383,396,422,427,433,527,581,659,670,721,728,744`<br>`src/node_server_actions_3.cpp:321,334`<br>`src/problem.cpp:462,507`<br>`src/grid.cpp:197,245,432,439,453,644,650,711,718,732,1244,1716,1726,1730,1963,1966,2005,2248,2255,2445,2471,2537,2544,2558`<br>`src/physcon.cpp:67`<br>`src/roe.cpp:68,78,94,104`<br>`octotiger/radiation/cpu_kernel.hpp:183,228`<br>`octotiger/radiation/kernel_interface.hpp:167`<br>`src/radiation/rad_grid.cpp:220`<br>`src/io/silo_out.cpp:362`<br>`src/io/silo_in.cpp:104` |
| `ipr_nr_tol` | `hydro.ipr.newton_tolerance` | `Real` / `ipr_nr_tol` | `1.48e-08` | HPX archive | `src/grid.cpp:1967`<br>`src/physcon.cpp:40` |
| `ipr_nr_maxiter` | `hydro.ipr.newton_max_iterations` | `integer` / `ipr_nr_maxiter` | `50` | HPX archive | `src/grid.cpp:1967`<br>`src/physcon.cpp:38,47` |
| `ipr_test` | `hydro.ipr.test` | `bool` / `ipr_test` | `false` | HPX archive | `src/grid.cpp:1967` |
| `ipr_eint_floor` | `hydro.ipr.internal_energy_floor` | `Real` / `ipr_eint_floor` | `0.0` | HPX archive | `src/grid.cpp:440,651,719,1065,1729,1967,2256,2472,2545` |
| `hydro` | `hydro.enabled` | `bool` / `hydro` | `true` | HPX archive; Silo metadata | `frontend/init_methods.cpp:422`<br>`src/node_server_actions_2.cpp:59`<br>`src/node_server_actions_3.cpp:280,708,718`<br>`src/grid.cpp:136,266,1757,2354`<br>`src/radiation/rad_grid.cpp:396`<br>`src/io/silo_out.cpp:364`<br>`src/io/silo_in.cpp:106`<br>`src/test_problems/radiation/radiation.cpp:47` |
| `periodic` | `mesh.boundary.periodic` | `bool` / `periodic` | `false` | HPX archive | `src/node_server_actions_2.cpp:68,436,457,507,540`<br>`src/node_server.cpp:417`<br>`src/radiation/rad_grid.cpp:418,969`<br>`src/test_problems/radiation/radiation.cpp:50` |
| `radiation` | `radiation.enabled` | `bool` / `radiation` | `false` | HPX archive; Silo metadata | `frontend/init_methods.cpp:363`<br>`octotiger/grid.hpp:48,74,88`<br>`src/node_server_actions_2.cpp:211,217`<br>`src/grid_scf.cpp:449,669`<br>`src/node_server_actions_3.cpp:280,333,371,379,521,643,704,752`<br>`src/node_server.cpp:527,541`<br>`src/grid.cpp:108,120,166,314,402,1267,1815,1841,1900,1914`<br>`src/physcon.cpp:67,87,92`<br>`src/node_server_actions_1.cpp:154`<br>`src/io/silo_out.cpp:368`<br>`src/io/silo_in.cpp:110`<br>`src/test_problems/radiation/radiation.cpp:47` |
| `correct_am_hydro` | legacy-only; execution rejected when enabled | `bool` / `correct_am_hydro` | `false` | HPX archive | `src/grid.cpp:1976` |
| `correct_am_grav` | `gravity.angular_momentum_correction` | `bool` / `correct_am_grav` | `true` | HPX archive | `src/grid_fmm.cpp:1624` |
| `rewrite_silo` | `output.rewrite_silo` | `bool` / `rewrite_silo` | `false` | HPX archive | `src/node_server_actions_3.cpp:430,434` |
| `rad_implicit` | `radiation.implicit` | `bool` / `rad_implicit` | `true` | HPX archive; Silo metadata | `src/node_server_actions_3.cpp:708`<br>`src/radiation/rad_grid.cpp:267,401`<br>`src/io/silo_out.cpp:369`<br>`src/io/silo_in.cpp:112`<br>`src/test_problems/radiation/radiation.cpp:50` |
| `rad_subcycling` | `radiation.subcycling` | `bool` / `rad_subcycling` | `true` | HPX archive; Silo metadata | `src/node_server_actions_3.cpp:718`<br>`src/radiation/rad_grid.cpp:396`<br>`src/io/silo_out.cpp:370`<br>`src/io/silo_in.cpp:113` |
| `rad_c_ratio` | `radiation.reduced_light_speed_ratio` | `Real` / `rad_c_ratio` | `1.0` | HPX archive; Silo metadata | `src/radiation/rad_grid.cpp:236,458,518,526,637`<br>`src/io/silo_out.cpp:371`<br>`src/io/silo_in.cpp:114`<br>`src/test_problems/radiation/radiation.cpp:11,28` |
| `rad_cfl` | `radiation.cfl` | `Real` / `rad_cfl` | `0.4` | HPX archive; Silo metadata | `src/radiation/rad_grid.cpp:518`<br>`src/io/silo_out.cpp:372`<br>`src/io/silo_in.cpp:115` |
| `rad_max_subcycles` | `radiation.max_subcycles` | `integer` / `rad_max_subcycles` | `1024` | HPX archive; Silo metadata | `src/node_server_actions_3.cpp:719`<br>`src/radiation/rad_grid.cpp:396`<br>`src/io/silo_out.cpp:373`<br>`src/io/silo_in.cpp:116` |
| `rad_theta` | `radiation.source_theta` | `Real` / `rad_theta` | `1.0` | HPX archive; Silo metadata | `src/radiation/rad_grid.cpp:251,667,677`<br>`src/io/silo_out.cpp:374`<br>`src/io/silo_in.cpp:117` |
| `rad_velocity_terms` | `radiation.velocity_terms` | `bool` / `rad_velocity_terms` | `true` | HPX archive; Silo metadata | `src/radiation/rad_grid.cpp:564,662`<br>`src/io/silo_out.cpp:375`<br>`src/io/silo_in.cpp:118` |
| `rad_opacity` | `radiation.opacity.constant` | `Real` / `rad_opacity` | `-1.0` | HPX archive; Silo metadata | `src/radiation/rad_grid.cpp:247,279,281,358`<br>`src/io/silo_out.cpp:376`<br>`src/io/silo_in.cpp:119` |
| `rad_energy_mode` | `radiation.energy_mode` | `std::string` / `rad_energy_mode` | `"thermal"` | HPX archive; Silo metadata | `src/radiation/rad_grid.cpp:235,572,665`<br>`src/io/silo_out.cpp:377`<br>`src/io/silo_in.cpp:122` |
| `rad_log_subcycles` | `radiation.log_subcycles` | `bool` / `rad_log_subcycles` | `false` | HPX archive; Silo metadata | `src/radiation/rad_grid.cpp:397`<br>`src/io/silo_out.cpp:378`<br>`src/io/silo_in.cpp:124` |
| `gravity` | `gravity.enabled` | `bool` / `gravity` | `true` | HPX archive; Silo metadata | `frontend/init_methods.cpp:364,374,380`<br>`frontend/frontend-helper.cpp:212`<br>`src/node_server_actions_2.cpp:316`<br>`src/node_server_actions_3.cpp:289,585,739`<br>`src/node_server.cpp:267,345,548,594`<br>`src/grid.cpp:115,294,457,1426,1919,2202,2207,2298,2311,2322,2338,2572`<br>`src/physcon.cpp:67,87`<br>`src/node_server_actions_1.cpp:233,313`<br>`src/radiation/rad_grid.cpp:404,435`<br>`src/io/silo_out.cpp:363`<br>`src/io/silo_in.cpp:105`<br>`src/test_problems/radiation/radiation.cpp:47` |
| `bench` | `runtime.benchmark` | `bool` / `bench` | `false` | HPX archive | `src/node_server_actions_3.cpp:533,594` |
| `datadir` | `output.directory` | `std::string` / `data_dir` | `"./"` | HPX archive; Silo metadata | `src/node_server_actions_2.cpp:325,360`<br>`src/grid_scf.cpp:650`<br>`src/node_server_actions_3.cpp:320,372,485,536,597`<br>`src/grid.cpp:1766`<br>`src/io/silo_out.cpp:168,259,530`<br>`radiation_results/cpp/maintenance.cpp:63` |
| `output` | `output.filename` | `std::string` / `output_filename` | `""` | HPX archive | `src/node_server_actions_3.cpp:343,346` |
| `odt` | `output.interval` | `Real` / `output_dt` | `1.0 / 100.0` | HPX archive; Silo metadata | `src/node_server_actions_3.cpp:356`<br>`src/io/silo_out.cpp:366`<br>`src/io/silo_in.cpp:108` |
| `dual_energy_sw1` | `hydro.dual_energy.switch1` | `Real` / `dual_energy_sw1` | `0.001` | HPX archive | `octotiger/roe.hpp:19`<br>`src/grid.cpp:1733,1739,1969`<br>`octotiger/radiation/kernel_interface.hpp:168`<br>`src/radiation/rad_grid.cpp:222` |
| `dual_energy_sw2` | `hydro.dual_energy.switch2` | `Real` / `dual_energy_sw2` | `0.1` | HPX archive | `octotiger/roe.hpp:20`<br>`src/grid.cpp:1969`<br>`octotiger/radiation/kernel_interface.hpp:168` |
| `hard_dt` | `timestep.fixed` | `Real` / `hard_dt` | `-1` | HPX archive | `src/node_server_actions_3.cpp:723` |
| `experiment` | `problem.experiment` | `int` / `experiment` | `0` | HPX archive | `src/grid.cpp:1975` |
| `unigrid` | `mesh.unigrid` | `bool` / `unigrid` | `false` | HPX archive | `src/problem.cpp:325`<br>`src/test_problems/radiation/radiation.cpp:47` |
| `inflow_bc` | `mesh.boundary.inflow` | `bool` / `inflow_bc` | `false` | HPX archive | `src/grid.cpp:2144` |
| `reflect_bc` | `mesh.boundary.reflecting` | `bool` / `reflect_bc` | `false` | HPX archive | `src/grid.cpp:2118,2142` |
| `cdisc_detect` | `hydro.contact_discontinuity_detection` | `bool` / `cdisc_detect` | `true` | HPX archive | `src/grid.cpp:1979` |
| `disable_output` | `output.disabled` | `bool` / `disable_output` | `false` | HPX archive; Silo metadata | `octotiger/util.hpp:47`<br>`src/grid_scf.cpp:470`<br>`src/node_server_actions_3.cpp:282,313,428,483,535,578,588,594`<br>`src/io/silo_out.cpp:525` |
| `rad_reference` | `radiation.test.reference` | `std::string` / `radReference` | `"gaussian_pulse.bin"` | HPX archive | `src/test_problems/radiation/radiation.cpp:21` |
| `rad_test_chi` | `radiation.test.extinction` | `Real` / `radTestChi` | `5` | runtime only | `src/test_problems/radiation/radiation.cpp:12` |
| `rad_test_width` | `radiation.test.width` | `Real` / `radTestWidth` | `.2` | runtime only | `src/test_problems/radiation/radiation.cpp:12` |
| `rad_test_background` | `radiation.test.background` | `Real` / `radTestBackground` | `1` | HPX archive | `src/test_problems/radiation/radiation.cpp:13` |
| `rad_test_amplitude` | `radiation.test.amplitude` | `Real` / `radTestAmplitude` | `.01` | runtime only | `src/test_problems/radiation/radiation.cpp:13` |
| `rad_test_luminosity` | `radiation.test.luminosity` | `Real` / `radTestLuminosity` | `.01` | runtime only | `src/test_problems/radiation/radiation.cpp:14` |
| `disable_analytic` | `problem.disable_analytic` | `bool` / `disable_analytic` | `false` | HPX archive | `src/node_server_actions_3.cpp:286,583` |
| `disable_diagnostics` | `runtime.disable_diagnostics` | `bool` / `disable_diagnostics` | `false` | HPX archive | `src/node_server_actions_2.cpp:307,323`<br>`src/node_server_actions_3.cpp:454`<br>`src/grid.cpp:385` |
| `problem` | `problem.name` | `problem_type` / `problem` | `NONE` | runtime only; Silo metadata | `frontend/init_methods.cpp:362,367,371,378,390,394,399,404,409,415,421`<br>`frontend/frontend-helper.cpp:217,220`<br>`test_problems/radiation.hpp:7,12`<br>`octotiger/diagnostics.hpp:118,135`<br>`src/node_server_actions_2.cpp:59,202,260,312`<br>`src/node_server_actions_3.cpp:334,403,451,556`<br>`src/node_server.cpp:76`<br>`src/grid.cpp:183,392,515,736,993,1006,1697,1724,1827,2068,2080,2147`<br>`src/physcon.cpp:106,135`<br>`octotiger/radiation/cpu_kernel.hpp:245`<br>`octotiger/radiation/kernel_interface.hpp:167`<br>`octotiger/radiation/opacities.hpp:27,29,31,33,49,51,53,55,68,79`<br>`src/radiation/rad_grid.cpp:100,103,123,126,237,449,739,753`<br>`src/io/silo_out.cpp:170,260,261,367`<br>`src/io/silo_in.cpp:109`<br>`src/test_problems/radiation/radiation.cpp:30,49,52` |
| `restart_filename` | `restart.filename` | `std::string` / `restart_filename` | `""` | HPX archive | `frontend/frontend-helper.cpp:183,184,185,220`<br>`src/grid_scf.cpp:683`<br>`src/node_server_actions_3.cpp:430`<br>`src/node_server.cpp:521`<br>`src/node_server_actions_1.cpp:145` |
| `stop_time` | `runtime.stop_time` | `Real` / `stop_time` | `std::numeric_limits<Real>::max()` | HPX archive | `src/node_server_actions_3.cpp:394,724,725`<br>`src/test_problems/radiation/radiation.cpp:11` |
| `stop_step` | `runtime.stop_step` | `integer` / `stop_step` | `std::numeric_limits<integer>::max() - 1` | HPX archive | `frontend/frontend-helper.cpp:212`<br>`src/node_server_actions_3.cpp:293,309,350,396,425,448` |
| `min_level` | `mesh.level.minimum` | `integer` / `min_level` | `1` | HPX archive | `frontend/init_methods.cpp:355` |
| `max_level` | `mesh.level.maximum` | `integer` / `max_level` | `1` | HPX archive | `frontend/init_methods.cpp:356`<br>`frontend/frontend-helper.cpp:194,202`<br>`src/node_server_actions_2.cpp:223`<br>`src/grid_scf.cpp:179`<br>`src/node_location.cpp:41`<br>`src/test_problems/blast/sedov.cpp:41`<br>`src/test_problems/radiation/radiation.cpp:54,73`<br>`src/test_problems/radiation/gaussianPulse.cpp:45,97,186` |
| `amr_boundary_kernel_type` | `execution.kernel.amr_boundary` | `amr_boundary_type` / `amr_boundary_kernel_type` | `AMR_OPTIMIZED` | runtime only | `src/node_server.cpp:381` |
| `multipole_host_kernel_type` | `execution.kernel.multipole.host` | `interaction_host_kernel_type` / `multipole_host_kernel_type` | `KOKKOS` | HPX archive | `src/multipole_interactions/multipole_kernel_interface.cpp:78`<br>`src/multipole_interactions/legacy/multipole_interaction_interface.cpp:79` |
| `multipole_device_kernel_type` | `execution.kernel.multipole.device` | `interaction_device_kernel_type` / `multipole_device_kernel_type` | `OFF` | HPX archive | `src/multipole_interactions/multipole_kernel_interface.cpp:79` |
| `monopole_host_kernel_type` | `execution.kernel.monopole.host` | `interaction_host_kernel_type` / `monopole_host_kernel_type` | `KOKKOS` | HPX archive | `src/monopole_interactions/monopole_kernel_interface.cpp:102`<br>`src/monopole_interactions/legacy/monopole_interaction_interface.cpp:77`<br>`src/monopole_interactions/legacy/p2m_interaction_interface.cpp:125,146` |
| `monopole_device_kernel_type` | `execution.kernel.monopole.device` | `interaction_device_kernel_type` / `monopole_device_kernel_type` | `OFF` | HPX archive | `src/monopole_interactions/monopole_kernel_interface.cpp:103` |
| `hydro_host_kernel_type` | `execution.kernel.hydro.host` | `interaction_host_kernel_type` / `hydro_host_kernel_type` | `KOKKOS` | HPX archive | `src/grid.cpp:1987`<br>`src/unitiger/hydro_impl/hydro_kernel_interface.cpp:93` |
| `hydro_device_kernel_type` | `execution.kernel.hydro.device` | `interaction_device_kernel_type` / `hydro_device_kernel_type` | `OFF` | HPX archive | `src/grid.cpp:1988`<br>`src/unitiger/hydro_impl/hydro_kernel_interface.cpp:94` |
| `number_gpus` | `execution.gpu.count` | `size_t` / `number_gpus` | `size_t(0)` | HPX archive | `frontend/init_methods.cpp:152,153,157,158,243,256,264,299,305,323,330`<br>`octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:84,103`<br>`octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:66`<br>`octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:58`<br>`src/monopole_interactions/monopole_kernel_interface.cpp:79,82,85,114`<br>`src/multipole_interactions/multipole_kernel_interface.cpp:89`<br>`src/cuda_util/cuda_scheduler.cpp:32`<br>`src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:99`<br>`src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:65`<br>`src/unitiger/hydro_impl/hydro_kernel_interface.cpp:64,67,70,112,137,158`<br>`src/unitiger/hydro_impl/hydro_cuda_interface.cpp:79,85` |
| `executors_per_gpu` | `execution.gpu.executors_per_gpu` | `size_t` / `executors_per_gpu` | `size_t(0)` | HPX archive | `frontend/init_methods.cpp:83,246,259,267,301,307,325,332`<br>`src/monopole_interactions/monopole_kernel_interface.cpp:76`<br>`src/unitiger/hydro_impl/hydro_kernel_interface.cpp:61` |
| `max_gpu_executor_queue_length` | `execution.gpu.max_queue_length` | `size_t` / `max_gpu_executor_queue_length` | `size_t(5)` | HPX archive | `src/node_server.cpp:396`<br>`src/grid.cpp:1989`<br>`src/monopole_interactions/monopole_kernel_interface.cpp:119`<br>`src/multipole_interactions/multipole_kernel_interface.cpp:94`<br>`src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:103`<br>`src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:68` |
| `polling-threads` | `execution.polling_threads` | `int` / `polling_threads` | `0` | HPX archive | `frontend/init_methods.cpp:100,109,200,211` |
| `max_kernels_fused` | `execution.max_kernels_fused` | `size_t` / `max_kernels_fused` | `size_t(1)` | HPX archive | `octotiger/aggregation_util.hpp:76`<br>`octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:331,953,1369`<br>`octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:103,300,571,722,842,936,1000,1041,1106,1257,1410`<br>`src/monopole_interactions/monopole_kernel_interface.cpp:70`<br>`src/unitiger/hydro_impl/hydro_kernel_interface.cpp:55`<br>`src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:27,72`<br>`src/unitiger/hydro_impl/hydro_cuda_interface.cpp:75,150` |
| `root_node_on_device` | `execution.root_node_on_device` | `bool` / `root_node_on_device` | `true` | HPX archive | `src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:71,72` |
| `optimize_local_communication` | `execution.optimize_local_communication` | `bool` / `optimize_local_communication` | `true` | HPX archive | `src/node_server_actions_3.cpp:739`<br>`src/node_server.cpp:196,197,431,538`<br>`src/node_server_actions_1.cpp:225`<br>`src/radiation/rad_grid.cpp:404,435` |
| `print_times_per_timestep` | `runtime.print_times_per_timestep` | `bool` / `print_times_per_timestep` | `false` | HPX archive | `src/node_server_actions_3.cpp:812` |
| `input_file` | `problem.input_file` | `std::string` / `input_file` | `""` | HPX archive | no direct `opts()` consumer |
| `config_file` | `runtime.config_file` | `std::string` / `config_file` | `""` | HPX archive | no direct `opts()` consumer |
| `n_species` | `hydro.species.count` | `integer` / `n_species` | `5` | HPX archive; Silo metadata | `frontend/init_methods.cpp:343,368`<br>`octotiger/physcon.hpp:20`<br>`src/grid_scf.cpp:398`<br>`src/grid.cpp:74,89,175,188,238,443,722,748,1069,1240,1336,1342,1570,1587,1593,1667,1683,1981,2456,2462,2466,2483,2512,2548`<br>`src/physcon.cpp:276`<br>`octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:1402`<br>`src/radiation/rad_grid.cpp:380`<br>`src/io/silo_out.cpp:361,389`<br>`src/io/silo_in.cpp:103,128,129,130,131`<br>`src/test_problems/radiation/streamingFront.cpp:36`<br>`src/test_problems/radiation/streamingWave.cpp:37`<br>`src/test_problems/rotating_star/rotating_star.cpp:123`<br>`src/unitiger/hydro_impl/hydro_kernel_interface.cpp:119,181,214`<br>`src/unitiger/hydro_impl/hydro_cuda_interface.cpp:148,249,265` |
| `atomic_mass` | `hydro.species.atomic_mass` | `std::vector<Real>` / `atomic_mass` | `none` | HPX archive; Silo metadata | `frontend/init_methods.cpp:345`<br>`src/grid_scf.cpp:399,400,528,529`<br>`src/problem.cpp:509`<br>`src/grid.cpp:1684`<br>`src/physcon.cpp:278`<br>`src/io/silo_out.cpp:392`<br>`src/io/silo_in.cpp:129,133`<br>`src/test_problems/radiation/equilibriumSphere.cpp:102` |
| `atomic_number` | `hydro.species.atomic_number` | `std::vector<Real>` / `atomic_number` | `none` | HPX archive; Silo metadata | `frontend/init_methods.cpp:345`<br>`src/grid_scf.cpp:400,528,529`<br>`src/problem.cpp:509`<br>`src/grid.cpp:1684`<br>`src/physcon.cpp:278`<br>`src/io/silo_out.cpp:393`<br>`src/io/silo_in.cpp:128,132`<br>`src/test_problems/radiation/equilibriumSphere.cpp:101` |
| `X` | obsolete; no canonical alias (use `hydro.species.atomic_mass` and `hydro.species.atomic_number`) | `std::vector<Real>` / `X` | `none` | HPX archive; Silo metadata | `src/grid.cpp:1686`<br>`src/physcon.cpp:279`<br>`src/io/silo_out.cpp:390`<br>`src/io/silo_in.cpp:130,134` |
| `Z` | obsolete; no canonical alias (use `hydro.species.atomic_mass` and `hydro.species.atomic_number`) | `std::vector<Real>` / `Z` | `none` | HPX archive; Silo metadata | `src/grid.cpp:1687`<br>`src/physcon.cpp:280`<br>`src/io/silo_out.cpp:391`<br>`src/io/silo_in.cpp:131,135` |
| `code_to_g` | `units.grams` | `Real` / `code_to_g` | `1` | HPX archive; Silo metadata | `src/grid.cpp:187,212,1685,1710,1715,1728,1729`<br>`src/physcon.cpp:88,93,99,107,112,136,140`<br>`src/radiation/rad_grid.cpp:102,125`<br>`src/io/silo_out.cpp:358,503`<br>`src/io/silo_in.cpp:100,150` |
| `code_to_cm` | `units.centimeters` | `Real` / `code_to_cm` | `1` | HPX archive; Silo metadata | `src/grid.cpp:184,210,1710,1715,1728,1729`<br>`src/physcon.cpp:89,94,100,109,111,138,142`<br>`src/radiation/rad_grid.cpp:102,103,125,126`<br>`src/io/silo_out.cpp:72,307,360`<br>`src/io/silo_in.cpp:102,150` |
| `code_to_s` | `units.seconds` | `Real` / `code_to_s` | `1` | HPX archive; Silo metadata | `src/grid.cpp:155,186,211,1710,1728,1729`<br>`src/physcon.cpp:90,95,108,113,137,141`<br>`src/radiation/rad_grid.cpp:102,103,125,126`<br>`src/io/silo_out.cpp:101,359,365`<br>`src/io/silo_in.cpp:101,107,150,163`<br>`radiation_results/cpp/run.cpp:102` |
| `rotating_star_amr` | `problem.rotating_star.amr` | `bool` / `rotating_star_amr` | `false` | HPX archive | `src/problem.cpp:288` |
| `rotating_star_x` | `problem.rotating_star.center_x` | `Real` / `rotating_star_x` | `0.0` | HPX archive | `src/test_problems/rotating_star/rotating_star.cpp:98` |

## Compatibility note

Existing command lines and configuration files remain accepted with unchanged values and defaults, except for the explicitly unsupported `correct_am_hydro=true`, `X`, and `Z` inputs. Other legacy names remain accepted and produce one consolidated migration warning. Checkpoint/restart compatibility is retained by keeping the historical flat fields and archive order; dotted names are parser aliases and hierarchical views only. Explicit radiation controls from either spelling continue to override checkpoint metadata.

## Review status

Every supported parser spelling has one migration entry; `correct_am_hydro`, `X`, and `Z` are documented as unsupported instead. No option was split into a new physical setting. In particular, the single existing `rad_opacity` remains one constant-gray-opacity control named `radiation.opacity.constant`; absorption/scattering controls were not invented because the snapshot does not yet define those separate semantics. `sod_gamma` maps to `hydro.gamma` because its parser description and sole initialization consumer establish it as the gas ratio of specific heats.



Step 04 adds canonical-only `radiation.opacity.model`, `.units`, `.absorption`,
`.scattering`, and `.transport_absorption`; no historical alias is removed.
See [grey opacity compatibility](grey-opacity-step-04.md).
