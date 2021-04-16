---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Dominic
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Gregor
    affiliation: 2
  - name: Patrick
    affiliation: 2
  - name: Geoff C. Clayton
    affiliation: 2
  - name: Juhan Frank
    affiliation: 2
  - name: Kevin Huck
    affiliation: 2
  - name: Hartmut Kaiser
    affiliation: 2
  - name: Orsola De Marco
    affiliation: 2
  - name: Sagiv
    affiliation: 2
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution 2
   index: 2
date: 13 August 2017
bibliography: paper.bib
---

# Summary


## Implementation details

### HPX

[@Kaiser2020] [@daiss2019piz]

### Kokkos and CUDA integration

Using HPX and AMR, Octo-Tiger strives to use fine-grained tasks for parallelization. The compute-intensive kernels (like in the gravity solver) were thus only operating on a small subset of the grid each, making them an excellent target for SIMD vectorization with Vc as one CPU core could process a compute kernel in reasonable time. Multicore usage is achieved by each core executing a different HPX task (and thus a different compute kernel invocation). However, when porting Octo-Tiger to use GPUs, these small kernels were individually insufficient to properly utilize a GPU. To solve this, we use HPX and CUDA streams together to integrate GPU kernels into the HPX runtime as tasks. Similarly to how we achieve multicore usage on the CPU, this allows us to overlap the execution of GPU kernels with other GPU kernels (allowing for better GPU utilization), GPU/CPU data transfers, arbitrary CPU tasks and internode communication. 
Overall, we achieve good performance using this approach, even in large scale test runs with up to 5400 GPUs.

However, to support both CPU and GPU runs, we had to maintain both a CUDA and a Vc version for each of the compute-intensive kernels. Furthermore, using CUDA limits us to using NVIDIA GPUs. Currently, we are porting Octo-Tiger to Kokkos to solve these issues. We integrate HPX with Kokkos (using the hpx-kokkos library) similarly as before to maintain all advantages of our previous CUDA implementation. Additionally, using Kokkos enables us to use a wider range of accelerators (Intel, AMD, Nvidia) and, using the Kokkos-SIMD wrappers, still allows us to use explicit SIMD vectorization within the same kernel implementation in case of a CPU run. Early results after porting the gravity solver to Kokkos are promising, achieving comparable or better performance on both CPU and GPU.

HPX is integrated with APEX, an auto-tuning performance library for asynchronous tasking systems.  APEX has integrated support for CUDA and Kokkos, and is currently adding support for Kokkos auto-tuning, planned for the next Kokkos release.  We have successfuly used HPX counters and APEX to measure the Octo-Tiger simulation on leading HPC systems[0] with very low overheads.

# Statement of need

Octotiger is designed to solve fluid dynamics and self-gravity for astrophysical applications. While there are many astrophysical fluid dynamics codes, Octotiger specializes in resolving the fluid dynamics of the early mass transfer and merging processes. Only so we will be able to accurately model the light properties of dynamical transients, such as mergers and other strong binary interactions, where the properties of the photospheric layers dominate the lightcurve properties. Achieving sufficient resolution invariably means large problems that cannot be calculated within reasonable wall-clock times. To expedite the calculation, Octotiger uses a fast, asynchronous many-task parallelization technique, HPX [@Kaiser2020], that allows efficient scaling to tens of thousands of cores, utilizing CPU and GPU architectures simultaneously [@daiss2019piz]. Additionally, Octotiger makes some choices at the hydrodynamics and gravity solver level to achieve an accurate solution, including  a fully three-dimensional reconstruction at cell faces and machine precision conservation of energy in both gravity and hydro solvers. Also, Octotiger’s gravity solver conserves angular momentum to machine precision. While the inclusion of hydrodynamics worsens the conservation, the use of a frame rotating at the orbital frequency allows for superior overall conservation properties with a low diffusion leveluses 

# Methods
Octoiger uses a finite volume method on an octree based adaptive mesh refinement (AMR) mesh to solve the hydrodynamics. While most grid based codes use an iterative approach to solve Poisson's equation for the gravitational field, Octotiger uses a fast multipole method (FMM) to solve the gravitational field. The FMM has the advantage of conserving linear momenta to machine precision, and Octotiger's FMM has been specially designed to also conserve angular momentum to machine precision. This enables it to also conserve energy in the rotating frame which is important for maintaining equilibrium stellar structures. 

# Acknowledgements

This work was supported by National Science Foundation Award 1814967. The numerical work was carried out using the computational resources (QueenBee2) of the Louisiana Optical Network Initiative (LONI); Louisiana State University’s High Performance Computing (LSU HPC); resources of the National Energy Research Scientific Computing Center, a U.S. Departmentof Energy Office of Science User Facility operated underContract No. DE-AC02-05CH11231; and by Lilly Endowment,  Inc.,through its support for the Indiana University PervasiveTechnology Institute. This research  was  undertaken  with  the  assistance  of  resources  (Gadi) from  the  National  Computational  Infrastructure  (NCI  Australia), an NCRIS enabled capability supported by the Australian Government.


# References

[0] Diehl, Patrick, et al. "Performance Measurements within Asynchronous Task-based Runtime Systems: A Double White Dwarf Merger as an Application." arXiv preprint arXiv:2102.00223 (2021).
