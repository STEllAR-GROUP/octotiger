---
title: 'Octo-Tiger: a full 3D adaptive multigrid code astrophysics application accelerated using the asynchronous many-task runtime HPX '
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Dominic Marcello
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" 
  - name: Gregor Dai\ss
    affiliation: 6
  - name: Patrick Diehl
    orcid: 0000-0003-3922-8419
    affiliation: 2
  - name: Sagiv Shiber
    orcid: 0000-0001-6107-0887
    affiliation: 1
  - name: Parsa Amini
    affiliation: 2
  - name: Geoff C. Clayton
    orcid: 0000-0002-0141-7436
    affiliation: 1
  - name: Juhan Frank
    affiliation: 1
  - name: Kevin Huck
    orcid: 0000-0001-7064-8417
    affiliation: 3
  - name: Hartmut Kaiser
    orcid: 0000-0002-8712-2806
    affiliation: 2
  - name: Orsola De Marco
    affiliation: "4, 5"
  - name: Bryce Adelstein Lelbach
    orcid: 0000-0002-7995-5226
    affiliation: 7
affiliations:
 - name: Department of Physics & Astronomy, Louisiana State University, Baton Rouge, LA, United States of America
   index: 1
 - name: Center of Computation and Technology, Louisiana State University, Baton Rouge, LA, United States of America
   index: 2
 - name: OACISS, University of Oregon, Eugene, OR, United States of America
   index: 3
 - name: Department of Physics and Astronomy, Macquarie University, Sydney, NSW 2109, Australia
   index: 4
 - name: Astronomy, Astrophysics and Astrophotonics Research Centre, Macquarie University, Sydney, NSW 2109, Australia
   index: 5
 - name: IPVS, University of Stuttgart, Stuttgart, Germany
   index: 6
 - name: NVIDIA
   index: 7
date: 13 August 2017
bibliography: paper.bib
---

# Summary

Octo-Tiger uses finite volume methods to advance the fluid and the fast multi-pole method (FMM) to compute the gravitational field [@10.1093/mnras/stab937]. The evolution variables are evolved in a Cartesian octree AMR mesh.  The features of Octo-Tiger are ideally suited for evolving interacting binary systems which begin their evolution in near equilibrium, such as a binary system at the beginning of mass transfer. The AMR mesh rotates with the initial orbital frequency, reducing numerical viscosity. The hydrodynamics module employs a fully three-dimensional reconstruction, with quadrature points for the flux at cell faces, edges, and vertices, enabling a better representation of (near) spherical structures on the Cartesian mesh. The FMM employed by Octo-Tiger, conserves both linear and angular momenta to machine precision [@Marcello2017], enabling it to couple to the hydrodynamics solver in a way that conserves energy and linear momenta to machine precision,. Machine precision conservation of energy prevents equilibrium stellar structures from dissipating due to numerical viscosity. Octo-Tiger has been used to study mergers as progenitors of the R Coronae Borealis stars [@Lauer2019]; [@Staff2018]. It has also been used to study the possibility that Betelgeuse may be the product of a merger [@Betelgeuse]

## Implementation details

### The C++ standard library for parallelism and concurrency (HPX)

In addition to the astrophysics aspect, Octo-Tiger explores the usage of asynchronous many-task systems (AMT) as an alternative to the common MPI+X approach. Here, the C++ standard library for Concurrency and Parallelism (HPX) [@Kaiser2020] is used. A comparative review of HPX with various other AMTs is available in [@thoman2018taxonomy]. Some notable AMT solutions with a focus on distributed computing are: Uintah [@germain2000uintah], Chapel [@chamberlain2007parallel], Charm++ [@kale1993charm], Kokkos [@edwards2014kokkos], Legion [@bauer2012legion], and PaRSEC [@bosilca2013parsec]. Note that we only refer to distributed memory solutions, since we intend to do large scale runs to solve Octo-Tiger's multi physics on a fine resolution. According to this review, the major showpiece of HPX compared to the mentioned distributed AMTs is its future-proof C++ standard-conforming API. Charm++ is another AMT utilized in astrophysics simulations, e.g.\ ChaNGa [@jetley2008massively] or Enzo-P [@10.5555/2462077.2462081].  

HPX provides four advantages for Octo-Tiger: (i) fine-grained, (II) task-based parallelism using lightweight user-space threads, (iii) the use of C++ futures to encapsulate both local and remote work, and (iv) an active global address space (AGAS) [@amini2019assessing,@kaiser2014hpx], whereby global and local objects are accessible using the same API. For more details, we refer to [@10.1093/mnras/stab937].

HPX is integrated with APEX, an auto-tuning performance library for asynchronous tasking systems.  APEX has integrated support for CUDA and Kokkos and is currently adding support for Kokkos auto-tuning, planned for the next Kokkos release.  We have successfully used HPX counters and APEX to measure the Octo-Tiger simulation on leading HPC systems [@diehl2021performance] with very low overheads.

### Kokkos and CUDA integration

Using HPX and AMR, Octo-Tiger strives to use fine-grained tasks for parallelization. The compute-intensive kernels (like in the gravity solver) were only operating on a small subset of the grid, making them an excellent target for SIMD vectorization with Vc as one CPU core could process a compute kernel in a reasonable time. Multicore usage is achieved by each core executing a different HPX task (and thus a different compute kernel invocation). However, when porting Octo-Tiger to use GPUs, these small kernels were individually insufficient to utilize GPUs properly. We use HPX and CUDA streams to integrate GPU kernels into the HPX runtime as tasks to solve this. In a similar manner to how we achieve multicore usage on the CPU, this allows us to overlap the execution of GPU kernels with other GPU kernels (allowing for better GPU utilization), GPU/CPU data transfers, arbitrary CPU tasks, and internode communication. 
Overall, we achieve good performance using this approach, even in large-scale test runs with up to 5400 GPUs [@daiss2019piz].

However, to support both CPU and GPU runs, we maintain a CUDA and a Vc version for each of the compute-intensive kernels. Furthermore, using CUDA limits us to using NVIDIA GPUs. Currently, we are porting Octo-Tiger to Kokkos to solve these issues. We integrate HPX with Kokkos (using the HPX-Kokkos library) similarly as before to maintain all advantages of our previous CUDA implementation. Additionally, using Kokkos enables us to use a wider range of accelerators (Intel, AMD, Nvidia). Using the Kokkos-SIMD wrappers still allows us to use explicit SIMD vectorization within the same kernel implementation in a CPU run. Early results after porting the gravity solver to Kokkos are promising, achieving comparable or better performance on both CPU and GPU.

# Statement of need

Octo-Tiger is designed to solve fluid dynamics and self-gravity for astrophysical applications. 
Octo-Tiger specializes in resolving  the fluid dynamics of early stellar mass transfer and merging processes to accurately model dynamical transients, such as mergers and other strong binary interactions, where the properties of the photospheric layers dominate the lightcurve. Until recently, transients due to stellar interactions were rarely
discovered, new large telescopes will be producing observations for which we have, at this time, no satisfactory lightcurve model. Only a handful of 3D fluid dynamics codes come close to tackling this challenging problem because achieving sufficient resolution invariably means large problems that cannot be calculated within reasonable wall-clock times. To expedite the calculation, Octo-Tiger uniquely uses a fast, asynchronous many-task parallelization technique, HPX [@Kaiser2020], that allows efficient scaling to tens of thousands of cores, utilizing CPU and GPU architectures simultaneously [@daiss2019piz]. Additionally, Octo-Tiger makes some choices at the hydrodynamics and gravity solver level to achieve an accurate solution, including a fully three-dimensional reconstruction at cell faces and machine precision conservation of energy in both gravity and hydro solvers. Also, Octo-Tiger’s gravity solver conserves angular momentum to machine precision. While the inclusion of hydrodynamics worsens the conservation, the use of a frame rotating at the orbital frequency allows for superior overall conservation properties with low diffusion.


# Acknowledgments

This work was supported by National Science Foundation Award 1814967. The numerical work was carried out using the computational resources (QueenBee2) of the Louisiana Optical Network Initiative (LONI); Louisiana State University's High-Performance Computing (LSU HPC); resources of the National Energy Research Scientific Computing Center, a U.S. Department of Energy Office of Science User Facility operated under contract No. DE-AC02-05CH11231; and by Lilly Endowment,  Inc., through its support for the Indiana University PervasiveTechnology Institute. This research was undertaken with the assistance of resources  (Gadi) from the  National  Computational  Infrastructure  (NCI  Australia), an NCRIS enabled capability supported by the Australian Government.

For an updated list of previous and current funding, we refer to the corresponding [Octo-Tiger website](https://github.com/STEllAR-GROUP/octotiger#funding).


# References
