# SPDX-License-Identifier: BSL-1.0
import os
import subprocess

from spack.package import *
from spack.pkg.octotiger_hpx2rc.hpx import compiler_args, compiler_environment


class Octotiger(CMakePackage, CudaPackage, ROCmPackage):
    homepage = "https://github.com/STEllAR-GROUP/octotiger"
    git = "https://github.com/STEllAR-GROUP/octotiger.git"
    # dev-build uses the Jenkins checkout, whose exact revision is archived.
    version("develop", branch="hpx2rc")
    generator("ninja")
    variant("kokkos", default=True, description="Kokkos kernels")
    variant("sycl", default=False, description="SYCL kernels")
    variant("griddim", default="8", values=("8", "16"), multi=False, description="Tested subgrid size")
    variant("theta_minimum", default="0.34", values=("0.5", "0.34", "0.26", "0.16"), multi=False, description="Minimum theta")
    variant("kokkos_hpx_kernels", default=False, description="Use HPX execution space for host kernels")
    variant("monopole_host_tasks", default="1", values=("1", "4", "16"), multi=False, description="Monopole tasks")
    variant("multipole_host_tasks", default="1", values=("1", "4", "16", "64"), multi=False, description="Multipole tasks")
    variant("hydro_host_tasks", default="1", values=lambda value: str(value).isdigit(), multi=False, description="Hydro tasks")
    variant("simd_library", default="KOKKOS", values=("KOKKOS", "STD"), multi=False, description="SIMD library")
    variant("simd_extension", default="DISCOVER", values=("DISCOVER", "SCALAR", "AVX", "AVX512", "NEON", "SVE"), multi=False, description="SIMD extension")
    variant("fast_fp_contract", default=False, description="Aggressive floating point contraction")
    depends_on("cmake@3.22:", type="build")
    depends_on("git", type="build")
    depends_on("python@3:", type="build")
    depends_on("vc@1.4.1")
    depends_on("boost@1.74: cxxstd=2a +program_options +filesystem +system")
    depends_on("hdf5 +threadsafe +szip +hl ~mpi")
    depends_on("silo@4.10.2-bsd:4.11-bsd ~mpi")
    depends_on("silo ~fortran", when="%clang")
    depends_on("hpx@2.0.0rc1 cxxstd=20")
    depends_on("cppuddle@0.4.0 cxxstd=20 +hpx")
    depends_on("cppuddle +kokkos", when="+kokkos")
    depends_on("cppuddle ~kokkos", when="~kokkos")
    depends_on("hpx-kokkos@0.4.1 cxxstd=20", when="+kokkos")
    depends_on("kokkos@5.2.2 std=20 +hpx +serial +aggressive_vectorization", when="+kokkos")
    depends_on("dpcpp@2024.2.1: +cuda", when="+sycl")
    conflicts("~kokkos", when="+sycl")
    conflicts("~kokkos", when="+kokkos_hpx_kernels")
    conflicts("+cuda", when="+rocm")
    conflicts("+cuda", when="+sycl")
    conflicts("+rocm", when="+sycl")
    for backend in ("cuda", "rocm", "sycl"):
        for enabled in ("+", "~"):
            depends_on("hpx " + enabled + backend, when=enabled + backend)
            depends_on("cppuddle " + enabled + backend, when=enabled + backend)
            depends_on("hpx-kokkos " + enabled + backend, when="+kokkos " + enabled + backend)
            depends_on("kokkos " + enabled + backend, when="+kokkos " + enabled + backend)
    for arch in CudaPackage.cuda_arch_values:
        for package in ("hpx", "cppuddle"):
            depends_on(package + " cuda_arch=" + arch, when="+cuda cuda_arch=" + arch)
        for package in ("hpx-kokkos", "kokkos"):
            depends_on(package + " cuda_arch=" + arch, when="+kokkos +cuda cuda_arch=" + arch)
    for arch in ROCmPackage.amdgpu_targets:
        for package in ("hpx", "cppuddle"):
            depends_on(package + " amdgpu_target=" + arch, when="+rocm amdgpu_target=" + arch)
        for package in ("hpx-kokkos", "kokkos"):
            depends_on(package + " amdgpu_target=" + arch, when="+kokkos +rocm amdgpu_target=" + arch)

    build_directory = "spack-build"

    def setup_build_environment(self, env):
        compiler_environment(self, env)

    def cmake_args(self):
        spec = self.spec
        clang_frontend = spec.satisfies("%clang") or "+rocm" in spec or "+sycl" in spec
        # Preserve the current matrix's enabled test families, including Clang blast.
        blast = self.run_tests and "+rocm" not in spec and "+sycl" not in spec
        args = [self.define("OCTOTIGER_WITH_CXX17", False), self.define("OCTOTIGER_WITH_CXX20", True),
                self.define("CMAKE_CXX_STANDARD", "20"), self.define("CMAKE_CUDA_STANDARD", "20"),
                self.define_from_variant("OCTOTIGER_WITH_CUDA", "cuda"),
                self.define_from_variant("OCTOTIGER_WITH_HIP", "rocm"),
                self.define_from_variant("OCTOTIGER_WITH_KOKKOS", "kokkos"),
                self.define("OCTOTIGER_WITH_TESTS", self.run_tests),
                self.define("OCTOTIGER_WITH_BLAST_TEST", blast),
                self.define("OCTOTIGER_WITH_BOOST_MULTIPRECISION", clang_frontend),
                self.define("OCTOTIGER_WITH_VC", True), self.define("OCTOTIGER_WITH_LEGACY_VC", False),
                self.define_from_variant("OCTOTIGER_KOKKOS_SIMD_LIBRARY", "simd_library"),
                self.define_from_variant("OCTOTIGER_KOKKOS_SIMD_EXTENSION", "simd_extension"),
                self.define_from_variant("OCTOTIGER_WITH_GRIDDIM", "griddim"),
                self.define_from_variant("OCTOTIGER_THETA_MINIMUM", "theta_minimum"),
                self.define_from_variant("OCTOTIGER_WITH_FAST_FP_CONTRACT", "fast_fp_contract"),
                self.define("OCTOTIGER_WITH_MAX_NUMBER_FIELDS", "15"),
                self.define("OCTOTIGER_DISABLE_ILIST", False),
                self.define("OCTOTIGER_ARCH_FLAG", "-march=native "),
                self.define("OCTOTIGER_WITH_UNBUFFERED_STDOUT", False),
                self.define("CMAKE_EXPORT_COMPILE_COMMANDS", True)]
        for kernel in ("monopole", "multipole", "hydro"):
            if int(spec.variants[kernel + "_host_tasks"].value) > 1 and "~kokkos_hpx_kernels" in spec:
                raise InstallError(kernel + "_host_tasks > 1 requires +kokkos_hpx_kernels")
            args.append(self.define_from_variant("OCTOTIGER_WITH_" + kernel.upper() + "_HOST_HPX_EXECUTOR", "kokkos_hpx_kernels"))
            args.append(self.define_from_variant("OCTOTIGER_WITH_KOKKOS_" + kernel.upper() + "_TASKS", kernel + "_host_tasks"))
        if "+cuda" in spec:
            arch = spec.variants["cuda_arch"].value[0]
            if arch == "none":
                raise InstallError("An explicit CUDA architecture is required")
            args.append(self.define("OCTOTIGER_CUDA_ARCH", "sm_" + arch))
        if spec.satisfies("+cuda +kokkos %gcc"):
            args.append(self.define("CMAKE_CXX_COMPILER_LAUNCHER", "python3;" + join_path(self.stage.source_path, ".jenkins", "lsu", "nvccHostDefinitions.py")))
        return args + compiler_args(self, kokkos="+kokkos" in spec)

    def check(self):
        if not self.run_tests:
            return
        if "+sycl" in self.spec:
            for name in ("CUDA_VISIBLE_DEVICES", "KOKKOS_VISIBLE_DEVICES", "SYCL_DEVICE_FILTER", "ONEAPI_DEVICE_SELECTOR"):
                value = repr(os.environ[name]) if name in os.environ else "<unset>"
                print("SYCL test environment: " + name + "=" + value, flush=True)
            executable = join_path(self.spec["dpcpp"].prefix, "bin", "sycl-ls")
            try:
                result = subprocess.run([executable], timeout=30, check=False)
                print("SYCL device diagnostic exit status: " + str(result.returncode), flush=True)
            except (OSError, subprocess.TimeoutExpired) as error:
                print("SYCL device diagnostic unavailable: " + str(error), flush=True)
        # Invoke CTest directly: preserve streamed progress and prevent Spack
        # from adding parallelism; HPX uses the row's allocated worker count.
        with working_dir(self.build_directory):
            ctest("--output-on-failure", "--timeout", "3600", "--parallel", "1")
