# SPDX-License-Identifier: BSL-1.0
from spack.package import *
from spack.pkg.octotiger_hpx2rc.hpx import compiler_args, compiler_environment


class Cppuddle(CMakePackage, CudaPackage, ROCmPackage):
    homepage = "https://github.com/SC-SGS/CPPuddle"
    git = "https://github.com/SC-SGS/CPPuddle.git"
    version("0.4.0", commit="5d189e5b910d03901b358ffa5c2539e2e71edb06", preferred=True)
    patch("cppuddle.patch")
    generator("ninja")
    variant("cxxstd", default="20", values=("20",), multi=False, description="Validated C++ standard")
    variant("hpx", default=True, description="HPX integration")
    variant("kokkos", default=False, description="Kokkos integration")
    variant("sycl", default=False, description="SYCL integration")
    depends_on("cmake@3.22:", type="build")
    depends_on("hpx@2.0.0rc1 cxxstd=20", when="+hpx")
    depends_on("hpx-kokkos@0.4.1 cxxstd=20", when="+kokkos")
    depends_on("kokkos@5.2.2 std=20 +hpx +serial", when="+kokkos")
    depends_on("dpcpp@2024.2.1: +cuda", when="+sycl")
    conflicts("~hpx")
    conflicts("~kokkos", when="+sycl")
    for backend in ("cuda", "rocm", "sycl"):
        for enabled in ("+", "~"):
            depends_on("hpx " + enabled + backend, when=enabled + backend)
            depends_on("hpx-kokkos " + enabled + backend, when="+kokkos " + enabled + backend)
    for arch in CudaPackage.cuda_arch_values:
        depends_on("hpx cuda_arch=" + arch, when="+cuda cuda_arch=" + arch)
        depends_on("hpx-kokkos cuda_arch=" + arch, when="+kokkos +cuda cuda_arch=" + arch)
    for arch in ROCmPackage.amdgpu_targets:
        depends_on("hpx amdgpu_target=" + arch, when="+rocm amdgpu_target=" + arch)
        depends_on("hpx-kokkos amdgpu_target=" + arch, when="+kokkos +rocm amdgpu_target=" + arch)

    def setup_build_environment(self, env):
        compiler_environment(self, env)

    def cmake_args(self):
        return [self.define("CMAKE_CXX_STANDARD", "20"),
                self.define("CPPUDDLE_WITH_HPX", True),
                self.define_from_variant("CPPUDDLE_WITH_KOKKOS", "kokkos"),
                # This option enables CPPuddle's own tests, which are off;
                # GPU support in the headers comes from the matching HPX.
                self.define("CPPUDDLE_WITH_CUDA", False),
                self.define("CPPUDDLE_WITH_TESTS", False),
                self.define("CPPUDDLE_WITH_HPX_AWARE_ALLOCATORS", True),
                self.define("CPPUDDLE_WITH_HPX_MUTEX", True),
                self.define("CPPUDDLE_WITH_MAX_NUMBER_GPUS", "1"),
                self.define("CPPUDDLE_WITH_NUMBER_BUCKETS", "128")] + compiler_args(self, kokkos="+kokkos" in self.spec)
