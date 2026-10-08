# SPDX-License-Identifier: BSL-1.0
from spack.package import *
from spack.pkg.octotiger_hpx2rc.hpx import compiler_args, compiler_environment


class HpxKokkos(CMakePackage, CudaPackage, ROCmPackage):
    homepage = "https://github.com/STEllAR-GROUP/hpx-kokkos"
    git = "https://github.com/STEllAR-GROUP/hpx-kokkos.git"
    version("0.4.1", commit="5b794d0b9ca10fb1eecc950d49992abfa7fb609c", preferred=True)
    patch("hpx-kokkos.patch")
    generator("ninja")
    variant("cxxstd", default="20", values=("20",), multi=False, description="Validated C++ standard")
    variant("sycl", default=False, description="SYCL integration")
    variant("future_type", default="polling", values=("polling", "callback"), multi=False, description="CUDA future implementation")
    depends_on("cmake@3.22:", type="build")
    depends_on("hpx@2.0.0rc1 cxxstd=20")
    depends_on("kokkos@5.2.2 std=20 +hpx +serial")
    depends_on("dpcpp@2024.2.1: +cuda", when="+sycl")
    for backend in ("cuda", "rocm", "sycl"):
        for enabled in ("+", "~"):
            depends_on("hpx " + enabled + backend, when=enabled + backend)
            depends_on("kokkos " + enabled + backend, when=enabled + backend)
    for arch in CudaPackage.cuda_arch_values:
        depends_on("kokkos cuda_arch=" + arch, when="+cuda cuda_arch=" + arch)
    for arch in ROCmPackage.amdgpu_targets:
        depends_on("kokkos amdgpu_target=" + arch, when="+rocm amdgpu_target=" + arch)

    def setup_build_environment(self, env):
        compiler_environment(self, env)

    def cmake_args(self):
        return [self.define("CMAKE_CXX_STANDARD", "20"),
                self.define("HPX_KOKKOS_CUDA_FUTURE_TYPE", "event" if "future_type=polling" in self.spec else "callback"),
                self.define("HPX_KOKKOS_SYCL_FUTURE_TYPE", "event"),
                self.define("HPX_KOKKOS_ENABLE_TESTS", False),
                self.define("HPX_KOKKOS_ENABLE_BENCHMARKS", False)] + compiler_args(self, kokkos=True)
