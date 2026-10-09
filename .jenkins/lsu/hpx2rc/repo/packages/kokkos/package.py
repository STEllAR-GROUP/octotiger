# SPDX-License-Identifier: BSL-1.0
"""Pristine Kokkos 5.2.2: no source patch directives."""
from spack.package import *
from spack.pkg.octotiger_hpx2rc.hpx import compiler_args, compiler_environment


class Kokkos(CMakePackage, CudaPackage, ROCmPackage):
    homepage = "https://github.com/kokkos/kokkos"
    git = "https://github.com/kokkos/kokkos.git"
    version("5.2.2", commit="8e15454876298832f013ebcaf08d53825b5bda40", preferred=True)
    generator("ninja")
    variant("std", default="20", values=("20",), multi=False, description="Validated C++ standard")
    variant("serial", default=True, description="Serial execution space")
    variant("hpx", default=True, description="HPX execution space")
    variant("sycl", default=False, description="SYCL execution space")
    variant("shared", default=True, description="Shared libraries")
    variant("aggressive_vectorization", default=True, description="Vectorization hints")
    variant("use_unsupported_sycl_arch", default="70", values=("70",), multi=False, description="SYCL NVIDIA architecture")
    depends_on("cmake@3.22:", type="build")
    depends_on("hpx@2.0.0rc1 cxxstd=20", when="+hpx")
    depends_on("hpx +cuda", when="+hpx +cuda")
    depends_on("hpx ~cuda", when="+hpx ~cuda")
    depends_on("hpx +rocm", when="+hpx +rocm")
    depends_on("hpx ~rocm", when="+hpx ~rocm")
    depends_on("hpx +sycl", when="+hpx +sycl")
    depends_on("hpx ~sycl", when="+hpx ~sycl")
    depends_on("cuda@12.2:", when="+cuda")
    depends_on("hip@6.2:", when="+rocm")
    depends_on("dpcpp@2024.2.1: +cuda", when="+sycl")
    depends_on("onedpl@2022.7.1", when="+sycl", type=("build", "link"))
    conflicts("%gcc@:11", when="~sycl")
    conflicts("%clang@:15")
    conflicts("+cuda", when="+rocm")
    conflicts("+cuda", when="+sycl")
    conflicts("+rocm", when="+sycl")
    conflicts("+shared", when="+cuda %clang", msg="Use ~shared to match the validated native-Clang CUDA stack")
    for arch in CudaPackage.cuda_arch_values:
        depends_on("hpx cuda_arch=" + arch, when="+hpx +cuda cuda_arch=" + arch)
    for arch in ROCmPackage.amdgpu_targets:
        depends_on("hpx amdgpu_target=" + arch, when="+hpx +rocm amdgpu_target=" + arch)

    def setup_build_environment(self, env):
        compiler_environment(self, env)

    def setup_dependent_build_environment(self, env, dependent_spec):
        compiler_environment(self, env)

    def cmake_args(self):
        args = [self.define("CMAKE_CXX_STANDARD", "20"),
                self.define("CMAKE_POSITION_INDEPENDENT_CODE", True),
                self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
                self.define_from_variant("Kokkos_ENABLE_SERIAL", "serial"),
                self.define_from_variant("Kokkos_ENABLE_HPX", "hpx"),
                self.define_from_variant("Kokkos_ENABLE_IMPL_HPX_ASYNC_DISPATCH", "hpx"),
                self.define_from_variant("Kokkos_ENABLE_CUDA", "cuda"),
                self.define_from_variant("Kokkos_ENABLE_HIP", "rocm"),
                self.define_from_variant("Kokkos_ENABLE_SYCL", "sycl"),
                self.define_from_variant("Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION", "aggressive_vectorization"),
                self.define("Kokkos_ENABLE_TESTS", False), self.define("Kokkos_ENABLE_EXAMPLES", False)]
        if "+cuda" in self.spec or "+sycl" in self.spec:
            if "+cuda" in self.spec and not self.spec.satisfies("cuda_arch=70"):
                raise InstallError("This Jenkins matrix requires CUDA architecture 70")
            args.append(self.define("Kokkos_ARCH_VOLTA70", True))
        if "+sycl" in self.spec:
            args.append(self.define("Kokkos_ENABLE_UNSUPPORTED_ARCHS", True))
            args.append(self.define("oneDPL_DIR", self.spec["onedpl"].prefix.lib.cmake.oneDPL))
        if "+rocm" in self.spec:
            if not self.spec.satisfies("amdgpu_target=gfx908"):
                raise InstallError("This Jenkins matrix requires AMD architecture gfx908")
            args.append(self.define("Kokkos_ARCH_AMD_GFX908", True))
        return args + compiler_args(self, kokkos=True, own_kokkos=True)
