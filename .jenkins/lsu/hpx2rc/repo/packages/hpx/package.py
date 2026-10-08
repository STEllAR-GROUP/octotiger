# SPDX-License-Identifier: BSL-1.0
"""Pinned HPX 2 RC stack; copied into a private Jenkins Spack repository."""
from spack.package import *


def compiler_args(pkg, kokkos=False, own_kokkos=False):
    """Select one compiler family consistently through the complete stack."""
    spec = pkg.spec
    args = []
    if "+sycl" in spec:
        args.append(pkg.define("CMAKE_CXX_COMPILER", join_path(spec["dpcpp"].prefix, "bin", "clang++")))
    elif "+rocm" in spec:
        args.append(pkg.define("CMAKE_CXX_COMPILER", spec["hip"].hipcc))
        args.append(pkg.define("GPU_TARGETS", ";".join(spec.variants["amdgpu_target"].value)))
    elif "+cuda" in spec:
        archs = [arch for arch in spec.variants["cuda_arch"].value if arch != "none"]
        if archs:
            args.append(pkg.define("CMAKE_CUDA_ARCHITECTURES", ";".join(archs)))
        if spec.satisfies("%clang"):
            args.append(pkg.define("CMAKE_CXX_COMPILER", pkg.compiler.cxx))
            args.append(pkg.define("CMAKE_CUDA_COMPILER", pkg.compiler.cxx))
        else:
            args.append(pkg.define("CMAKE_CUDA_COMPILER", join_path(spec["cuda"].prefix, "bin", "nvcc")))
            args.append(pkg.define("CMAKE_CUDA_HOST_COMPILER", pkg.compiler.cxx))
            if kokkos:
                prefix = pkg.stage.source_path if own_kokkos else spec["kokkos"].prefix
                args.append(pkg.define("CMAKE_CXX_COMPILER", join_path(prefix, "bin", "nvcc_wrapper")))
    return args


def compiler_environment(pkg, env):
    if pkg.spec.satisfies("+cuda %gcc"):
        env.set("NVCC_WRAPPER_DEFAULT_COMPILER", pkg.compiler.cxx)
        env.set("NVCC_WRAPPER_DEFAULT_COMPILER_FLAGS", "")
    if pkg.spec.satisfies("+cuda %clang"):
        env.set("CUDACXX", pkg.compiler.cxx)


class Hpx(CMakePackage, CudaPackage, ROCmPackage):
    homepage = "https://github.com/TheHPXProject/hpx"
    git = "https://github.com/TheHPXProject/hpx.git"
    version("2.0.0rc1", commit="2b2018189ab8ffd190c7ad04fd4da5534286726b", preferred=True)
    patch("hpx.patch")
    resource(name="asio", git="https://github.com/chriskohlhoff/asio.git",
             commit="12e0ce9e0500bf0f247dbd1ae894272656456079",
             destination="hpx2rc-resources", placement="asio")
    generator("ninja")
    variant("cxxstd", default="20", values=("20",), multi=False, description="Validated C++ standard")
    variant("malloc", default="system", values=("system", "jemalloc"), multi=False, description="Allocator")
    variant("max_cpu_count", default="128", values=lambda value: str(value).isdigit(), multi=False, description="HPX worker capacity")
    variant("networking", default="tcp", values=("tcp", "none"), multi=False, description="Parcel transport")
    variant("async_cuda", default=False, description="CUDA futures integration")
    variant("sycl", default=False, description="SYCL integration")
    variant("sycl_target_arch", default="70", values=("70",), multi=False, description="SYCL NVIDIA architecture")
    depends_on("cmake@3.22:", type="build")
    depends_on("git", type="build")
    depends_on("hwloc")
    depends_on("boost@1.74: cxxstd=2a +program_options")
    depends_on("jemalloc", when="malloc=jemalloc")
    depends_on("cuda@12.2:", when="+cuda")
    depends_on("hip@6.2:", when="+rocm")
    depends_on("dpcpp@2024.2.1: +cuda", when="+sycl")
    conflicts("%gcc@:11", when="~sycl")
    conflicts("%clang@:15")
    conflicts("+cuda", when="+rocm")
    conflicts("+cuda", when="+sycl")
    conflicts("+rocm", when="+sycl")
    conflicts("+async_cuda", when="~cuda")

    def setup_build_environment(self, env):
        compiler_environment(self, env)

    def setup_dependent_build_environment(self, env, dependent_spec):
        compiler_environment(self, env)

    def cmake_args(self):
        args = [self.define("HPX_WITH_CXX_STANDARD", "20"),
                self.define_from_variant("HPX_WITH_MALLOC", "malloc"),
                self.define_from_variant("HPX_WITH_MAX_CPU_COUNT", "max_cpu_count"),
                self.define_from_variant("HPX_WITH_CUDA", "cuda"),
                self.define_from_variant("HPX_WITH_HIP", "rocm"),
                self.define_from_variant("HPX_WITH_SYCL", "sycl"),
                self.define("HPX_WITH_DISTRIBUTED_RUNTIME", True),
                self.define("HPX_WITH_NETWORKING", "networking=tcp" in self.spec),
                self.define("HPX_WITH_PARCELPORT_TCP", "networking=tcp" in self.spec),
                self.define("HPX_WITH_PARCELPORT_MPI", False),
                self.define("HPX_WITH_TESTS", False), self.define("HPX_WITH_EXAMPLES", False),
                self.define("HPX_WITH_FETCH_ASIO", True),
                self.define("FETCHCONTENT_SOURCE_DIR_ASIO", join_path(self.stage.source_path, "hpx2rc-resources", "asio")),
                self.define("HPX_WITH_ASIO_TAG", "asio-1-30-2"),
                self.define("HPX_WITH_FETCH_STDEXEC", True),
                self.define("HPX_WITH_STDEXEC_TAG", "04de75de5807e73a229acf047c6976575e313359"),
                self.define("BOOST_ROOT", self.spec["boost"].prefix),
                self.define("HWLOC_ROOT", self.spec["hwloc"].prefix)]
        if "+sycl" in self.spec:
            args.append(self.define("HPX_WITH_SYCL_FLAGS", "-fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70"))
        return args + compiler_args(self)
