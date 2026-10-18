# SPDX-License-Identifier: BSL-1.0
"""HPX master stack; copied into a private Jenkins Spack repository."""
import hashlib
import os

from spack.package import (
    CMakePackage,
    CudaPackage,
    InstallError,
    ROCmPackage,
    conflicts,
    depends_on,
    generator,
    join_path,
    patch,
    resource,
    run_before,
    variant,
    version,
    which,
    working_dir,
)


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
    version("master", commit="dc521c20f9d2a8b9de3b9ea53f634c6afa3ea258", preferred=True)
    patch("hpx-nvcc.patch", when="+cuda %gcc")
    patch("hpx-sycl.patch", when="+sycl")
    resource(name="asio", git="https://github.com/chriskohlhoff/asio.git",
             commit="12e0ce9e0500bf0f247dbd1ae894272656456079",
             destination="hpx-master-resources", placement="asio")
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

    @run_before("cmake")
    def verify_source_revision(self):
        expected = "dc521c20f9d2a8b9de3b9ea53f634c6afa3ea258"
        source = self.stage.source_path
        if os.path.exists(join_path(source, ".git")):
            git = which("git", required=True)
            with working_dir(source):
                actual = git("rev-parse", "HEAD", output=str).strip()
            if actual != expected:
                raise InstallError("HPX checkout differs from pinned master: " + actual)
            print("Verified HPX master checkout: " + actual, flush=True)
        else:
            # Spack mirrors can archive Git sources without their .git metadata.
            # The recipe commit fixes the fetch; check identifying source files
            # too, without accidentally reading the parent Octotiger checkout.
            fingerprints = {'CMakeLists.txt': 'e1e7fd70ae93f0379ce5afebd5e17c8f481439155910080aa8e37cabf61975e7', 'cmake/HPX_SetupAsio.cmake': '9df0e5af35aae2b0dc1ff49334fd5fcbf4b5c648f19832fdbc31152a8430aab5', 'libs/core/async_sycl/src/sycl_event_callback.cpp': 'f89247751e3781adebdffe655ff41fc08d9c7ee6cab359d9f9c228802cc09cd1'}
            for relative, wanted in fingerprints.items():
                with open(join_path(source, relative), "rb") as handle:
                    actual = hashlib.sha256(handle.read()).hexdigest()
                if actual != wanted:
                    raise InstallError("HPX pinned archive source differs: " + relative)
            print("Verified HPX master pinned archive source hashes: " + expected, flush=True)

        if self.spec.satisfies("+cuda %gcc"):
            # Verify the reviewed compatibility subset after Spack patches the
            # pinned base, for both Git checkouts and source archives.
            fingerprints = {'cmake/HPX_SetupStdexec.cmake': '3bda25c8babfd7098d31efb748699f83c89b9d70cbe98961b77f5a118297df69', 'cmake/HPX_StdexecNvccGcc13.patch': '4c9c98f102e7c0c943a3903aa4cf3260aa62f2dbb67c7451286423bcba6787b4', 'libs/core/executors/include/hpx/executors/thread_pool_continues_on_sender.hpp': '80136adef178d4a18f4f9d03a13e7cbc4820eed0ed9d3bdb6cb7968e78564fce'}
            for relative, wanted in fingerprints.items():
                with open(join_path(source, relative), "rb") as handle:
                    actual = hashlib.sha256(handle.read()).hexdigest()
                if actual != wanted:
                    raise InstallError("HPX NVCC compatibility source differs: " + relative)
            print("Verified HPX NVCC compatibility subset: PR 7711 revision "
                  "8767d155e4cedec8c545951bfea92f92820fd671", flush=True)

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
                self.define("FETCHCONTENT_SOURCE_DIR_ASIO", join_path(self.stage.source_path, "hpx-master-resources", "asio")),
                self.define("HPX_WITH_ASIO_TAG", "asio-1-30-2"),
                self.define("HPX_WITH_FETCH_STDEXEC", True),
                self.define("HPX_WITH_STDEXEC_TAG", "04de75de5807e73a229acf047c6976575e313359"),
                self.define("BOOST_ROOT", self.spec["boost"].prefix),
                self.define("HWLOC_ROOT", self.spec["hwloc"].prefix)]
        if "+rocm" in self.spec:
            # Octotiger uses HIP execution, not HPX's optional BLAS executor.
            # Site hipBLAS discovery enables that executor despite its missing
            # legacy hipblas.h header under ROCm 6.4.1.
            args.append(self.define("CMAKE_DISABLE_FIND_PACKAGE_hipblas", True))
        if "+sycl" in self.spec:
            args.append(self.define("HPX_WITH_SYCL_FLAGS", "-fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70"))
        return args + compiler_args(self)
