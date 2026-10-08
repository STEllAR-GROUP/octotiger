# SPDX-License-Identifier: BSL-1.0
from os.path import dirname

from spack.package import *


class Dpcpp(Package):
    """Source-built Intel LLVM with the CUDA adapter for the V100 SYCL row.

    Intel LLVM v6.0.1 corresponds to the oneAPI 2025.0 release series. Its
    documented CUDA architecture floor includes Volta, unlike newer releases.
    """

    homepage = "https://github.com/intel/llvm"
    git = "https://github.com/intel/llvm.git"
    version("2025.0.1", commit="e42590ef4b94d24d708f51a095c8b6c56f1748ad")
    variant("cuda", default=True, description="Build the CUDA SYCL adapter")
    conflicts("~cuda")
    depends_on("cmake@3.22:", type="build")
    depends_on("ninja", type="build")
    depends_on("python@3.8:", type="build")
    depends_on("git", type="build")
    depends_on("pkgconfig", type="build")
    depends_on("zlib-api")
    depends_on("libxml2")
    depends_on("cuda@12.2:12", when="+cuda")
    phases = ["configure", "build", "install"]

    @property
    def build_directory(self):
        return join_path(self.stage.path, "dpcpp-build")

    def configure(self, spec, prefix):
        # Use the release's own build orchestration, which also selects its
        # pinned SPIR-V and Unified Runtime sources and builds device libraries.
        python = Executable(join_path(spec["python"].prefix, "bin", "python3"))
        options = [
            "-DCMAKE_INSTALL_PREFIX=" + str(prefix),
            "-DCMAKE_C_COMPILER=" + spack_cc,
            "-DCMAKE_CXX_COMPILER=" + spack_cxx,
            "-DCUDA_TOOLKIT_ROOT_DIR=" + str(spec["cuda"].prefix),
            "-DLLVM_PARALLEL_LINK_JOBS=1",
            "-DLLVM_INCLUDE_TESTS=OFF",
            "-DSYCL_INCLUDE_TESTS=OFF",
            "-DSYCL_ENABLE_PLUGINS=cuda",
        ]
        if spec.satisfies("%gcc"):
            options.append("-DGCC_INSTALL_PREFIX=" + str(dirname(dirname(self.compiler.cxx))))
        python(join_path(self.stage.source_path, "buildbot", "configure.py"),
               "--src-dir", self.stage.source_path,
               "--obj-dir", self.build_directory,
               "--cuda", "--host-target", "X86", "--no-assertions",
               "--disable-jit", "--disable-preview-lib",
               *["--cmake-opt=" + option for option in options])

    def build(self, spec, prefix):
        with working_dir(self.build_directory):
            ninja()

    def install(self, spec, prefix):
        with working_dir(self.build_directory):
            ninja("install")

    def setup_build_environment(self, env):
        env.set("CUDA_LIB_PATH", join_path(self.spec["cuda"].prefix, "lib64", "stubs"))

    def setup_run_environment(self, env):
        env.prepend_path("PATH", self.prefix.bin)
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib)
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib64)

    def setup_dependent_build_environment(self, env, dependent_spec):
        self.setup_run_environment(env)
