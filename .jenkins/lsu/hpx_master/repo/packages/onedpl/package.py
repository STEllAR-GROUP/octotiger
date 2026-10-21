# SPDX-License-Identifier: BSL-1.0
from spack.package import *


class Onedpl(Package):
    """Header-only oneDPL for the pinned Intel LLVM 2025.0 SYCL stack."""

    homepage = "https://github.com/uxlfoundation/oneDPL"
    git = "https://github.com/uxlfoundation/oneDPL.git"
    version("2022.7.1", commit="4d9921f2203ea452a020bf2e52947a7e576c35e1")
    depends_on("cmake@3.11:", type="build")

    def install(self, spec, prefix):
        install_tree("include", prefix.include)
        install_tree("licensing", prefix.share.licenses.onedpl)
        # Use upstream's installation generator without configuring its test
        # suite. The config selects backends with the consuming SYCL compiler.
        config = prefix.lib.cmake.oneDPL
        cmake = Executable(join_path(spec["cmake"].prefix, "bin", "cmake"))
        cmake("-DOUTPUT_DIR=" + str(config), "-DSKIP_HEADERS_SUBDIR=TRUE",
              "-P", "cmake/scripts/generate_config.cmake")
