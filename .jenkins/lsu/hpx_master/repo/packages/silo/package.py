# SPDX-License-Identifier: BSL-1.0
"""Use the same pristine Silo release as the validated local Octotiger stack."""
from spack.package import *


class Silo(CMakePackage):
    homepage = "https://github.com/LLNL/Silo"
    git = "https://github.com/LLNL/Silo.git"
    # Includes the HDF5 >=1.13.3 VFD table fields and corrected DB_CURVE
    # dataset-name access, both diagnosed by modern Clang in Silo 4.11.
    version("4.12.1", commit="f05c5a8e6c22b784b7f5eb958d50218cae873d07", preferred=True)
    generator("ninja")
    variant("shared", default=True, description="Build shared Silo libraries")
    variant("fortran", default=False, description="Build the optional Fortran interface")
    variant("mpi", default=False, description="MPI support (not used by this stack)")
    conflicts("+mpi", msg="The pinned Octotiger stack uses serial Silo I/O")
    depends_on("cmake@3.19:", type="build")
    depends_on("hdf5@1.10.2:1.14 +hl ~mpi")

    def cmake_args(self):
        return [self.define_from_variant("SILO_ENABLE_SHARED", "shared"),
                self.define_from_variant("SILO_ENABLE_FORTRAN", "fortran"),
                self.define("SILO_ENABLE_HDF5", True),
                self.define("SILO_HDF5_DIR", self.spec["hdf5"].prefix),
                self.define("SILO_BUILD_FOR_BSD_LICENSE", True),
                self.define("SILO_ENABLE_ZFP", False),
                self.define("SILO_ENABLE_FPZIP", False),
                self.define("SILO_ENABLE_HZIP", False),
                self.define("SILO_ENABLE_SILOCK", False),
                self.define("SILO_ENABLE_SILEX", False),
                self.define("SILO_ENABLE_BROWSER", True),
                self.define("BUILD_TESTING", False)]
