//  Copyright (c) 2024 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once

#if defined(OCTOTIGER_HAVE_KOKKOS) && defined(KOKKOS_ENABLE_SYCL)
#include <CL/sycl.hpp>

namespace octotiger {
namespace sycl_util {

    // We encounter segfaults on Intel GPUs when running the normal kernels for the first time after
    // the program starts. This seems to be some initialization issue as we can simply fix it by
    // (non-concurrently) run simple dummy kernel first right after starting octotiger
    // (presumably initializes something within the intel gpu runtime).
    // Curiousely we have to do this not once per program, but once per octolib and hydrolib.
    //
    // Somewhat of an ugly workaround but it does the trick and allows us to target Intel GPUs as
    // Octo-Tiger runs as expected after applying this workaround.

    // TODO(daissgr) Check again in the future to see if the runtime has matured and we don't need this anymore. 
    // (last check was 02/2024)

    /// Utility function working around segfault on Intel GPU. Initializes something within the runtime by runnning
    ///a dummy kernel
    int touch_sycl_device_by_running_a_dummy_kernel(void) {
        try {
            cl::sycl::queue q(cl::sycl::default_selector_v, cl::sycl::property::queue::in_order{});
            cl::sycl::event my_kernel_event = q.submit(
                [&](cl::sycl::handler& h) {
                    h.parallel_for(512, [=](auto i) {});
                },
                cl::sycl::detail::code_location{});
            my_kernel_event.wait();
        } catch (sycl::exception const& e) {
            std::cerr << "ERROR: Caught sycl::exception during SYCL dummy
                kernel !\n "; std::cerr << " {what}
              : " << e.what() << "\n "; std::cerr << " Aborting now...\n ";
            std::cerr << "Running on device: "
                      << q.get_device().get_info<cl::sycl::info::device::name>() << "\n";
              return 2;

        } catch (std::exception const& e) {
            std::cerr << "ERROR: Caught std::exception during SYCL dummy kernel!\n";
            std::cerr << "{what}: " << e.what() << "\n";
            std::cerr << "Running on device: "
                      << q.get_device().get_info<cl::sycl::info::device::name>() << "\n";
            std::cerr << "Aborting now...\n";
            return 3;
        }
        return 1;
    }

    /// Dummy variable to ensure the touch_sycl_device_by_running_a_dummy_kernel is being run
    const int init_sycl_device = touch_sycl_device_by_running_a_dummy_kernel();

}    // namespace sycl_util
}    // namespace octotiger

#endif
