// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include <Kokkos_Core.hpp>
#include "octotiger/radiation/accelerator.hpp"
#include "octotiger/radiation/accelerator_math.hpp"

#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace radiation_accelerator { namespace detail {
namespace {
using Execution = Kokkos::DefaultExecutionSpace;
template <class T> using View = Kokkos::View<T*, Execution>;
using Error = Kokkos::View<int, Execution>;
using Range = Kokkos::RangePolicy<Execution, Kokkos::IndexType<size_t>>;
std::mutex execution_mutex;
bool owns_kokkos = false;

void require_supported_execution()
{
#ifdef KOKKOS_ENABLE_HPX
    if constexpr (std::is_same_v<Execution, Kokkos::Experimental::HPX>) {
        // An HPX fence may suspend its lightweight caller. Serializing those
        // callers with this backend's ordinary mutex can exhaust HPX workers.
        // Supporting that execution space needs independent execution instances
        // and workspaces; do not silently run the shared-cache path instead.
        throw std::runtime_error("The radiation Kokkos backend does not support HPX as its default execution space; "
            "select --radiation_backend=cpu or use a Kokkos build with CUDA, Serial, or OpenMP as its default");
    }
#endif
}

void require_initialized()
{
    require_supported_execution();
    if (!Kokkos::is_initialized()) {
        throw std::runtime_error("Initialize the radiation Kokkos backend inside the application runtime before use");
    }
}

struct Workspace {
    Execution execution;
    View<double> state, coordinates, primitives, flux, before;
    View<CouplingInput> input;
    View<CouplingOutput> output;
    Error error{"radiation error"};
    bool valid = true;
};
// Explicit teardown avoids destroying Kokkos views after the runtime shuts down.
Workspace* cached_workspace = nullptr;

Workspace& workspace()
{
    if (!cached_workspace) {
        cached_workspace = new Workspace;
    }
    if (!cached_workspace->valid) {
        throw std::runtime_error("Radiation Kokkos workspace had an execution failure; finalize before reuse");
    }
    return *cached_workspace;
}

struct Fence {
    Workspace& workspace;
    explicit Fence(Workspace& value) : workspace(value) {}
    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;
    ~Fence()
    {
        try {
            // In particular, drain copies before host destinations are destroyed
            // if an operation throws before reaching its normal completion fence.
            workspace.execution.fence("radiation workspace release");
        } catch (...) {
            workspace.valid = false;
        }
    }
};

template <class T>
void reserve(View<T>& storage, size_t count, const char* label)
{
    if (storage.extent(0) < count) {
        storage = View<T>(label, count);
    }
}

template <class T>
auto active(const View<T>& storage, size_t count)
{
    // Capacity can exceed this call's shape after a larger block/batch. Restrict
    // every transfer to its active span rather than copying the retained tail.
    return Kokkos::subview(storage, std::make_pair(size_t{0}, count));
}

template <class T>
void upload(const Execution& execution, const View<T>& destination, const std::vector<T>& source)
{
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host(source.data(), source.size());
    Kokkos::deep_copy(execution, active(destination, source.size()), host);
}

template <class T>
void download(const Execution& execution, std::vector<T>& destination, const View<T>& source)
{
    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host(destination.data(), destination.size());
    Kokkos::deep_copy(execution, host, active(source, destination.size()));
}

bool valid(const Execution& execution, const Error& error)
{
    execution.fence("radiation numerical validity");
    int status = 0;
    Kokkos::deep_copy(status, error);
    return status == 0;
}

void finish(const Execution& execution, const Error& error)
{
    if (!valid(execution, error)) {
        throw std::runtime_error("Radiation Kokkos cell failed its numerical validity checks; outputs retained");
    }
}
} // namespace

void kokkos_initialize()
{
    require_supported_execution();
    std::lock_guard<std::mutex> lock(execution_mutex);
    if (!Kokkos::is_initialized()) {
        if (Kokkos::is_finalized()) {
            throw std::runtime_error("Cannot initialize radiation Kokkos after Kokkos was finalized");
        }
        Kokkos::initialize();
        owns_kokkos = true;
    }
}

void kokkos_finalize()
{
    std::lock_guard<std::mutex> lock(execution_mutex);
    if (cached_workspace) {
        require_initialized();
        cached_workspace->execution.fence("radiation workspace finalization");
        delete cached_workspace;
        cached_workspace = nullptr;
    }
    // Views must be released above even when the application owns Kokkos and
    // will finalize it separately after this function returns.
    if (owns_kokkos) {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
        owns_kokkos = false;
    }
}

void kokkos_flux(Grid grid, const std::vector<double>& state, const std::vector<double>& coordinates,
    std::vector<double>& flux)
{
    // Retain one capacity-grown workspace under the default-execution lock.
    // Distinct HPX callers cannot interleave operations on its shared storage.
    std::lock_guard<std::mutex> lock(execution_mutex);
    require_initialized();
    const size_t count = size_t(grid.n) * grid.n * grid.n;
    std::vector<double> host_result(12 * count);
    auto& storage = workspace();
    Fence fence(storage);
    reserve(storage.state, state.size(), "radiation state");
    reserve(storage.coordinates, coordinates.size(), "radiation coordinates");
    reserve(storage.primitives, 4 * count, "radiation primitives");
    reserve(storage.flux, 12 * count, "radiation flux");
    const auto execution = storage.execution;
    const auto device_state = storage.state;
    const auto device_coordinates = storage.coordinates;
    const auto primitives = storage.primitives;
    const auto result = storage.flux;
    const auto error = storage.error;
    upload(execution, device_state, state);
    upload(execution, device_coordinates, coordinates);
    Kokkos::deep_copy(execution, active(primitives, 4 * count), 0.);
    Kokkos::deep_copy(execution, active(result, 12 * count), 0.);
    Kokkos::deep_copy(execution, error, 0);
    Kokkos::parallel_for("radiation primitives", Range(execution, 0, count), KOKKOS_LAMBDA(size_t r) {
        if (!math::primitives(grid, device_state.data(), primitives.data(), r)) {
            Kokkos::atomic_exchange(&error(), 1);
        }
    });
    for (int normal = 0; normal < 3; ++normal) {
        Kokkos::parallel_for("radiation faces", Range(execution, 0, count), KOKKOS_LAMBDA(size_t r) {
            if (!math::face_flux(grid, primitives.data(), device_coordinates.data(), result.data(), normal, r)) {
                Kokkos::atomic_exchange(&error(), 1);
            }
        });
    }
    finish(execution, error);
    download(execution, host_result, result);
    execution.fence("radiation flux download");
    flux.swap(host_result);
}

void kokkos_advance(Grid grid, std::vector<double>& state, const std::vector<double>& flux,
    std::vector<double>& before_rotation)
{
    std::lock_guard<std::mutex> lock(execution_mutex);
    require_initialized();
    const size_t count = size_t(grid.n) * grid.n * grid.n;
    std::vector<double> host_result(state.size()), host_before(2 * count);
    auto& storage = workspace();
    Fence fence(storage);
    reserve(storage.state, state.size(), "radiation advanced state");
    reserve(storage.flux, flux.size(), "radiation input flux");
    reserve(storage.before, 2 * count, "radiation before rotation");
    const auto execution = storage.execution;
    const auto result = storage.state;
    const auto device_flux = storage.flux;
    const auto before = storage.before;
    const auto error = storage.error;
    upload(execution, result, state);
    upload(execution, device_flux, flux);
    Kokkos::deep_copy(execution, active(before, 2 * count), 0.);
    Kokkos::deep_copy(execution, error, 0);
    Kokkos::parallel_for("radiation advance", Range(execution, 0, count), KOKKOS_LAMBDA(size_t r) {
        if (!math::advance_cell(grid, result.data(), device_flux.data(), before.data(), r)) {
            Kokkos::atomic_exchange(&error(), 1);
        }
    });
    finish(execution, error);
    download(execution, host_result, result);
    download(execution, host_before, before);
    execution.fence("radiation advance download");
    state.swap(host_result);
    before_rotation.swap(host_before);
}

void kokkos_couple(const std::vector<CouplingInput>& input, double dt, double c,
    double reduced_c_factor, bool marshak, std::vector<CouplingOutput>& output)
{
    std::lock_guard<std::mutex> lock(execution_mutex);
    require_initialized();
    if (input.empty()) {
        output.clear();
        return;
    }
    std::vector<CouplingOutput> host_result(input.size());
    auto& storage = workspace();
    Fence fence(storage);
    reserve(storage.input, input.size(), "radiation coupling input");
    reserve(storage.output, input.size(), "radiation coupling output");
    const auto execution = storage.execution;
    const auto device_input = storage.input;
    const auto result = storage.output;
    const auto error = storage.error;
    upload(execution, device_input, input);
    Kokkos::deep_copy(execution, error, 0);
    Kokkos::parallel_for("radiation coupling", Range(execution, 0, input.size()), KOKKOS_LAMBDA(size_t r) {
        if (!math::couple_cell(device_input(r), dt, c, reduced_c_factor, marshak, result(r))) {
            Kokkos::atomic_exchange(&error(), 1);
        }
    });
    finish(execution, error);
    download(execution, host_result, result);
    execution.fence("radiation coupling download");
    output.swap(host_result);
}

bool kokkos_couple_implicit(const std::vector<CouplingInput>& input, double dt, double c,
    double reduced_c_factor, bool marshak, std::vector<CouplingOutput>& output)
{
    std::lock_guard<std::mutex> lock(execution_mutex);
    require_initialized();
    if (input.empty()) {
        output.clear();
        return true;
    }
    std::vector<CouplingOutput> host_result(input.size());
    auto& storage = workspace();
    Fence fence(storage);
    reserve(storage.input, input.size(), "radiation implicit coupling input");
    reserve(storage.output, input.size(), "radiation implicit coupling output");
    const auto execution = storage.execution;
    const auto device_input = storage.input;
    const auto result = storage.output;
    const auto error = storage.error;
    upload(execution, device_input, input);
    Kokkos::deep_copy(execution, error, 0);
    Kokkos::parallel_for("radiation implicit coupling", Range(execution, 0, input.size()), KOKKOS_LAMBDA(size_t r) {
        if (!implicit_couple_cell(device_input(r), dt, c, reduced_c_factor, marshak, result(r))) {
            Kokkos::atomic_exchange(&error(), 1);
        }
    });
    if (!valid(execution, error)) {
        return false;
    }
    download(execution, host_result, result);
    execution.fence("radiation implicit coupling download");
    output.swap(host_result);
    return true;
}
} } // namespace radiation_accelerator::detail
