// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/radiation/accelerator.hpp"
#include "octotiger/radiation/accelerator_math.hpp"
#include <cuda_runtime.h>

#include <array>
#include <mutex>
#include <stdexcept>
#include <string>

namespace radiation_accelerator { namespace detail {
namespace {
void checked(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Radiation CUDA ") + operation + ": " + cudaGetErrorString(status));
    }
}

struct Stream {
    cudaStream_t value{};
    Stream() { checked(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    void reset() noexcept
    {
        if (value) {
            cudaStreamDestroy(value);
            value = nullptr;
        }
    }
    ~Stream() { reset(); }
};

template <class T> struct Buffer {
    T* data = nullptr;
    size_t capacity = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void reset() noexcept
    {
        if (data) {
            cudaFree(data);
            data = nullptr;
            capacity = 0;
        }
    }
    ~Buffer() { reset(); }
    void reserve(size_t count)
    {
        if (count > capacity) {
            T* replacement = nullptr;
            checked(cudaMalloc(reinterpret_cast<void**>(&replacement), count * sizeof(T)), "allocate workspace");
            T* previous = data;
            data = replacement;
            capacity = count;
            if (previous) {
                checked(cudaFree(previous), "grow workspace");
            }
        }
    }
    void upload(const T* source, size_t count, cudaStream_t stream)
    {
        checked(cudaMemcpyAsync(data, source, count * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
    }
    void download(T* destination, size_t count, cudaStream_t stream)
    {
        checked(cudaMemcpyAsync(destination, data, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "download");
    }
    void zero(size_t count, cudaStream_t stream)
    {
        checked(cudaMemsetAsync(data, 0, count * sizeof(T), stream), "zero buffer");
    }
};

struct Workspace {
    int device;
    Stream stream;
    Buffer<double> state, coordinates, primitives, flux, before;
    Buffer<CouplingInput> input;
    Buffer<CouplingOutput> output;
    Buffer<int> error;
    int host_error = 0;
    explicit Workspace(int selected_device) : device(selected_device) {}
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    ~Workspace()
    {
        int previous_device = device;
        const bool restore = cudaGetDevice(&previous_device) == cudaSuccess && previous_device != device;
        if (restore) {
            cudaSetDevice(device);
        }
        // Also drain exceptional paths before releasing their device allocations.
        cudaStreamSynchronize(stream.value);
        state.reset();
        coordinates.reset();
        primitives.reset();
        flux.reset();
        before.reset();
        input.reset();
        output.reset();
        error.reset();
        stream.reset();
        if (restore) {
            cudaSetDevice(previous_device);
        }
    }
};

// Retain at most twelve independent workspaces. Exhaustion uses an uncached
// workspace instead of blocking an HPX worker waiting for another task's lease.
struct Slot {
    Workspace* workspace = nullptr;
    bool leased = false;
};
std::array<Slot, 12> workspaces;
std::mutex workspace_mutex;
size_t active_leases = 0;
bool finalizing = false;

class Lease {
    int slot = -1;
    Workspace* workspace = nullptr;
public:
    Lease()
    {
        int device = 0;
        checked(cudaGetDevice(&device), "get device");
        {
            std::lock_guard<std::mutex> lock(workspace_mutex);
            if (finalizing) {
                throw std::runtime_error("Radiation CUDA workspace finalization is in progress");
            }
            for (size_t i = 0; i < workspaces.size(); ++i) {
                if (!workspaces[i].leased && workspaces[i].workspace && workspaces[i].workspace->device == device) {
                    slot = static_cast<int>(i);
                    break;
                }
            }
            if (slot < 0) {
                for (size_t i = 0; i < workspaces.size(); ++i) {
                    if (!workspaces[i].leased && !workspaces[i].workspace) {
                        slot = static_cast<int>(i);
                        break;
                    }
                }
            }
            if (slot >= 0) {
                workspaces[slot].leased = true;
                workspace = workspaces[slot].workspace;
            }
            ++active_leases;
        }
        try {
            if (!workspace) {
                workspace = new Workspace(device);
                if (slot >= 0) {
                    std::lock_guard<std::mutex> lock(workspace_mutex);
                    workspaces[slot].workspace = workspace;
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(workspace_mutex);
            if (slot >= 0) {
                workspaces[slot].leased = false;
            }
            --active_leases;
            throw;
        }
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease()
    {
        // Synchronization is required even when a launch/copy/numerical check
        // throws. A workspace with an asynchronous CUDA failure is not reused.
        const bool discard = cudaStreamSynchronize(workspace->stream.value) != cudaSuccess || slot < 0;
        if (discard) {
            delete workspace;
        }
        std::lock_guard<std::mutex> lock(workspace_mutex);
        if (slot >= 0) {
            if (discard) {
                workspaces[slot].workspace = nullptr;
            }
            workspaces[slot].leased = false;
        }
        --active_leases;
    }
    Workspace& get() { return *workspace; }
};

constexpr unsigned threads = 256;
unsigned blocks(size_t count) { return static_cast<unsigned>((count + threads - 1) / threads); }

__global__ void primitives_kernel(Grid grid, const double* state, double* primitives, size_t count, int* error)
{
    const size_t r = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < count && !math::primitives(grid, state, primitives, r)) {
        atomicExch(error, 1);
    }
}

__global__ void flux_kernel(Grid grid, const double* primitives, const double* coordinates,
    double* flux, size_t count, int normal, int* error)
{
    const size_t r = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < count && !math::face_flux(grid, primitives, coordinates, flux, normal, r)) {
        atomicExch(error, 1);
    }
}

__global__ void advance_kernel(Grid grid, double* state, const double* flux,
    double* before, size_t count, int* error)
{
    const size_t r = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < count && !math::advance_cell(grid, state, flux, before, r)) {
        atomicExch(error, 1);
    }
}

__global__ void couple_kernel(const CouplingInput* input, CouplingOutput* output, size_t count,
    double dt, double c, double reduced_c_factor, bool marshak, int* error)
{
    const size_t r = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < count && !math::couple_cell(input[r], dt, c, reduced_c_factor, marshak, output[r])) {
        atomicExch(error, 1);
    }
}

__global__ void couple_implicit_kernel(const CouplingInput* input, CouplingOutput* output, size_t count,
    double dt, double c, double reduced_c_factor, bool marshak, int* error)
{
    const size_t r = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < count && !implicit_couple_cell(input[r], dt, c, reduced_c_factor, marshak, output[r])) {
        atomicExch(error, 1);
    }
}

bool valid(Workspace& workspace)
{
    workspace.error.download(&workspace.host_error, 1, workspace.stream.value);
    checked(cudaStreamSynchronize(workspace.stream.value), "synchronize");
    return workspace.host_error == 0;
}

void finish(Workspace& workspace)
{
    if (!valid(workspace)) {
        throw std::runtime_error("Radiation CUDA cell failed its numerical validity checks; outputs retained");
    }
}
} // namespace

void cuda_finalize()
{
    std::array<Workspace*, 12> retired{};
    {
        std::lock_guard<std::mutex> lock(workspace_mutex);
        if (active_leases != 0 || finalizing) {
            throw std::runtime_error("Cannot finalize radiation CUDA while operations are active");
        }
        finalizing = true;
        for (size_t i = 0; i < workspaces.size(); ++i) {
            retired[i] = workspaces[i].workspace;
            workspaces[i].workspace = nullptr;
        }
    }
    for (Workspace* workspace : retired) {
        delete workspace;
    }
    std::lock_guard<std::mutex> lock(workspace_mutex);
    finalizing = false;
}

void cuda_flux(Grid grid, const std::vector<double>& state, const std::vector<double>& coordinates,
    std::vector<double>& flux)
{
    const size_t count = size_t(grid.n) * grid.n * grid.n;
    // Host transfer destinations outlive the lease's exception-path fence.
    std::vector<double> host_result(12 * count);
    Lease lease;
    auto& workspace = lease.get();
    auto& device_state = workspace.state;
    auto& device_coordinates = workspace.coordinates;
    auto& primitives = workspace.primitives;
    auto& result = workspace.flux;
    auto& error = workspace.error;
    const auto stream = workspace.stream.value;
    device_state.reserve(state.size());
    device_coordinates.reserve(coordinates.size());
    primitives.reserve(4 * count);
    result.reserve(12 * count);
    error.reserve(1);
    device_state.upload(state.data(), state.size(), stream);
    device_coordinates.upload(coordinates.data(), coordinates.size(), stream);
    primitives.zero(4 * count, stream);
    result.zero(12 * count, stream);
    error.zero(1, stream);
    primitives_kernel<<<blocks(count), threads, 0, stream>>>(grid, device_state.data, primitives.data, count, error.data);
    checked(cudaGetLastError(), "primitive launch");
    for (int normal = 0; normal < 3; ++normal) {
        flux_kernel<<<blocks(count), threads, 0, stream>>>(grid, primitives.data, device_coordinates.data,
            result.data, count, normal, error.data);
        checked(cudaGetLastError(), "flux launch");
    }
    finish(workspace);
    result.download(host_result.data(), host_result.size(), stream);
    checked(cudaStreamSynchronize(stream), "flux download");
    flux.swap(host_result);
}

void cuda_advance(Grid grid, std::vector<double>& state, const std::vector<double>& flux,
    std::vector<double>& before_rotation)
{
    const size_t count = size_t(grid.n) * grid.n * grid.n;
    std::vector<double> host_result(state.size()), host_before(2 * count);
    Lease lease;
    auto& workspace = lease.get();
    auto& result = workspace.state;
    auto& device_flux = workspace.flux;
    auto& before = workspace.before;
    auto& error = workspace.error;
    const auto stream = workspace.stream.value;
    result.reserve(state.size());
    device_flux.reserve(flux.size());
    before.reserve(2 * count);
    error.reserve(1);
    result.upload(state.data(), state.size(), stream); // Ghost entries remain untouched by the kernel.
    device_flux.upload(flux.data(), flux.size(), stream);
    before.zero(2 * count, stream);
    error.zero(1, stream);
    advance_kernel<<<blocks(count), threads, 0, stream>>>(grid, result.data, device_flux.data,
        before.data, count, error.data);
    checked(cudaGetLastError(), "advance launch");
    finish(workspace);
    result.download(host_result.data(), host_result.size(), stream);
    before.download(host_before.data(), host_before.size(), stream);
    checked(cudaStreamSynchronize(stream), "advance download");
    state.swap(host_result);
    before_rotation.swap(host_before);
}

void cuda_couple(const std::vector<CouplingInput>& input, double dt, double c,
    double reduced_c_factor, bool marshak, std::vector<CouplingOutput>& output)
{
    if (input.empty()) {
        output.clear();
        return;
    }
    std::vector<CouplingOutput> host_result(input.size());
    Lease lease;
    auto& workspace = lease.get();
    auto& device_input = workspace.input;
    auto& result = workspace.output;
    auto& error = workspace.error;
    const auto stream = workspace.stream.value;
    device_input.reserve(input.size());
    result.reserve(input.size());
    error.reserve(1);
    device_input.upload(input.data(), input.size(), stream);
    error.zero(1, stream);
    couple_kernel<<<blocks(input.size()), threads, 0, stream>>>(device_input.data, result.data,
        input.size(), dt, c, reduced_c_factor, marshak, error.data);
    checked(cudaGetLastError(), "coupling launch");
    finish(workspace);
    result.download(host_result.data(), host_result.size(), stream);
    checked(cudaStreamSynchronize(stream), "coupling download");
    output.swap(host_result);
}

bool cuda_couple_implicit(const std::vector<CouplingInput>& input, double dt, double c,
    double reduced_c_factor, bool marshak, std::vector<CouplingOutput>& output)
{
    if (input.empty()) {
        output.clear();
        return true;
    }
    std::vector<CouplingOutput> host_result(input.size());
    Lease lease;
    auto& workspace = lease.get();
    auto& device_input = workspace.input;
    auto& result = workspace.output;
    auto& error = workspace.error;
    const auto stream = workspace.stream.value;
    device_input.reserve(input.size());
    result.reserve(input.size());
    error.reserve(1);
    device_input.upload(input.data(), input.size(), stream);
    error.zero(1, stream);
    couple_implicit_kernel<<<blocks(input.size()), threads, 0, stream>>>(device_input.data, result.data,
        input.size(), dt, c, reduced_c_factor, marshak, error.data);
    checked(cudaGetLastError(), "implicit coupling launch");
    if (!valid(workspace)) {
        return false;
    }
    result.download(host_result.data(), host_result.size(), stream);
    checked(cudaStreamSynchronize(stream), "implicit coupling download");
    output.swap(host_result);
    return true;
}
} } // namespace radiation_accelerator::detail
