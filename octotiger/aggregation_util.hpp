
#pragma once
#include <hpx/futures/future.hpp>
#ifdef OCTOTIGER_HAVE_KOKKOS
//#define KOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION
#include <hpx/kokkos/executors.hpp>
#include <Kokkos_Core.hpp>
#include <hpx/kokkos.hpp>

#include <stream_manager.hpp>
#include <aggregation_manager.hpp>

// ============================================================ 
// Aggregation Helpers // TODO(daissgr) Move to cppuddle?
// ============================================================ 
//
#ifdef __NVCC__
#include <cuda/std/tuple>
#if defined(HPX_CUDA_VERSION) && (HPX_CUDA_VERSION < 1202)
// cuda::std::tuple structured bindings are broken in CUDA < 1202
// See https://github.com/NVIDIA/libcudacxx/issues/316
// According to https://github.com/NVIDIA/libcudacxx/pull/317 the fix for this 
// is to move tuple element and tuple size into the std namespace
// which the following snippet does. This is only necessary for old CUDA versions
// the newer ones contain a fix for this issue
namespace std {
    template<size_t _Ip, class... _Tp>
    struct tuple_element<_Ip, _CUDA_VSTD::tuple<_Tp...>> 
      : _CUDA_VSTD::tuple_element<_Ip, _CUDA_VSTD::tuple<_Tp...>> {};
    template <class... _Tp>
    struct tuple_size<_CUDA_VSTD::tuple<_Tp...>> 
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>> {};
}
#endif
#endif
static const char hydro_kokkos_kernel_identifier[] = "hydro_kernel_aggregator_kokkos";
template<typename executor_t>
using hydro_kokkos_agg_executor_pool = aggregation_pool<hydro_kokkos_kernel_identifier, executor_t,
                                       round_robin_pool<executor_t>>;

template <typename Agg_view_t>
CUDA_GLOBAL_METHOD typename Agg_view_t::view_type get_slice_subview(
    const size_t slice_id, const size_t max_slices, const Agg_view_t& agg_view) {
    const size_t slice_size = agg_view.size() / max_slices;
    return Kokkos::subview(agg_view,
        std::make_pair<size_t, size_t>(slice_id * slice_size, (slice_id + 1) * slice_size));
}

template <typename Integer, std::enable_if_t<std::is_integral<Integer>::value, bool> = true,
    typename Agg_view_t, typename... Args>
CUDA_GLOBAL_METHOD auto map_views_to_slice(const Integer slice_id, const Integer max_slices,
    const Agg_view_t& current_arg, const Args&... rest) {
    static_assert(
        Kokkos::is_view<typename Agg_view_t::view_type>::value, "Argument not an aggregated view");
#if defined(HPX_COMPUTE_DEVICE_CODE) && defined(__NVCC__)
    if constexpr (sizeof...(Args) > 0) {
        return cuda::std::tuple_cat(cuda::std::make_tuple(get_slice_subview(slice_id, max_slices, current_arg)),
            map_views_to_slice(slice_id, max_slices, rest...));
    } else {
        return cuda::std::make_tuple(get_slice_subview(slice_id, max_slices, current_arg));
    }
#else
    if constexpr (sizeof...(Args) > 0) {
        return std::tuple_cat(std::make_tuple(get_slice_subview(slice_id, max_slices, current_arg)),
            map_views_to_slice(slice_id, max_slices, rest...));
    } else {
        return std::make_tuple(get_slice_subview(slice_id, max_slices, current_arg));
    }
#endif
}

template <typename Agg_executor_t, typename Agg_view_t, std::enable_if_t<Kokkos::is_view<typename Agg_view_t::view_type>::value, bool> = true, typename... Args>
CUDA_GLOBAL_METHOD auto map_views_to_slice(const Agg_executor_t& agg_exec, const Agg_view_t& current_arg,
    const Args&... rest) {
    const size_t slice_id = agg_exec.id;
    const size_t max_slices = opts().max_kernels_fused;
    static_assert(
        Kokkos::is_view<typename Agg_view_t::view_type>::value, "Argument not an aggregated view");
    if constexpr (sizeof...(Args) > 0) {
        return std::tuple_cat(std::make_tuple(get_slice_subview(slice_id, max_slices, current_arg)),
            map_views_to_slice(agg_exec, rest...));
    } else {
        return std::make_tuple(get_slice_subview(slice_id, max_slices, current_arg));
    }
}

template <typename Agg_executor_t, typename TargetView_t, typename SourceView_t>
void aggregated_deep_copy(Agg_executor_t& agg_exec, TargetView_t& target, SourceView_t& source) {
    if (agg_exec.sync_aggregation_slices()) {
        Kokkos::deep_copy(agg_exec.get_underlying_executor().instance(), target, source);
    }
}

template <typename Agg_executor_t, typename TargetView_t, typename SourceView_t>
void aggregated_deep_copy(
    Agg_executor_t& agg_exec, TargetView_t& target, SourceView_t& source, int elements_per_slice) {
    if (agg_exec.sync_aggregation_slices()) {
        const size_t number_slices = agg_exec.number_slices;
        auto target_slices = Kokkos::subview(
            target, std::make_pair<size_t, size_t>(0, number_slices * elements_per_slice));
        auto source_slices = Kokkos::subview(
            source, std::make_pair<size_t, size_t>(0, number_slices * elements_per_slice));
        Kokkos::deep_copy(
            agg_exec.get_underlying_executor().instance(), target_slices, source_slices);
    }
}

template <typename executor_t, typename TargetView_t, typename SourceView_t>
hpx::shared_future<void> aggregrated_deep_copy_async(
    typename Aggregated_Executor<executor_t>::Executor_Slice& agg_exec, TargetView_t& target,
    SourceView_t& source) {
    const size_t gpu_id = agg_exec.parent.gpu_id;
    auto launch_copy_lambda = [gpu_id](TargetView_t& target, SourceView_t& source,
                                  executor_t& exec) -> hpx::shared_future<void> {
        stream_pool::select_device<executor_t,
              round_robin_pool<executor_t>>(gpu_id);
        return hpx::kokkos::deep_copy_async(exec.instance(), target, source);
    };
    return agg_exec.wrap_async(
        launch_copy_lambda, target, source, agg_exec.get_underlying_executor());
}

template <typename executor_t, typename TargetView_t, typename SourceView_t>
hpx::shared_future<void> aggregrated_deep_copy_async(
    typename Aggregated_Executor<executor_t>::Executor_Slice& agg_exec, TargetView_t& target,
    SourceView_t& source, int elements_per_slice) {
    const size_t number_slices = agg_exec.number_slices;
    const size_t gpu_id = agg_exec.parent.gpu_id;
    auto launch_copy_lambda = [gpu_id, elements_per_slice, number_slices](TargetView_t& target,
                                  SourceView_t& source,
                                  executor_t& exec) -> hpx::shared_future<void> {
        stream_pool::select_device<executor_t,
              round_robin_pool<executor_t>>(gpu_id);
        auto target_slices = Kokkos::subview(
            target, std::make_pair<size_t, size_t>(0, number_slices * elements_per_slice));
        auto source_slices = Kokkos::subview(
            source, std::make_pair<size_t, size_t>(0, number_slices *
              elements_per_slice));
        return hpx::kokkos::deep_copy_async(exec.instance(), target_slices, source_slices);
    };
    return agg_exec.wrap_async(
        launch_copy_lambda, target, source, agg_exec.get_underlying_executor());
}
#endif
