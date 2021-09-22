//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

//#include "kernel_simd_types.hpp"
#include "../taylor.hpp"
#include "../space_vector.hpp"

#ifdef OCTOTIGER_HAVE_CUDA
#include "../cuda_util/cuda_helper.hpp"
#include <cuda_buffer_util.hpp>
#elif OCTOTIGER_HAVE_HIP
#include "../cuda_util/cuda_helper.hpp"
#include <hip_buffer_util.hpp>
#endif
#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>


#include "interaction_constants.hpp"

namespace octotiger {
namespace fmm {

    // component type has to be indexable, and has to have a size() operator
    template <typename AoS_type, typename component_type, size_t num_components, size_t entries,
        size_t padding, typename backend_t = std::vector<component_type>>
    class struct_of_array_data
    {
    private:
        /// data in SoA form - usually a vector (with optional custom-allocator)
        backend_t data;

    public:
        static constexpr size_t padded_entries_per_component = entries + padding;
        inline component_type* const get_pod() {
            return data.data();
        }
        template <size_t component_access>
        inline component_type* pointer(const size_t flat_index) {
            constexpr size_t component_array_offset =
                component_access * padded_entries_per_component;
            // should result in single move instruction, indirect addressing: reg + reg +
            // constant
            return data.data() + flat_index + component_array_offset;
        }
        inline component_type& operator [] (const size_t index) {
            return data[index];
        }

        // careful, this returns a copy!
        template <size_t component_access, typename VectorType>
        inline VectorType value(const size_t flat_index) const {
            constexpr size_t component_array_offset =
                component_access * padded_entries_per_component;
            return VectorType(data.data() + flat_index + component_array_offset);
        }
        template <size_t component_access>
        inline component_type at(const size_t flat_index) const {
            constexpr size_t component_array_offset =
                component_access * padded_entries_per_component;
            return data[flat_index + component_array_offset];
        }

        template <typename AoS_temp_type>
        void set_AoS_value(AoS_temp_type&& value, size_t flatindex) {
            for (size_t component = 0; component < num_components; component++) {
                data[component * padded_entries_per_component + flatindex] = value[component];
            }
        }
        template <typename AoS_temp_type>
        inline void set_value(AoS_temp_type&& value, size_t flatindex) {
            data[flatindex] = value;
            for (size_t component = 1; component < num_components; component++) {
                data[component * padded_entries_per_component + flatindex] = 0.0;
            }
        }

        template <typename T>
        void concatenate_vectors(std::vector<std::vector<T>>& input) {
            size_t result_size = input.size() * input[0].size();
            auto iter = data.begin();
            for (size_t i = 0; i < input.size(); i++) {
                iter = std::copy(input[i].begin(), input[i].end(), iter);
            }
        }

        explicit struct_of_array_data(const std::vector<AoS_type>& org)
          : data(num_components * padded_entries_per_component) {
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    data[component * padded_entries_per_component + entry] = org[entry][component];
                }
            }
        }
        struct_of_array_data()
          : data(num_components * padded_entries_per_component) {
            for (auto i = 0; i < num_components * padded_entries_per_component; ++i) {
                data[i] = 0.0;
            }
        }

        ~struct_of_array_data() {}

        struct_of_array_data(const struct_of_array_data& other) = delete;

        struct_of_array_data(const struct_of_array_data&& other) = delete;

        struct_of_array_data& operator=(const struct_of_array_data& other) = delete;

        // write back into non-SoA style array
        void to_non_SoA(std::vector<AoS_type>& org) {
            // constexpr size_t padded_entries_per_component = entries + padding;
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    org[entry][component] = data[component * padded_entries_per_component + entry];
                }
            }
        }
        void add_to_non_SoA(std::vector<AoS_type>& org) {
            // constexpr size_t padded_entries_per_component = entries + padding;
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    org[entry][component] += data[component * padded_entries_per_component + entry];
                }
            }
        }

        void print(std::ostream& out, size_t number_entries = padded_entries_per_component) {
            for (size_t entry = 0; entry < number_entries; entry++) {
                for (size_t component = 0; component < num_components; component++) {
                    out << component << ": "
                        << data[component * padded_entries_per_component + entry] << " "
                        << std::endl;
                }
                out << std::endl;
            }
        }
        bool all_zero(std::ostream& out, size_t number_entries = padded_entries_per_component) {
            for (size_t entry = 0; entry < number_entries; entry++) {
                for (size_t component = 0; component < num_components; component++) {
                    if (std::abs(data[component * padded_entries_per_component + entry]) > 1e-18) {
                    out << component << " / " << entry << " : " 
                        << data[component * padded_entries_per_component + entry] << " "
                        << std::endl;
                        return false;
                    }
                }
            }
            return true;
        }
        void compare(std::ostream& out, std::vector<AoS_type>& org_copy, std::vector<AoS_type>& org, size_t number_entries = padded_entries_per_component) {
            for (size_t entry = 0; entry < 16; entry++) {
                for (size_t component = 0; component < num_components; component++) {
                    out << component << ": " << 
                        data[component * padded_entries_per_component + entry] << " " << data[component * padded_entries_per_component + entry] + org_copy[entry][component] << " " << org[entry][component] 
                        << std::endl;
                }
                out << std::endl;
            }
        }
    };

    constexpr uint64_t SIMD_LENGTH_BYTES = 32;

    using cpu_expansion_buffer_t = struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING,
        std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>;
    using cpu_space_vector_buffer_t =
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING,
            std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>;
    using cpu_expansion_result_buffer_t =
        struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING,
            std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>;
    using cpu_angular_result_t =
        struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
            std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>;
    using cpu_monopole_buffer_t =
        std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>;

#ifdef OCTOTIGER_HAVE_CUDA
    using cuda_expansion_buffer_t = struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING,
        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>;
    using cuda_space_vector_buffer_t =
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING,
            std::vector<real, recycler::recycle_allocator_cuda_host<real>>>;
    using cuda_expansion_result_buffer_t =
        struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING,
            std::vector<real, recycler::recycle_allocator_cuda_host<real>>>;
    using cuda_angular_result_t =
        struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
            std::vector<real, recycler::recycle_allocator_cuda_host<real>>>;
    using cuda_monopole_buffer_t =
        std::vector<real, recycler::recycle_allocator_cuda_host<real>>;
#elif OCTOTIGER_HAVE_HIP
    using cuda_expansion_buffer_t = struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING,
        std::vector<real, recycler::recycle_allocator_hip_host<real>>>;
    using cuda_space_vector_buffer_t =
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING,
            std::vector<real, recycler::recycle_allocator_hip_host<real>>>;
    using cuda_expansion_result_buffer_t =
        struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING,
            std::vector<real, recycler::recycle_allocator_hip_host<real>>>;
    using cuda_angular_result_t =
        struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
            std::vector<real, recycler::recycle_allocator_hip_host<real>>>;
    using cuda_monopole_buffer_t =
        std::vector<real, recycler::recycle_allocator_hip_host<real>>;
#endif

}    // namespace fmm
}    // namespace octotiger
