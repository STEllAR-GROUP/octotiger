#pragma once

#include "kernel_simd_types.hpp"

// #include "interaction_constants.hpp"

namespace octotiger {
namespace fmm {

    // component type has to be indexable, and has to have a size() operator
    template <typename AoS_type, typename component_type, size_t num_components, size_t entries,
        size_t padding>
    class struct_of_array_data
    {
    private:
        // data in SoA form

        component_type* const data;

    public:
        static constexpr size_t padded_entries_per_component = entries + padding;
        inline component_type* const get_pod(void) {
            return data;
        }
        template <size_t component_access>
        inline component_type* pointer(const size_t flat_index) const {
            constexpr size_t component_array_offset =
                component_access * padded_entries_per_component;
            // should result in single move instruction, indirect addressing: reg + reg +
            // constant
            return data + flat_index + component_array_offset;
        }

        // careful, this returns a copy!
        template <size_t component_access>
        inline m2m_vector value(const size_t flat_index) const {
            return m2m_vector(
                this->pointer<component_access>(flat_index), Vc::flags::element_aligned);
        }

        // template <size_t component_access>
        // inline component_type& reference(const size_t flat_index) const {
        //     return *this->pointer<component_access>(flat_index);
        // }
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

        struct_of_array_data(const std::vector<AoS_type>& org)
          : data(new component_type[num_components * padded_entries_per_component]) {
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    data[component * padded_entries_per_component + entry] = org[entry][component];
                }
            }
        }
        struct_of_array_data(void)
          : data(new component_type[num_components * padded_entries_per_component]) {
            for (auto i = 0; i < num_components * padded_entries_per_component; ++i) {
                data[i] = 0.0;
            }
        }

        struct_of_array_data(const size_t entries_per_component)
          : data(new component_type[num_components * padded_entries_per_component]) {}

        ~struct_of_array_data() {
            delete[] data;
        }

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

      void print(std::ofstream &out) {
            for (size_t entry = 0; entry < padded_entries_per_component; entry++) {
                for (size_t component = 0; component < num_components; component++) {
                  out << data[component * padded_entries_per_component + entry] << std::endl;
                }
                out << std::endl;
            }
        }
    };
}    // namespace fmm
}    // namespace octotiger
