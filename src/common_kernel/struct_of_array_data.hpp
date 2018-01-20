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
            static constexpr size_t padded_entries_per_component = entries + padding;

            component_type* const data;

        public:
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

            struct_of_array_data(const std::vector<AoS_type>& org)
              : data(new component_type[num_components * padded_entries_per_component]) {
                for (size_t component = 0; component < num_components; component++) {
                    for (size_t entry = 0; entry < org.size(); entry++) {
                        data[component * padded_entries_per_component + entry] =
                            org[entry][component];
                    }
                }
            }
            struct_of_array_data(void)
              : data(new component_type[num_components * padded_entries_per_component]) {
                for (auto i = 0; i < num_components * padded_entries_per_component; ++i) {
                    data[i] = 0;
                }

            }

            struct_of_array_data(const size_t entries_per_component)
              : data(new component_type[num_components * padded_entries_per_component]) {}

            ~struct_of_array_data() {
                delete[] data;
            }

            void update_data(const std::vector<AoS_type>& org) {
                for (size_t component = 0; component < num_components; component++) {
                    for (size_t entry = 0; entry < org.size(); entry++) {
                        data[component * padded_entries_per_component + entry] =
                            org[entry][component];
                    }
                }
            }

            struct_of_array_data(const struct_of_array_data& other) = delete;

            struct_of_array_data(const struct_of_array_data&& other) = delete;

            struct_of_array_data& operator=(const struct_of_array_data& other) = delete;

            // write back into non-SoA style array
            void to_non_SoA(std::vector<AoS_type>& org) {
                // constexpr size_t padded_entries_per_component = entries + padding;
                for (size_t component = 0; component < num_components; component++) {
                    for (size_t entry = 0; entry < org.size(); entry++) {
                        org[entry][component] =
                            data[component * padded_entries_per_component + entry];
                    }
                }
            }
        };
}    // namespace fmm
}    // namespace octotiger
