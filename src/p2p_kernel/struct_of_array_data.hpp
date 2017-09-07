#pragma once

#include "m2m_simd_types.hpp"

// #include "interaction_constants.hpp"

namespace octotiger {
namespace fmm {
namespace p2p_kernel {

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
            // should result in single move instruction, indirect addressing: reg + reg + constant
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
                    data[component * padded_entries_per_component + entry] = org[entry][component];
                }
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
    };

    // template <typename AoS_type, typename component_type, size_t num_components>
    // class struct_of_array_data;

    // template <typename AoS_type, typename component_type, size_t num_components>
    // class struct_of_array_iterator;

    // template <typename AoS_type, typename component_type, size_t num_components>
    // class struct_of_array_view
    // {
    // protected:
    //     struct_of_array_data<AoS_type, component_type, num_components>& data;
    //     size_t flat_index;

    // public:
    //     struct_of_array_view<AoS_type, component_type, num_components>(
    //         struct_of_array_data<AoS_type, component_type, num_components>& data, size_t
    //         flat_index)
    //       : data(data)
    //       , flat_index(flat_index) {}

    //     inline m2m_vector component(size_t component_index) const {
    //         return m2m_vector(this->component_pointer(component_index),
    //         Vc::flags::element_aligned);
    //     }

    //     // returning pointer so that Vc can convert into simdarray
    //     // (notice: a reference would be silently broadcasted)
    //     inline component_type* component_pointer(size_t component_index) const {
    //         return data.access(component_index, flat_index);
    //     }
    // };

    // // component type has to be indexable, and has to have a size() operator
    // template <typename AoS_type, typename component_type, size_t num_components>
    // class struct_of_array_data
    // {
    // private:
    //     // data in SoA form
    //     std::vector<component_type> data;
    //     const size_t padded_entries_per_component;

    //     friend class struct_of_array_iterator<AoS_type, component_type, num_components>;

    // public:
    //     struct_of_array_data(const std::vector<AoS_type>& org)
    //       : padded_entries_per_component(org.size() + SOA_PADDING) {
    //         data = std::vector<component_type>(num_components * (org.size() + SOA_PADDING));
    //         for (size_t component = 0; component < num_components; component++) {
    //             for (size_t entry = 0; entry < org.size(); entry++) {
    //                 data[component * padded_entries_per_component + entry] =
    //                 org[entry][component];
    //             }
    //         }
    //     }

    //     struct_of_array_data(const size_t entries_per_component)
    //       : padded_entries_per_component(entries_per_component + SOA_PADDING) {
    //         data = std::vector<component_type>(num_components * padded_entries_per_component);
    //     }

    //     inline component_type* access(const size_t component_index, const size_t
    //     flat_entry_index) {
    //         return data.data() +
    //             (component_index * padded_entries_per_component + flat_entry_index);
    //     }

    //     struct_of_array_view<AoS_type, component_type, num_components> get_view(size_t
    //     flat_index) {
    //         return struct_of_array_view<AoS_type, component_type, num_components>(
    //             *this, flat_index);
    //     }

    //     // write back into non-SoA style array
    //     void to_non_SoA(std::vector<AoS_type>& org) {
    //         for (size_t component = 0; component < num_components; component++) {
    //             for (size_t entry = 0; entry < org.size(); entry++) {
    //                 org[entry][component] = data[component * padded_entries_per_component +
    //                 entry];
    //             }
    //         }
    //     }
    // };

    // template <typename AoS_type, typename component_type, size_t num_components>
    // class struct_of_array_iterator
    // {
    // private:
    //     component_type* current;
    //     size_t component_offset;

    // public:
    //     struct_of_array_iterator(
    //         struct_of_array_data<AoS_type, component_type, num_components>& data,
    //         size_t flat_index) {
    //         current = data.data.data() + flat_index;
    //         component_offset = data.padded_entries_per_component;
    //     }

    //     inline component_type* pointer() {
    //         return current;
    //     }

    //     // int is dummy parameter
    //     inline void operator++(int) {
    //         current += component_offset;
    //     }
    //     inline void increment(size_t num) {
    //         current += component_offset * num;
    //     }
    //     inline void decrement(size_t num) {
    //         current -= component_offset * num;
    //     }

    //     inline m2m_vector value() {
    //         return m2m_vector(this->pointer(), Vc::flags::element_aligned);
    //     }
    // };

}    // namespace p2p_kernel
}    // namespace fmm
}    // namespace octotiger
