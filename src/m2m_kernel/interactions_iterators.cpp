#include "interactions_iterators.hpp"

namespace octotiger {
namespace fmm {

    bool expansion_comparator(const expansion& ref, const expansion& mine) {
        if (ref.size() != mine.size()) {
            std::cout << "size of expansion doesn't match" << std::endl;
            return false;
        }
        for (size_t i = 0; i < mine.size(); i++) {
            if (std::abs(ref[i] - mine[i]) >= 10000.0 * std::numeric_limits<real>::epsilon()) {
                std::cout << "error: index padded: " << i << ", mine[" << i << "] != ref[" << i
                          << "] <=> " << mine[i] << " != " << ref[i] << ", "
                          << std::abs(ref[i] - mine[i])
                          << " >= " << 1000.0 * std::numeric_limits<real>::epsilon() << std::endl;
                return false;
            }
        }
        return true;
    }

    bool space_vector_comparator(const space_vector& ref, const space_vector& mine) {
        for (size_t i = 0; i < mine.size(); i++) {
            if (std::abs(ref[i] - mine[i]) >= 10000.0 * std::numeric_limits<real>::epsilon()) {
                std::cout << "error: index padded: " << i << ", mine[" << i << "] != ref[" << i
                          << "] <=> " << mine[i] << " != " << ref[i] << ", "
                          << std::abs(ref[i] - mine[i])
                          << " >= " << 1000.0 * std::numeric_limits<real>::epsilon() << std::endl;
                return false;
            }
        }
        return true;
    }

}    // namespace fmm
}    // namespace octotiger
