/*
 * index_map.hpp
 *
 *  Created on: Feb 14, 2026
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_INDEX_MAP_HPP_
#define OCTOTIGER_INDEX_MAP_HPP_


/*AαΑBβΒGγΓDδΔEεΕZζΖHηΗThθΘIιΙKκΚLλΛMμΜNνΝXξΞOοΟPπΠRρΡSσΣTτΤYυΥFφΦChχΧPsψΨOωΩ*/
/*₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵨ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᵅᵝᵞᵟᵋᵠᵡ*/

template <size_t D, size_t V>
consteval auto const_array() {
    std::array<size_t, D> A;
    A.fill(V);
    return A;
}

template <size_t D, size_t V, size_t I, size_t E>
consteval auto const_array_with_exception() {
    std::array<size_t, D> A;
    A.fill(V);
    A[I] = E;
    return A;
}

template <size_t D, std::array<size_t, D> N, std::array<size_t, D> B, std::array<size_t, D> E>
consteval auto create_index_map() {
    constexpr size_t eV = std::accumulate(N.begin(), N.end(), 1, std::multiplies<size_t>{});
    constexpr size_t iV = std::transform_reduce(
        E.begin(), E.end(), B.begin(), 1, std::multiplies<size_t>{}, std::minus<size_t>{});
    std::array<size_t, iV> map{};
    std::array<size_t, D> idx{};
    idx = B;
    size_t j = 0;
    do {
        size_t i = 0;
        for (size_t d = 0; d < D; d++) {
            i = N[d] * i + idx[d];
        }
        map[j++] = i;
        for (int d = D - 1; d >= 0; d--) {
            if (++idx[d] < E[d]) {
                break;
            }
            idx[d] = B[d];
        }
    } while (idx != B);
    return map;
}

template <size_t D, size_t N, size_t B, size_t E>
consteval auto create_index_map() {
    return create_index_map<D, const_array<D, N>(), const_array<D, B>(), const_array<D, E>()>();
}

template <size_t D, size_t N, std::array<size_t, D> B, std::array<size_t, D> E>
consteval auto create_index_map() {
    return create_index_map<D, const_array<D, N>(), B, E>();
}



#endif /* OCTOTIGER_INDEX_MAP_HPP_ */
