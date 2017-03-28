// Convention: i is the dimension with the least stride.
// Left/Fortran/Column-Major Layout : L[i][j]
// Right/C++/Row-Major Layout       : L[j][i] 

// TODO: FUSION/FISSION
// TODO: LOOP INVARIANT CODE MOTION
// TODO: POST-VECTORIZATION UNROLLING 
// TODO: Split into two phases
// TODO: Fix SoA in grid
// TODO: Implement set_basis

inline void grid::compute_interactions_initialize_L_c(std::true_type) noexcept
{
    // TODO: PERF MODEL

    #if defined(HPX_HAVE_DATAPAR)
        hpx::parallel::fill(hpx::parallel::execution::dataseq,
                            L_c.begin(), L_c.end(), ZERO);
    #else
        std::fill(std::begin(L_c), std::end(L_c), ZERO);
    #endif
}

inline void grid::compute_interactions_initialize_L_c(std::false_type) noexcept {}

///////////////////////////////////////////////////////////////////////////////
        
template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_initialize_n_ang_mom(
    integer i_begin
  , integer i_end
  , taylor<4, std::array<real, TileWidth>>& n0
  , taylor<4, std::array<real, TileWidth>>& n1
  , std::true_type
    ) noexcept
{
    auto& M = *M_ptr;

    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)                // TRIP COUNT: 10
        //#pragma GCC ivdep
        #pragma omp simd
        for (integer i = i_begin; i < i_end; ++i)                               // TRIP COUNT: TileWidth CONTROLLED; UNIT STRIDE
        {
            integer const ti = i - i_begin;

            integer const iii0 = (*IList)[i].first;                             // 1 INT LOAD FROM MEM (indirect addressing)
            integer const iii1 = (*IList)[i].second;                            // 1 INT LOAD FROM MEM (indirect addressing)

            auto const M00 = M[iii0]();                                         // 1 FP LOAD FROM MEM
            auto const M10 = M[iii1]();                                         // 1 FP LOAD FROM MEM

            // NOTE: Due to the order of iteration, these computations are
            // performed redundantly.
            auto const M10divM00 = M10 / M00;                                   // 1 FP DIV
            auto const M00divM10 = M00 / M10;                                   // 1 FP DIV

            auto const M0j = M[iii0]();                                         // 1 FP LOAD FROM MEM
            auto const M1j = M[iii1]();                                         // 1 FP LOAD FROM MEM

            n0[j][ti] = M1j - M0j * M10divM00;                                  // 1 FMA, 1 STORE TO TILE
            n1[j][ti] = M0j - M1j * M00divM10;                                  // 1 FMA, 1 STORE TO TILE
        }
}

template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_initialize_n_ang_mom(
    integer i_begin
  , integer i_end
  , taylor<4, std::array<real, TileWidth>>& n0
  , taylor<4, std::array<real, TileWidth>>& n1
  , std::false_type
    ) noexcept
{
    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)                // TRIP COUNT: 10
        #pragma GCC ivdep
        for (integer i = i_begin; i != i_end; ++i)                              // TRIP COUNT: TileWidth CONTROLLED; UNIT STRIDE
        {
            integer const ti = i - i_begin;

            n0[j][ti] = ZERO;                                                   // 1 FP STORE TO TILE
            n1[j][ti] = ZERO;                                                   // 1 FP STORE TO TILE
        }
}

///////////////////////////////////////////////////////////////////////////////

template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_A0_A1_0(
    taylor<4, std::array<real, TileWidth>>& A0
  , taylor<4, std::array<real, TileWidth>>& A1
  , taylor<4, std::array<real, TileWidth>> const& m0
  , taylor<4, std::array<real, TileWidth>> const& m1
  , taylor<5, std::array<real, TileWidth>> const& D
  , std::true_type
    ) noexcept
{}

template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_A0_A1_0(
    taylor<4, std::array<real, TileWidth>>& A0
  , taylor<4, std::array<real, TileWidth>>& A1
  , taylor<4, std::array<real, TileWidth>> const& m0
  , taylor<4, std::array<real, TileWidth>> const& m1
  , taylor<5, std::array<real, TileWidth>> const& D
  , std::false_type
    ) noexcept
{
    for (integer j = taylor_sizes[0]; j != taylor_sizes[1]; ++j)                // TRIP COUNT: 3
        #pragma GCC ivdep
        for (integer ti = 0; ti != TileWidth; ++ti)                             // TRIP COUNT: TileWidth; UNIT STRIDE
        {
            A0[0][ti] -= m0[j][ti] * D[j][ti];                                  // 3 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
            A1[0][ti] += m1[j][ti] * D[j][ti];                                  // 3 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
        }
}

///////////////////////////////////////////////////////////////////////////////

template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_B0_B1(
    std::array<std::array<real, TileWidth>, NDIM>& B0 
  , std::array<std::array<real, TileWidth>, NDIM>& B1
  , taylor<4, std::array<real, TileWidth>> const& n0
  , taylor<4, std::array<real, TileWidth>> const& n1
  , taylor<5, std::array<real, TileWidth>> const& D
  , std::true_type
    ) noexcept
{
    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
        for (integer b = 0; b != NDIM; ++b)                                     // TRIP COUNT: 9
            for (integer c = b; c != NDIM; ++c)                                 // TRIP COUNT: ?
                for (integer d = c; d != NDIM; ++d)                             // TRIP COUNT: ?
                    #pragma GCC ivdep
                    for (integer ti = 0; ti != TileWidth; ++ti)                 // TRIP COUNT: TileWidth; UNIT STRIDE
                    {
                        auto const tmp = D(a, b, c, d)[ti]
                                       * sixth_factor(b, c, d);                 // 1 FP LOAD FROM TILE, 1 FP MUL
                        B0[a][ti] -= n0(b, c, d)[ti] * tmp;                     // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
                        B1[a][ti] -= n1(b, c, d)[ti] * tmp;                     // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
                    }
}

template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_B0_B1(
    std::array<std::array<real, TileWidth>, NDIM>& B0 
  , std::array<std::array<real, TileWidth>, NDIM>& B1
  , taylor<4, std::array<real, TileWidth>> const& n0
  , taylor<4, std::array<real, TileWidth>> const& n1
  , taylor<5, std::array<real, TileWidth>> const& D
  , std::false_type
    ) noexcept
{}

///////////////////////////////////////////////////////////////////////////////

template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
    >
inline void grid::store_to_L_c(
    integer i_begin
  , integer i_end
  , std::array<std::array<real, TileWidth>, NDIM>& B0 
  , std::array<std::array<real, TileWidth>, NDIM>& B1
  , std::true_type
    ) noexcept
{
    for (integer j = 0; j != NDIM; ++j)                                         // TRIP COUNT: 3
        #pragma GCC ivdep
        for (integer i = i_begin; i != i_end; ++i)                              // TRIP COUNT: TileWidth CONTROLLED; UNIT STRIDE
        {
            integer const ti = i_begin - i;

            integer const iii0 = (*IList)[i].first;                             // 1 INT LOAD FROM MEM (indirect addressing)
            integer const iii1 = (*IList)[i].second;                            // 1 INT LOAD FROM MEM (indirect addressing)

            L_c[iii0][j] += B0[j][ti];                                           // 1 FP LOAD FROM CACHE, 1 FP LOAD FROM TILE, 1 FP STORE TO MEM
            L_c[iii1][j] += B1[j][ti];                                           // 1 FP LOAD FROM CACHE, 1 FP LOAD FROM TILE, 1 FP STORE TO MEM
        }
}

template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
    >
inline void grid::store_to_L_c(
    integer i_begin
  , integer i_end
  , std::array<std::array<real, TileWidth>, NDIM>& B0 
  , std::array<std::array<real, TileWidth>, NDIM>& B1
  , std::false_type
    ) noexcept
{}

///////////////////////////////////////////////////////////////////////////////

template <std::size_t TileWidth>
inline taylor<5, std::array<real, TileWidth>>
set_basis(
    taylor<5, std::array<real, TileWidth>>& A
  , std::array<std::array<real, TileWidth>, NDIM> const& X
    ) noexcept
{
    // TODO: Perf model.

    std::array<real, TileWidth> d0, d1, d2, d3;

    #pragma GCC ivdep
    for (integer ti = 0; ti != TileWidth; ++ti)                                 // TRIP COUNT: TileWidth; UNIT STRIDE
    {
        real const r2 = sqr(X[0][ti]) + sqr(X[1][ti]) + sqr(X[2][ti]);

        real const r2inv = (r2 > 0.0) ? ONE / std::max(r2, 1.0e-20) : ZERO;

        d0[ti] = -sqrt(r2inv);
        d1[ti] = -d0[ti] * r2inv;
        d2[ti] = real(-3) * d1[ti] * r2inv;
        d3[ti] = real(-5) * d2[ti] * r2inv;
    }

    #pragma GCC ivdep
    for (integer ti = 0; ti != TileWidth; ++ti)                                 // TRIP COUNT: TileWidth; UNIT STRIDE
        A[0][ti] = d0[ti];

    for (integer j = taylor_sizes[0], a = 0; a != NDIM; ++a, ++j)
        #pragma GCC ivdep
        for (integer ti = 0; ti != TileWidth; ++ti)                             // TRIP COUNT: TileWidth; UNIT STRIDE
            A[j][ti] = X[a][ti] * d1[ti];

    for (integer j = taylor_sizes[1], a = 0; a != NDIM; ++a)
        for (integer b = a; b != NDIM; ++b, ++j)
            #pragma GCC ivdep
            for (integer ti = 0; ti != TileWidth; ++ti)                         // TRIP COUNT: TileWidth; UNIT STRIDE
            {
                A[j][ti] = X[a][ti] * X[b][ti] * d2[ti];
            }

    for (integer j = taylor_sizes[2], a = 0; a != NDIM; ++a)
        for (integer b = a; b != NDIM; ++b) 
            for (integer c = b; c != NDIM; ++c, ++j) 
                #pragma GCC ivdep
                for (integer ti = 0; ti != TileWidth; ++ti)                     // TRIP COUNT: TileWidth; UNIT STRIDE
                {
                    A[j][ti] = X[a][ti] * X[b][ti] * X[c][ti] * d3[ti];
                }

    for (integer j = taylor_sizes[3]; j != taylor_sizes[4]; ++j)
        #pragma GCC ivdep
        for (integer ti = 0; ti != TileWidth; ++ti)                             // TRIP COUNT: TileWidth; UNIT STRIDE
        {
            A[j][ti] = ZERO;
        }

    for (integer a = 0; a != NDIM; ++a)
        #pragma GCC ivdep
        for (integer ti = 0; ti != TileWidth; ++ti)                             // TRIP COUNT: TileWidth; UNIT STRIDE
        {
            auto const d2ti = d2[ti]; 
            auto const Xati = X[a][ti];
            A(a, a)[ti]       += d1[ti];
            A(a, a, a)[ti]    += Xati * d2ti;
            A(a, a, a, a)[ti] += Xati * Xati + 2.0 * d2ti;
        }

    for (integer a = 0; a != NDIM; ++a)
        for (integer b = a; b != NDIM; ++b)
            #pragma GCC ivdep
            for (integer ti = 0; ti != TileWidth; ++ti)                         // TRIP COUNT: TileWidth; UNIT STRIDE
            {
                auto const d2ti = d2[ti]; 
                auto const d3ti = d3[ti]; 
                auto const Xati = X[a][ti];
                auto const Xbti = X[b][ti];
                A(a, a, b)[ti]    += Xbti * d2ti;
                A(a, b, b)[ti]    += Xati * d2ti;
                A(a, a, a, b)[ti] += Xati * Xbti * d3ti;
                A(a, b, b, b)[ti] += Xati * Xbti * d3ti;
                A(a, a, b, b)[ti] += d2ti;
            }

    for (integer a = 0; a != NDIM; ++a)
         for (integer b = a; b != NDIM; ++b)
            for (integer c = b; c != NDIM; ++c)
                #pragma GCC ivdep
                for (integer ti = 0; ti != TileWidth; ++ti)                     // TRIP COUNT: TileWidth; UNIT STRIDE
                {
                    auto const d2ti = d2[ti]; 
                    auto const d3ti = d3[ti]; 
                    auto const Xati = X[a][ti];
                    auto const Xbti = X[b][ti];
                    auto const Xcti = X[c][ti];
                    A(a, a, b, c)[ti] += Xati * Xcti * d2ti;
                    A(a, b, b, c)[ti] += Xbti * Xcti * d3ti;
                    A(a, b, c, c)[ti] += Xati * Xbti * d3ti;
                }
}

///////////////////////////////////////////////////////////////////////////////

struct op_stats
{
    integer fp_adds;
    integer fp_fmas;
    integer fp_divs;
    integer fp_memloads; 
    integer fp_memstores; 
    integer fp_cacheloads; 
    integer fp_cachestores; 
};

// Computes the interactions between interior points in non-leaf nodes using
// taylor expansions [David].
template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
  , ang_con_type AngConKind
  , gsolve_type SolveKind
    >
inline void grid::compute_interactions_non_leaf()
{
    static_assert(0 < TileWidth, "TileWidth cannot be negative.");
    static_assert(0 == (TileWidth % 16), "TileWidth must be a multiple of 16.");

    auto& M = *M_ptr;
    std::vector<space_vector> const& com = *(com_ptr[0]);

    std::integral_constant<
        bool, ANG_CON_ON == AngConKind
    > ang_con_is_on{};
    std::integral_constant<
        bool, RHO == SolveKind
    > is_rho_type{};
    std::integral_constant<
        bool, ANG_CON_ON == AngConKind && RHO == SolveKind
    > ang_con_is_on_and_is_rho_type{};

    // L stores the gravitational potential.
    // #10 in the paper ([Bryce] FIXME: Link to the paper) [Dominic].
    #if defined(HPX_HAVE_DATAPAR)
        hpx::parallel::fill(hpx::parallel::execution::dataseq,
                            L.begin(), L.end(), ZERO);
    #else
        std::fill(std::begin(L), std::end(L), ZERO);
    #endif

    // X and Y are the two cells interacting [David].
    // X and Y store the 3D center of masses (per simd element, SoA style) [David].
    // dX is distance between X and Y [David].
    std::array<std::array<real, TileWidth>, NDIM> dX; // 3 * TileWidth FPs

    // m multipole moments of the cells [David].
    taylor<4, std::array<real, TileWidth>> m0; // 20 * TileWidth FPs
    taylor<4, std::array<real, TileWidth>> m1; // 20 * TileWidth FPs
    // n angular momentum of the cells [David].
    taylor<4, std::array<real, TileWidth>> n0; // 20 * TileWidth FPs
    taylor<4, std::array<real, TileWidth>> n1; // 20 * TileWidth FPs

    // R_i in paper is the dX in the code
    // D is taylor expansion value for a given X expansion of the gravitational
    // potential (multipole expansion) [David].
    taylor<5, std::array<real, TileWidth>> D; // 35 * TileWidth FPs

    // A0, A1 are the contributions to L [David].
    taylor<4, std::array<real, TileWidth>> A0; // 20 * TileWidth FPs
    taylor<4, std::array<real, TileWidth>> A1; // 20 * TileWidth FPs

    // B0, B1 are the contributions to L_c (for each cell) [David].
    std::array<std::array<real, TileWidth>, NDIM> B0; // 3 * TileWidth FPs
    std::array<std::array<real, TileWidth>, NDIM> B1; // 3 * TileWidth FPs

    // calculates all D-values, calculate all coefficients of 1/r (not the potential),
    // formula (6)-(9) and (19) [David].

    for (integer i_begin = 0; i_begin != IList->size(); i_begin += TileWidth)
    {
        integer const i_end = i_begin + TileWidth;

        ///////////////////////////////////////////////////////////////////////
        // GATHER 

        for (integer j = 0; j != NDIM; ++j)                                     // TRIP COUNT: 3
            #pragma GCC ivdep
            for (integer i = i_begin; i != i_end; ++i)                          // TRIP COUNT: TileWidth CONTROLLED; UNIT STRIDE
            {
                integer const ti = i - i_begin;

                integer const iii0 = (*IList)[i].first;                         // 1 INT LOAD FROM MEM (indirect addressing)
                integer const iii1 = (*IList)[i].second;                        // 1 INT LOAD FROM MEM (indirect addressing)

                real const x = com.at(iii0)[j];                                    // 1 FP LOAD FROM MEM
                real const y = com.at(iii1)[j];                                    // 1 FP LOAD FROM MEM
                real const d = x - y;                                           // 1 FP ADD
                dX[j][ti] = d;                                                  // 1 FP STORE TO TILE
            }

        for (integer j = 0; j != taylor_sizes[3]; ++j)                          // TRIP COUNT: 20
            #pragma GCC ivdep
            for (integer i = i_begin; i != i_end; ++i)                          // TRIP COUNT: TileWidth CONTROLLED; UNIT STRIDE
            {
                integer const ti = i - i_begin;

                integer const iii0 = (*IList)[i].first;                         // 1 INT LOAD FROM MEM (indirect addressing)
                integer const iii1 = (*IList)[i].second;                        // 1 INT LOAD FROM MEM (indirect addressing)

                m0[j][ti] = M[iii0](j);                                         // 1 FP LOAD FROM MEM, 1 FP STORE TO TILE
                m1[j][ti] = M[iii1](j);                                         // 1 FP LOAD FROM MEM, 1 FP STORE TO TILE
            }

        compute_interactions_initialize_n_ang_mom<IList, TileWidth>(i_begin, i_end, n0, n1, is_rho_type);

        ///////////////////////////////////////////////////////////////////////
        // COMPUTE

        set_basis(D, dX);

        // FIXME: We're going to have problems here with the three back-to-back
        // loops writing to A0/A1[0]. We need to fuse them if possible; use
        // masking if we must. This probably has to be hand coded.

        #pragma GCC ivdep
        for (integer ti = 0; ti != TileWidth; ++ti)                             // TRIP COUNT: TileWidth; UNIT STRIDE
        {
            A0[0][ti] = m0[0][ti] * D[0][ti];                                   // 3 FP LOADS FROM TILE, 1 FP MUL, 1 FP STORE TO TILE
            A1[0][ti] = m1[0][ti] * D[0][ti];                                   // 3 FP LOADS FROM TILE, 1 FP MUL, 1 FP STORE TO TILE
        }

        for (integer j = taylor_sizes[1]; j != taylor_sizes[2]; ++j)            // TRIP COUNT: 6
            #pragma GCC ivdep
            for (integer ti = 0; ti != TileWidth; ++ti)                         // TRIP COUNT: TileWidth; UNIT STRIDE
            {
                auto const tmp1 = D[j][ti] * half_factor[j];                    // 1 FP LOAD FROM TILE, 1 FP MUL
                A0[0][ti] += m0[j][ti] * tmp1;                                  // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
                A1[0][ti] += m1[j][ti] * tmp1;                                  // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
            }

        for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)            // TRIP COUNT: 10
            #pragma GCC ivdep
            for (integer ti = 0; ti != TileWidth; ++ti)                         // TRIP COUNT: TileWidth; UNIT STRIDE
            {
                auto const tmp = D[j][ti] * sixth_factor[j];                    // 1 FP LOAD FROM TILE, 1 FP MUL
                A0[0][ti] -= m0[j][ti] * tmp;                                   // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
                A1[0][ti] += m1[j][ti] * tmp;                                   // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
            }

        for (integer a = 0; a != NDIM; ++a)                                     // TRIP COUNT: 3
            #pragma GCC ivdep
            for (integer ti = 0; ti != TileWidth; ++ti)                         // TRIP COUNT: TileWidth; UNIT STRIDE
            {
                auto const tmp = D(a)[ti];                                      // 1 FP LOAD FROM TILE
                A0(a)[ti] =  m0()[ti] * tmp;                                    // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                A1(a)[ti] = -m1()[ti] * tmp;                                    // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
            }

        compute_interactions_A0_A1_0<IList, TileWidth>(A0, A1, m0, m1, D, is_rho_type); 

        for (integer a = 0; a != NDIM; ++a)                                     // TRIP COUNT: 3
            for (integer b = 0; b != NDIM; ++b)                                 // TRIP COUNT: 9
                for (integer c = b; c != NDIM; ++c)                             // TRIP COUNT: ?
                    #pragma GCC ivdep
                    for (integer ti = 0; ti != TileWidth; ++ti)                 // TRIP COUNT: TileWidth; UNIT STRIDE
                    {
                        auto const tmp = D(a, b, c)[ti] * half_factor(c, b);    // 1 FP LOAD FROM TILE, 1 FP MUL
                        A0(a)[ti] += m0(c, b)[ti] * tmp;                        // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                        A1(a)[ti] -= m1(c, b)[ti] * tmp;                        // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                    }

        compute_interactions_B0_B1<IList, TileWidth>(B0, B1, n0, n1, D, is_rho_type);

        for (integer a = 0; a != NDIM; ++a)                                     // TRIP COUNT: 3
            for (integer b = a; b != NDIM; ++b)                                 // TRIP COUNT: 6
                for (integer ti = 0; ti != TileWidth; ++ti)                     // TRIP COUNT: TileWidth; UNIT STRIDE
                {
                    auto const tmp = D(a, b)[ti];                               // 1 FP LOAD FROM TILE
                    A0(a, b)[ti] = m0()[ti] * tmp;                              // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                    A1(a, b)[ti] = m1()[ti] * tmp;                              // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                }

        for (integer a = 0; a != NDIM; ++a)                                     // TRIP COUNT: 3
            for (integer b = a; b != NDIM; ++b)                                 // TRIP COUNT: 6
                for (integer c = 0; c != NDIM; ++c)                             // TRIP COUNT: ?
                    #pragma GCC ivdep
                    for (integer ti = 0; ti != TileWidth; ++ti)                 // TRIP COUNT: TileWidth; UNIT STRIDE
                    {
                        auto const tmp = D(a, b, c)[ti];                        // 1 FP LOAD FROM TILE
                        A0(a, b)[ti] -= m0(c)[ti] * tmp;                        // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                        A1(a, b)[ti] += m1(c)[ti] * tmp;                        // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                    }

        for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)            // TRIP COUNT: 10
            #pragma GCC ivdep
            for (integer ti = 0; ti != TileWidth; ++ti)                         // TRIP COUNT: TileWidth; UNIT STRIDE
            {
                auto const tmp = D[j][ti];                                      // 1 FP LOAD FROM TILE
                A0[j][ti] =  m0[0][ti] * tmp;                                   // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                A1[j][ti] = -m1[0][ti] * tmp;                                   // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
            }

        ///////////////////////////////////////////////////////////////////////
        // SCATTER: Store to L and L_c.

        for (integer j = 0; j != taylor_sizes[3]; ++j)                          // TRIP COUNT: 20 
            #pragma GCC ivdep
            for (integer i = i_begin; i != i_end; ++i)                          // TRIP COUNT: TileWidth CONTROLLED; UNIT STRIDE
            {
                integer const ti = i - i_begin;

                integer const iii0 = (*IList)[i].first;                         // 1 INT LOAD FROM MEM (indirect addressing)
                integer const iii1 = (*IList)[i].second;                        // 1 INT LOAD FROM MEM (indirect addressing)

                L[iii0][j] += A0[j][ti];                                        // 1 FP LOAD FROM TILE, 1 FP STORE TO MEM
                L[iii1][j] += A1[j][ti];                                        // 1 FP LOAD FROM TILE, 1 FP STORE TO MEM
            }

        // L_c stores the correction for angular momentum.
        // #20 in the paper ([Bryce] FIXME: Link to the paper) [Dominic].
        compute_interactions_initialize_L_c(ang_con_is_on);

        store_to_L_c<IList, TileWidth>(i_begin, i_end, B0, B1, ang_con_is_on_and_is_rho_type);
    }        
}

