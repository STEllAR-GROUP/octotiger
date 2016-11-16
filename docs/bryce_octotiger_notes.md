Issues
------

MPI_WTime() should be replaced with std::chrono and the MPI dependency should
be removed.

main() sets up the problem (either generating it from initial conditions by
calling initialize() or by loading it from a checkpoint file) and then call
node_server::start_run(). node_server::start_run() takes a bool that specifies
whether SCF should be run. SCF should only be run if the problem was generated
from initial conditions, so it seems like SCF should probably be in
initialize(), or be called by main().

Parts of node_client live in node_server_actions_*.cpp - the parts which need a
declaration or definition of the server actions. It should be possible to put
these into node_client.cpp and just use action declarations.

The problem type shouldn't be a global variable with free functions for setting
it.

The fenv.h setup should be moved out of initialize() to a new action which is
invoked on all localities.

Find and replace vector<vector<>>s, vector<array<vector<>>>, etc with
proper multidimensional arrays.

Lift multidimensional indexing calculations out of the innermost loop in
multidimensional loops to avoid the Intel compiler vectorizing the index
calculations instead of the math.

Remove the hpx::mutex = hpx::lcos::local::spinlock alias.

Futurization: Replace fork-join asynchrony with continuation-passing-style
asynchrony.

node_client action-invoking methods should be const correct.

Replace std::list with std::vector where appropriate.

There's a mix of global variables (problem function) and static members of
node_server (gravity_on flag). All of this state should be abstracted to a
single global configuration class.

Audit shared_ptr<> vs unique_ptr<> choices.

There seems to be a desire for a "solve gravity once, no hydro" mode; the logic
for this is split between main and node_server::start_run() in an odd way.

In the FMM code, are the interaction lists padded to SIMD vector width? If not,
they should be.

Put everything from taylor.hpp into a header.

Make compute_ilist constexpr if possible, even if heroics are needed. We have way
too much data movement in the gravity solve (compute_interactions()) - our FLOPS/byte
ratio is really low. We've got to reduce working set size wherever possible.

Questions
---------

What does regrid(gid_type root, bool rebalance_only) do when rebalance_only is true?
Why do we regrid(gid_type root, true) when loading from a checkpoint file? Presumably it was load balanced before it the checkpoint file was written?
Why pass the root_gid to regrid() instead of storing it as a member in node_server?
Can refinining and rebalancing be separated into two functions (instead of one with a bool flag to turn off refinement), or are they tightly coupled?

What does compute_ilist() do?
A: Only done at startup
A: Pre-computes the dependencies of each point in the subgrid.

What does node_server::form_tree() do (I have a basic understanding of this, octopus had a similar operation, but specifics would be good)?
What is the ene paramter in node_server::solve_gravity(bool ene) (I'm guessing ene is energy)?
Does node_server need to hold a shared_ptr to its grid?
If loading from a checkpoint, is it necessary to do solve_gravity() in main()? start_run() will call solve_gravity() early on (and there's no SCF if we loaded from a checkpoint. Similarly, should regrid be called in start_run() if we've loaded from a checkpoint and we're not doing SCF?
How do grid::solve_gravity() and node_server::solve_gravity() differ?

array<simd_vector, NDIM> seems wrong, shouldn't it be array<simd_vector, NDIM/simd_len>?
A: No, that's correct.

What array layout is used? *index() appears to be layout_right (aka C++ indexing)
What iteration order is used?
Is any vectorization done today?
Which extents are known at compile-time?

