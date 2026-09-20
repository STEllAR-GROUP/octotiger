# Gravity scenarios

Gravity uses the configured CTest adapter described in `../hydro/README.md`.
Sphere includes every registered field/Silo check and enabled kernel variant.
Rotating star includes the shared input generator, all registered EOS-WD and
CPU/GPU/combined variants, every field/Silo check and both levels of cleanup.
Numerical equations, input stop conditions and tolerance strings are unchanged.

```bash
python3 verification_results/runner.py run gravity Release --build /path/to/fresh/build --ctest /path/to/ctest --output /path/to/new/results
```

Raw data and generator output are archived before CTest cleanup. Tests absent or
disabled in a build are not certified; a pass only covers the exact registrations
retained in `ctest-inventory.json`. Physics validation requires running the actual
application and references. Small synthetic CTest fixtures validate harness
semantics only, not gravity or before/after numerical equivalence.
