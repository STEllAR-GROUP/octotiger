# *.diff tests fail when the output contains any .vals
set(OCTOTIGER_SILODIFF_FAIL_PATTERN ".vals")

add_subdirectory(blast)
add_subdirectory(marshak)
add_subdirectory(rotating_star)
add_subdirectory(sod)
add_subdirectory(sphere)
