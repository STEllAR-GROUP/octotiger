# Actual HPX/Silo application smoke test, not a numerical-kernel substitute.
# Keep unique output directories so repeats never overwrite a prior run.
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef runId)
set(runDirectory "${testRoot}/${dimensionCount}D-${runId}")
file(MAKE_DIRECTORY "${runDirectory}")
set(hydroEnabled on)
set(radiationEnabled off)
set(physicsArguments)
if(physicsMode STREQUAL "radiation" OR physicsMode STREQUAL "mixed")
  set(radiationEnabled on)
  if(physicsMode STREQUAL "radiation")
    set(hydroEnabled off)
  endif()
  list(APPEND physicsArguments
    --radiation.modular.source_free=on
    --radiation.modular.problem=streamingGaussian
    --radiation.opacity.model=legacy --radiation.opacity.constant=0
    --radiation.implicit=off --radiation.velocity_terms=off --radiation.subcycling=off
    --radiation.reduced_light_speed_ratio=0.1 --radiation.cfl=0.2
    --radiation.test.width=5e9 --radiation.test.background=1 --radiation.test.amplitude=0.2
    --units.centimeters=1 --units.seconds=1 --units.grams=1 --mesh.scale=3e10)
endif()
execute_process(
  COMMAND "${application}"
    --runtime.modular.enabled=on --hydro.modular.problem=advection
    --problem.name=NONE --hydro.enabled=${hydroEnabled}
    --gravity.enabled=off --radiation.enabled=${radiationEnabled}
    --mesh.ndim=${dimensionCount} --mesh.boundary.periodic=on
    --mesh.level.minimum=1 --mesh.level.maximum=2
    --runtime.stop_time=0.001 --runtime.stop_step=2
    --output.disabled=off --output.interval=0.001
    --output.directory=${runDirectory}
    ${physicsArguments}
  WORKING_DIRECTORY "${runDirectory}"
  RESULT_VARIABLE runResult OUTPUT_VARIABLE runOutput ERROR_VARIABLE runError
  TIMEOUT 150)
if(NOT runResult EQUAL 0 OR NOT runOutput MATCHES "Modular (hydro|transport) completed")
  message(FATAL_ERROR "Modular ${dimensionCount}D failed (${runResult}):\n${runOutput}\n${runError}")
endif()
foreach(frame IN ITEMS modular_000000.silo modular_000001.silo)
  if(NOT EXISTS "${runDirectory}/${frame}")
    message(FATAL_ERROR "Missing Silo frame: ${runDirectory}/${frame}")
  endif()
endforeach()
message(STATUS "Modular ${dimensionCount}D completed; Silo frames in ${runDirectory}")
# This checks successful emission, not Silo readback or visualization behavior.
