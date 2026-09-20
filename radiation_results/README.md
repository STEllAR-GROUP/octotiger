# Compatibility path

The canonical radiation verification harness is now
`verification_results/radiation`. These launchers preserve the historical
`radiation_results/{run.sh,run_live.sh,results.sh,build_cpp.sh}` commands for
one migration cycle. New work should use `verification_results/run.sh` or the
canonical radiation launcher.

Successful runs now publish the unified descriptor catalog as `index.html`.
The familiar four-test page is retained in the same batch as
`radiation-application.html`.
