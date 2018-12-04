$OCTOTIGER=$1
$SILODIFF=$2
cd sod
$OCTOTIGER --config_file=sod.ini
$SILODIFF -A 1.0e-10 -R 1.0e-10 original.silo final.silo
