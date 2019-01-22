$1/octotiger --config_file=sod.ini -t48
if [ ! -f "sod.4.silo" ]; then
	wget phys.lsu.edu/~dmarcel/sod.4.silo
fi
$2 -x 1.0 -R 1.0e-12 sod.4.silo final.silo > diff.txt
grep ".vals" diff.txt > vals.txt
cat vals.txt
if [[ $(wc -l < vals.txt) -gt 0 ]]; then
	echo 'Sod shock tube test failed comparison with original'
	exit 1
fi


