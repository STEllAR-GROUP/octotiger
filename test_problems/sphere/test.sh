$1/octotiger --config_file=sphere.ini 
if [ ! -f "sphere.1.silo" ]; then
	wget phys.lsu.edu/~dmarcel/sphere.1.silo
fi
$2 -x 1.0 -R 1.0e-12 sphere.1.silo final.silo > diff.txt
grep ".vals" diff.txt > vals.txt
cat vals.txt
if [[ $(wc -l < vals.txt) -gt 0 ]]; then
	echo 'Solid sphere test failed comparison with original'
	exit 1
fi

