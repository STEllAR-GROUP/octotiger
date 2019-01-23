$1/gen_rotating_star_init
$1/octotiger --config_file=rotating_star.ini 
if [ ! -f "rotating_star.1.silo" ]; then
	wget phys.lsu.edu/~dmarcel/rotating_star.1.silo
fi
$2 -x 1.0 -R 1.0e-12 rotating_star.1.silo final.silo > diff.txt
grep ".vals" diff.txt > vals.txt
cat vals.txt
if [[ $(wc -l < vals.txt) -gt 0 ]]; then
	echo 'Rotating star test failed comparison with original'
	exit 1
fi


