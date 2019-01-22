$1/octotiger --config_file=marshak.ini -t2
if [ ! -f "marshak.1.silo" ]; then
	wget phys.lsu.edu/~dmarcel/marshak.1.silo
fi
$2 -x 1.0 -R 1.0e-12 original_marshak.silo final.silo > diff.txt
cat diff.txt
if [[ $(wc -l <diff.txt) -gt 11 ]]; then
	echo 'marshak Wave test failed comparison with original'
	exit 1
fi


