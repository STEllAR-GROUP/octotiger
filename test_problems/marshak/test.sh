$1/octotiger --config_file=marshak.ini 
if [ ! -f "marshak.2.silo" ]; then
	wget phys.lsu.edu/~dmarcel/marshak.2.silo
fi
$2 -x 1.0 -R 1.0e-12 marshak.2.silo final.silo > diff.txt
cat diff.txt
if [[ $(wc -l <diff.txt) -gt 11 ]]; then
	echo 'marshak Wave test failed comparison with original'
	exit 1
fi


