$1 --config_file=marshak.ini -t48
if [ ! -f "marshak.1.silo" ]; then
	wget phys.lsu.edu/~dmarcel/marshak.1.silo
fi
$2 -x 1.0 -R 1.0e-12 marshak.1.silo final.silo > diff.txt
grep ".vals" diff.txt > vals.txt
cat vals.txt
if [[ $(wc -l < vals.txt) -gt 0 ]]; then
	echo 'marshak shock tube test failed comparison with original'
	exit 1
fi


