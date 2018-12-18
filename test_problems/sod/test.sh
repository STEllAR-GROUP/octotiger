$1 --config_file=sod.ini -t2
if [ ! -f "sod.3.silo" ]; then
	wget phys.lsu.edu/~dmarcel/sod.3.silo
fi
$2 -x 1.0 -R 1.0e-12 sod.3.silo final.silo > diff.txt
cat diff.txt
if [[ $(wc -l <diff.txt) -gt 11 ]]; then
	echo 'Sod shock tube test failed comparison with original'
	exit 1
fi


