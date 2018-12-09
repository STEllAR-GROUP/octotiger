$1 --config_file=sod.ini -t2
if [ ! -f "sod.2.silo" ]; then
	wget phys.lsu.edu/~dmarcel/sod.2.silo
fi
$2 -x 1.0 -R 1.0e-12 sod.2.silo final.silo > diff.txt
cat diff.txt
if [[ $(wc -l <diff.txt) -gt 11 ]]; then
	echo 'Sod shock tube test failed comparison with original'
	exit 1
fi


