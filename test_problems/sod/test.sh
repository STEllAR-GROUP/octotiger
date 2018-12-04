$1 --config_file=sod.ini -t2
if [ ! -f "original.silo" ]; then
	wget phys.lsu.edu/~dmarcel/original.silo
fi
$2 -A 1.0e-10 -R 1.0e-10 original.silo final.silo > diff.txt
cat diff.txt
if [[ $(wc -l <diff.txt) -gt 11 ]]; then
	echo 'Sod shock tube test failed comparison with original'
	exit 1
fi


