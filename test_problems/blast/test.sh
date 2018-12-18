$1 --config_file=blast.ini -t2
if [ ! -f "blast.3.silo" ]; then
	wget phys.lsu.edu/~dmarcel/blast.3.silo
fi
$2 -x 1.0 -R 1.0e-12 blast.3.silo final.silo > diff.txt
cat diff.txt
if [[ $(wc -l <diff.txt) -gt 11 ]]; then
	echo 'Blast Wave test failed comparison with original'
	exit 1
fi


