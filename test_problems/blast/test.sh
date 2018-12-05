$1 --config_file=blast.ini -t2
if [ ! -f "original_blast.silo" ]; then
	wget phys.lsu.edu/~dmarcel/original_blast.silo
fi
$2 -x 1.0 -R 1.0e-12 original_blast.silo final.silo > diff.txt
cat diff.txt
if [[ $(wc -l <diff.txt) -gt 11 ]]; then
	echo 'Blast Wave test failed comparison with original'
	exit 1
fi


