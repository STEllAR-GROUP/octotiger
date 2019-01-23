$1/octotiger --config_file=blast.ini
if [ ! -f "blast.4.silo" ]; then
	wget phys.lsu.edu/~dmarcel/blast.4.silo
fi
$2 -x 1.0 -R 1.0e-12 blast.4.silo final.silo > diff.txt
grep ".vals" diff.txt > vals.txt
cat vals.txt
if [[ $(wc -l < vals.txt) -gt 0 ]]; then
	echo 'Blast test failed comparison with original'
	exit 1
fi
rm vals.txt
rm diff.txt


