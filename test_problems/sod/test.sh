$1 -t8 --config_file=sod.ini
$2 original.silo final.silo > diff.txt
if [[ $(wc -l <diff.txt) -gt 11 ]]; then
	echo 'Sod shock tube test failed comparison with original'
	exit 1
fi


