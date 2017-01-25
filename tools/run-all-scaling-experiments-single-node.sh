#!/bin/bash

name=node_level_scaling
threads_per_job=15

job_count=0

for memory_type in `seq 0 1`;
do
    echo memory_type: ${memory_type}
    for level in `seq 7 9`;
    do
	echo level: ${level}
	# min
	threads=1
	max_threads=68
	while [ ${threads} -le ${max_threads} ]; do
	    echo "threads from: ${threads} to: $[${threads}+${threads_per_job}] (limited to 68)"
	    sbatch -N 1 -n 1 -J \"octotiger_N1_l${level}_t${threads}\" scaling-experiment-single-node.sh ${name} ${level} ${memory_type} ${threads} $[${threads}+${threads_per_job}]
	    let threads=threads+${threads_per_job}
	    let job_count=job_count+1
	done
    done
done

echo jobs: ${job_count}
