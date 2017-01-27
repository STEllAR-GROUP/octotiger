#!/bin/bash

name=node_level_scaling
jobs_per_thread_scaling_experiment=6
job_increment=${jobs}

job_count=0

for memory_type in `seq 0 1`;
do
    echo memory_type: ${memory_type}
    for level in `seq 7 9`;
    do
	echo level: ${level}
	# min
	threads=1
	while [ ${threads} -le ${jobs_per_thread_scaling_experiment} ]; do
	    sbatch -N 1 -n 1 -J \"octotiger_N1_l${level}_t${threads}\" scaling-experiment-single-node.sh ${name} ${level} ${memory_type} ${threads} ${jobs_per_thread_scaling_experiment}
	    let threads=threads+1
	    let job_count=job_count+1
	done
    done
done

echo jobs: ${job_count}
