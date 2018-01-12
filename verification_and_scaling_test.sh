#!/bin/bash

# Checks results - note do not use any variable names from main program...
check_results() {
	# Loop through all result names
	for result_name in "rho" "egas" "sx" "sy" "tau" "primary_core"; do
		# | sed -e 's/^[ \t]*// removes all leading whitespaces
		result_correct="$(cat "$1" | grep "$result_name" | sed -e 's/^[ \t]*//' | sed "s/$result_name//g")"
		result_actual="$(echo "$2" | grep "$result_name" | sed -e 's/^[ \t]*//' | sed "s/$result_name//g")"
		if [ "$result_correct" == "$result_actual" ]; then
		echo "->$result_name correct!" | tee -a LOG.txt
		else
			echo "==->ERROR: $result_name not correct!" | tee -a LOG.txt
			echo "==->Actual: $result_correct" | tee -a LOG.txt
			echo "==->Correct: $result_actual" | tee -a LOG.txt
			errorcount=$errorcount+1
		fi
	done
}

#Check whether all arguments are present
if [ $# -ne 5 ]; then
	echo "USAGE:"
	echo "------"
	echo "Argument 1: scenario file (for example path/to/scenarion/scenario1.txt)"
	echo "            scenario file contains octotiger parameters and expected results"
	echo "Argument 2: Number of HPX threads for first test (for example 1)"
	echo "Argument 3: Stepsize in which the numer of HPX threads is increased each test (for example 1)"
	echo "Argument 4: Number of HPX threads for the last test (for example 64)"
	echo "Argument 5: Path to octotiger executable"
	exit 128
fi

# Get base dir
basedir=`pwd`
# Get current date
today=`date +%Y-%m-%d_%H:%M:%S`
# Create Test folder
result_folder="Testrun-$today"
mkdir "$result_folder"

# Get scenario arguments and current commits
octotiger_args="$(cat "$1" | grep 'octotiger_args' | sed 's/octotiger_args: //g')"
current_commit=`git rev-parse HEAD`

# Log configuration
echo "Using scenarion file: $1" | tee "$result_folder/LOG.txt"
echo "Using Octotiger commit: $current_commit" | tee -a "$result_folder/LOG.txt"
echo "Octotiger arguments: $octotiger_args" | tee -a "$result_folder/LOG.txt"

# Create result files
cd "$result_folder"
echo "# Octotiger commit: $current_commit " > computation_time_results.txt
echo "# Date of run $today" >> computation_time_results.txt
echo "# Measuring computation time" >> computation_time_results.txt
echo "#" >> computation_time_results.txt
echo "#Number HPX threads,All off,m2m on,m2p on,p2p on,p2m on,All on,All on except p2m" >> computation_time_results.txt
echo "# Octotiger commit: $current_commit " > total_time_results.txt
echo "# Date of run $today" >> total_time_results.txt
echo "# Measuring total time" >> total_time_results.txt
echo "#" >> total_time_results.txt
echo "#Number HPX threads,All off,m2m on,m2p on,p2p on,p2m on,All on,All on except p2m" >> total_time_results.txt
# Save this version of the script for sanity checks later on
cp ../verification_and_scaling_test.sh used-script-copy.txt
# Save scenario file for sanity check later on
cp ../$1 $1

# Save disassembly of performance critical functions for later analysis
echo "" | tee -a LOG.txt
echo "Getting assembly/source mix of kernels..." | tee -a LOG.txt
mkdir kernel-mixed-assembly
gdb ../$5 -batch \
        -ex "disassemble /s octotiger::fmm::multipole_interactions::m2m_kernel::blocked_interaction_rho" \
        -ex "quit" > kernel-mixed-assembly/m2m_rho.asm
gdb ../$5 -batch \
        -ex "disassemble /s octotiger::fmm::multipole_interactions::m2m_kernel::blocked_interaction_non_rho" \
        -ex "quit" > kernel-mixed-assembly/m2m_non_rho.asm
echo "Got multipole multipole kernels!" | tee -a LOG.txt
gdb ../$5 -batch \
        -ex "disassemble /s octotiger::fmm::multipole_interactions::m2p_kernel::blocked_interaction_rho" \
        -ex "quit" > kernel-mixed-assembly/m2p_rho.asm
gdb ../$5 -batch \
        -ex "disassemble /s octotiger::fmm::multipole_interactions::m2p_kernel::blocked_interaction_non_rho" \
        -ex "quit" > kernel-mixed-assembly/m2p_non_rho.asm
echo "Got multipole monopole kernels!" | tee -a LOG.txt
gdb ../$5 -batch \
        -ex "disassemble /s octotiger::fmm::monopole_interactions::p2m_kernel::blocked_interaction_rho" \
        -ex "quit" > kernel-mixed-assembly/p2m_rho.asm
gdb ../$5 -batch \
        -ex "disassemble /s octotiger::fmm::monopole_interactions::p2m_kernel::blocked_interaction_non_rho" \
        -ex "quit" > kernel-mixed-assembly/p2m_non_rho.asm
echo "Got monopole multipole kernels!" | tee -a LOG.txt
gdb ../$5 -batch \
        -ex "disassemble /s octotiger::fmm::monopole_interactions::p2p_kernel::blocked_interaction" \
        -ex "quit" > kernel-mixed-assembly/p2p.asm
echo "Got monopole monopole kernel!" | tee -a LOG.txt
echo "" | tee -a LOG.txt
echo "Getting  raw assembly of kernels..." | tee -a LOG.txt
mkdir kernel-raw-assembly
gdb ../$5 -batch \
        -ex "disassemble octotiger::fmm::multipole_interactions::m2m_kernel::blocked_interaction_rho" \
        -ex "quit" > kernel-raw-assembly/m2m_rho.asm
gdb ../$5 -batch \
        -ex "disassemble octotiger::fmm::multipole_interactions::m2m_kernel::blocked_interaction_non_rho" \
        -ex "quit" > kernel-raw-assembly/m2m_non_rho.asm
echo "Got multipole multipole kernels!" | tee -a LOG.txt
gdb ../$5 -batch \
        -ex "disassemble octotiger::fmm::multipole_interactions::m2p_kernel::blocked_interaction_rho" \
        -ex "quit" > kernel-raw-assembly/m2p_rho.asm
gdb ../$5 -batch \
        -ex "disassemble octotiger::fmm::multipole_interactions::m2p_kernel::blocked_interaction_non_rho" \
        -ex "quit" > kernel-raw-assembly/m2p_non_rho.asm
echo "Got multipole monopole kernels!" | tee -a LOG.txt
gdb ../$5 -batch \
        -ex "disassemble octotiger::fmm::monopole_interactions::p2m_kernel::blocked_interaction_rho" \
        -ex "quit" > kernel-raw-assembly/p2m_rho.asm
gdb ../$5 -batch \
        -ex "disassemble octotiger::fmm::monopole_interactions::p2m_kernel::blocked_interaction_non_rho" \
        -ex "quit" > kernel-raw-assembly/p2m_non_rho.asm
echo "Got monopole multipole kernels!" | tee -a LOG.txt
gdb ../$5 -batch \
        -ex "disassemble octotiger::fmm::monopole_interactions::p2p_kernel::blocked_interaction" \
        -ex "quit" > kernel-raw-assembly/p1p.asm
echo "Got monopole monopole kernel!" | tee -a LOG.txt

# Need to use declare, otherwise bash assumes it is a string (untyped)
declare -i errorcount=0
# Running tests
echo "" | tee -a LOG.txt
echo "Running all tests..." | tee -a LOG.txt
for i in `seq $2 $3 $4`; do
	echo "Running test $i.1 - all off..." | tee -a LOG.txt
	output1="$(../$5 -Ihpx.stacks.use_guard_pages=0 -t$i $octotiger_args p2p_kernel_type=old p2m_kernel_type=old m2m_kernel_type=old m2p_kernel_type=old)"
	check_results $1 "$output1"
	echo "Running test $i,2 - m2m on..." | tee -a LOG.txt
	output2="$(../$5 -Ihpx.stacks.use_guard_pages=0 -t$i $octotiger_args p2p_kernel_type=old p2m_kernel_type=old m2m_kernel_type=soa_cpu m2p_kernel_type=old)"
	check_results $1 "$output2"
	echo "Running test $i.3 - m2p on..." | tee -a LOG.txt
	output3="$(../$5 -Ihpx.stacks.use_guard_pages=0 -t$i $octotiger_args p2p_kernel_type=old p2m_kernel_type=old m2m_kernel_type=old m2p_kernel_type=soa_cpu)"
	check_results $1 "$output3"
	echo "Running test $i.4 - p2p on..." | tee -a LOG.txt
	output4="$(../$5 -Ihpx.stacks.use_guard_pages=0 -t$i $octotiger_args p2p_kernel_type=soa_cpu p2m_kernel_type=old m2m_kernel_type=old m2p_kernel_type=old)"
	check_results $1 "$output4"
	echo "Running test $i.5 - p2m on..." | tee -a LOG.txt
	output5="$(../$5 -Ihpx.stacks.use_guard_pages=0 -t$i $octotiger_args p2p_kernel_type=old p2m_kernel_type=soa_cpu m2m_kernel_type=old m2p_kernel_type=old)"
	check_results $1 "$output5"
	echo "Running test $i.6 - All on..." | tee -a LOG.txt
	output6="$(../$5 -Ihpx.stacks.use_guard_pages=0 -t$i $octotiger_args p2p_kernel_type=soa_cpu p2m_kernel_type=soa_cpu m2m_kernel_type=soa_cpu m2p_kernel_type=soa_cpu)"
	check_results $1 "$output6"
	echo "Running test $i.7 - All on except p2m..." | tee -a LOG.txt
	output7="$(../$5 -Ihpx.stacks.use_guard_pages=0 -t$i $octotiger_args p2p_kernel_type=soa_cpu p2m_kernel_type=old m2m_kernel_type=soa_cpu m2p_kernel_type=soa_cpu)"
	check_results $1 "$output7"
	# Clean up results
	clean_output_computational="$i,$(echo "$output1" | grep 'Computation' | sed 's/Computation: //g'),$(echo "$output2" | grep 'Computation' | sed 's/Computation: //g'),$(echo "$output3" | grep 'Computation' | sed 's/Computation: //g'),$(echo "$output4" | grep 'Computation' | sed 's/Computation: //g'),$(echo "$output5" | grep 'Computation' | sed 's/Computation: //g'),$(echo "$output6" | grep 'Computation' | sed 's/Computation: //g'),$(echo "$output7" | grep 'Computation' | sed 's/Computation: //g')"
	clean_output_total="$i,$(echo "$output1" | grep 'Total' | sed 's/Total: //g'),$(echo "$output2" | grep 'Total' | sed 's/Total: //g'),$(echo "$output3" | grep 'Total' | sed 's/Total: //g'),$(echo "$output4" | grep 'Total' | sed 's/Total: //g'),$(echo "$output5" | grep 'Total' | sed 's/Total: //g'),$(echo "$output6" | grep 'Total' | sed 's/Total: //g'),$(echo "$output7" | grep 'Total' | sed 's/Total: //g')"
	# Print and save to files >> for appending
	echo "$clean_output_computational" >> "computation_time_results.txt"
	echo "$clean_output_computational" | tee -a LOG.txt
	echo "$clean_output_total" >> "total_time_results.txt"
	echo "$clean_output_total" | tee -a LOG.txt
done

# Plot results of computation and store into folder
echo "" | tee -a LOG.txt
echo "Plotting graphs..." | tee -a LOG.txt
cp ../plot-results.plt plot-results.plt
gnuplot -e "filename='computation_time_results.txt'" -e "outfile='computation_time_scaling.png'" plot-results.plt
gnuplot -e "filename='total_time_results.txt'" -e "outfile='total_time_scaling.png'" plot-results.plt

# Show error count and generate warning file if there are any
echo "" | tee -a LOG.txt
echo "All done!" | tee -a LOG.txt
echo "Errorcount: $errorcount" | tee -a LOG.txt
if [ $errorcount -gt 0 ]; then
	touch THERE_HAVE_BEEN_ERROR_IN_THIS_RUN
fi
