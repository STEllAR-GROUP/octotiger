#!/usr/bin/python

import matplotlib.pyplot as plt
import re
import os

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-d", "--display", dest="display",
                  action="store_true", default=False,
                  help="display the graphs in a window")
parser.add_option("-w", "--write-files",
                  dest="write_files",
                  action="store_true", default=False,
                  help="write graphs to png files")
parser.add_option("-c", "--compiler",
                  dest="compiler",
                  action="store", default="gnu",
                  help="results type, either \"gnu\" or \"intel\"")

(options, args) = parser.parse_args()

if options.display == None and options.write_files == None:
    print "no options set, nothing to do, (see \"--help\")"
    exit()

# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 18}

# plt.rc('font', **font)

performance_tuples = {}
result_dir = 'results-node-level-scaling'

compiler_suffix = None
if options.compiler == "gnu":
    compiler_suffix = "-gnu"
elif options.compiler == "intel":
    compiler_suffix = "-intel"
else:
    exit("error: invalid compiler specified");

result_dir = result_dir + compiler_suffix + "/"

for subdir, dirs, files in os.walk(result_dir):
    for f in files:
        match = re.search(r"node_level_scaling_N(.+?)_t(.+?)_l(.+?)_m(.+?)", f)
        if match:
            nodes = match.group(1)
            threads = match.group(2)
            level = match.group(3)

            if int(level) < 7:
                continue

            memory_type = match.group(4)

            # print "file:", f, "l:", level, "memory_type: ", memory_type, "nodes:", nodes, "threads:", threads
            result_file = open(result_dir + f, "r")
            result_text = result_file.read()
            result_file.close()
            # print result_text
            total_time = 0
            computation = 0
            regrid = 0
            find_localities = 0

            result_match = re.search(r"Total: (.+?)\n", result_text)
            if result_match:
                total_time = result_match.group(1)
            result_match = re.search(r"Computation: (.+?)\n", result_text)
            if result_match:
                computation = result_match.group(1)
            result_match = re.search(r"Regrid: (.+?)\n", result_text)
            if result_match:
                regrid = result_match.group(1)
            result_match = re.search(r"Find Localities: (.+?)\n", result_text)
            if result_match:
                find_localities = result_match.group(1)
            # print "total_time:", total_time, "computation:", computation, "regrid:", regrid, "find_localities:", find_localities

            if not memory_type in performance_tuples:
                performance_tuples[memory_type] = {}
            if not level in performance_tuples[memory_type]:
                performance_tuples[memory_type][level] = []
            performance_tuples[memory_type][level].append([int(threads), float(total_time)])

        else:
            print "warning: file ignored \"" + f + "\""

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

# total time comparison
for memory_type in performance_tuples.keys():
    for level in performance_tuples[memory_type].keys():

        tuples = sorted(performance_tuples[memory_type][level], key=lambda x: x[0])
        # print tuples

        x_list = []
        y_list = []

        for t in tuples:
            x_list.append(t[0])
            y_list.append(t[1])

        mem_label = "MCDRAM" if int(memory_type) == 1 else "DRAM"
        ax.plot(x_list, y_list, 'o--', label=str(mem_label + ", level=" + level))
        ax.legend()

plt.xlabel('threads')
plt.ylabel('time (s)')
if options.write_files:
    # plt.savefig("total_time.svg")
    fig.savefig("total-time-node-level-scaling" + compiler_suffix + ".png")
if options.display:
    plt.show()

plt.close()

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

# parallel efficieny (time compared to 1 thread)
for memory_type in performance_tuples.keys():
    for level in performance_tuples[memory_type].keys():

        tuples = sorted(performance_tuples[memory_type][level], key=lambda x: x[0])
        # print tuples

        x_list = []
        y_list = []

        for t in tuples:
            x_list.append(t[0])
            y_list.append(t[1])

        reference_value = y_list[0]

        for i in range(len(x_list)):
            # optimal parallel time divided by actual time
            if y_list[i] > 0.0:
                y_list[i] = (float(reference_value) / float(x_list[i])) / float(y_list[i])

        mem_label = "MCDRAM" if int(memory_type) == 1 else "DRAM"
        ax.plot(x_list, y_list, 'o--', label=str(mem_label + ", level=" + level))
        ax.legend(loc=3)

plt.xlabel('threads')
plt.ylabel('parallel efficiency')
if options.write_files:
    # plt.savefig("parallel_efficiency.svg")
    fig.savefig("parallel-efficiency-node-level-scaling" + compiler_suffix + ".png")
if options.display:
    plt.show()

plt.close()
