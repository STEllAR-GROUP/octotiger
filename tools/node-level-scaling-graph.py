#!/usr/bin/python

import matplotlib.pyplot as plt
import re
import os

# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 18}

# plt.rc('font', **font)

performance_tuples = {}
result_dir = 'results/'

for subdir, dirs, files in os.walk(result_dir):
    for f in files:
        match = re.search(r"node_level_scaling_N(.+?)_t(.+?)_l(.+?)_m(.+?)", f)
        if match:
            nodes = match.group(1)
            threads = match.group(2)
            level = match.group(3)
            memory_type = match.group(4)

            print "file:", f, "l:", level, "memory_type: ", memory_type, "nodes:", nodes, "threads:", threads
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
            print "total_time:", total_time, "computation:", computation, "regrid:", regrid, "find_localities:", find_localities

            if not memory_type in performance_tuples:
                performance_tuples[memory_type] = {}
            if not level in performance_tuples[memory_type]:
                performance_tuples[memory_type][level] = []
            performance_tuples[memory_type][level].append([int(threads), float(total_time)])

        else:
            print "warning: file ignored \"" + f + "\""

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
        plt.plot(x_list, y_list, 'o', label=str(mem_label + ", level=" + level))
        plt.legend()

plt.xlabel('threads')
plt.ylabel('time (s)')
plt.savefig("total_time.svg")
plt.savefig("total_time.png")
plt.show()

plt.close()

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
        plt.plot(x_list, y_list, 'o', label=str(mem_label + ", level=" + level))
        plt.legend()

plt.xlabel('threads')
plt.ylabel('parallel efficiency')
plt.savefig("parallel_efficiency.svg")
plt.savefig("parallel_efficiency.png")
plt.show()
