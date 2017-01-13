#!/usr/bin/python

import matplotlib.pyplot as plt
import re
import os

performance_tuples = []

for subdir, dirs, files in os.walk('results'):
    for f in files:
        match = re.search(r"level_(.+?)_m(.+?)_N(.+?)", f)
        if match:
            level = match.group(1)
            memory_type = match.group(2)
            nodes = match.group(3)
            print "file:", f, "l:", level, "memory_type: ", memory_type, "nodes:", nodes
            result_file = open('results/' + f, "r")
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

            performance_tuples.append([int(nodes), float(total_time)])

        else:
            print "warning: file ignored \"" + f + "\""

performance_tuples = sorted(performance_tuples, key=lambda x: x[0])
print performance_tuples

x_list = []
y_list = []

for t in performance_tuples:
    x_list.append(t[0])
    y_list.append(t[1])

plt.scatter(x_list, y_list)
plt.xticks(x_list)
plt.xlabel('nodes')
plt.ylabel('time (s)')
plt.show()
