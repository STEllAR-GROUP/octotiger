import sys
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

diff = False


def find(f):
   data = []
   for line in f:
       for s in ["rho", "egas", "sx", "sy", "tau", "primary_core"]:
           if s in line:
               data.append(line.replace("\n", "").lstrip())
   return data


def compare(old, test):
    for o, n in zip(old, test):
        if o != n:
            print("mismatch: {} and {}", o, n)
            diff = True
            index = [i for i, (a1, a2) in enumerate(zip(o, n)) if a1 != a2]
            for i in index:
                print("difference at {} where is {} instead {}".format(i, o[i], n[i]))


if (len(sys.argv) != 3):
    print(sys.argv[0] + " file1 file2")
    sys.exit()

file1 = open(sys.argv[1], 'r')
file2 = open(sys.argv[2], 'r')
old = find(file1)
test = find(file2)

compare(old, test)

if diff:
    print('fail')
    sys.exit(1)
else:
    print('success')
    sys.exit(0)
