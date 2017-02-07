import mmap
import sys
from itertools import izip

strings = ["rho","egas","sx","sy","tau","primary_core"]
diff = False

def find(f):
   data = []
   for line in f:
       for s in strings:
           if s in line:
               data.append(line.replace("\n","").lstrip())
   return data

def compare(old,test):
        for o, n in zip(old,test):
            if not (o == n):
                print "Missmatch: " + o + " and " + n
                diff = True
                index = [ i for i,(a1,a2) in enumerate(izip(o,n)) if a1!=a2 ]
                for i in index:
                    print "Difference at " + str(i) + " where is " + o[i] + " instead " + n[i]  

if (len(sys.argv) != 3):
    print sys.argv[0] + " file1 file2"
    sys.exit()

file = open(sys.argv[1],'r')
file2 = open(sys.argv[2],'r')
old = find(file)
test = find(file2)

compare(old,test)

if ( diff == True):
    sys.exit(1)
else:
    sys.exit(0)
