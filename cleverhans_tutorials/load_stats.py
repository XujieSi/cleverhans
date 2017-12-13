
import pickle
import numpy as np

import sys
import os

if len(sys.argv) != 2:
    print("Usage: %s pickle_file" % sys.argv[0] )
    exit(0)


pfile = sys.argv[1]
res = pickle.load( open(pfile, 'rb') )

print("res:", res)
