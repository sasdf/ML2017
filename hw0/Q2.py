#!/usr/bin/env python3
from PIL import Image
import sys
A, B = [ Image.open(fn) for fn in sys.argv[1:3] ]
P = [ (0,)*len(a) if a == b else b for a, b in zip(A.getdata(), B.getdata()) ]
A.putdata(P)
A.save('ans_two.png')
