#!/usr/bin/env python2
import sys, numpy
A, B = [ numpy.loadtxt(fn, delimiter=',', ndmin=2) for fn in sys.argv[1:3] ]
numpy.savetxt('ans_one.txt', numpy.sort(numpy.matmul(A, B)), fmt='%d', delimiter='\n')
