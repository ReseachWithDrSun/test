make:
	#for python2 
	#python setup.py build_ext --inplace
	#for python3
	python3 setup.py build_ext --inplace

qintegrate:
	gcc -o qintegrate qintegrate.c -lm

rotate:
	#gcc -o testrotate testrotate.c rotate.c printmat.c -lm -lgsl -lgslcblas
	gcc -o testrotate testrotate.c rotate.c printmat.c -lm
