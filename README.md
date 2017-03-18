# Exact-CUDA-kNN
CUDA code with exact k-NN algorithm for multiple GPU system.

AUTHORS
=======
	
	Kłusek Adrian
	Witold Dzwinel
	
REQUIREMENTS
============

	- CUDA device with 2.0 Compute Capability and above
	- CUDA drivers and CUDA toolkit
	- C++ compiler
	- make
	
COMPILATION & EXECUTION
=======================

	For C++
		1.	Compile program by make
		3.	Execute the program by
			hello.exe <input_file> <output_file> <number_of_neighbours> <distance_Algorithm> <id_gpu> <cluster: T/F> <--debug>
			Example launching: hello.exe input.csv output.csv 2 EUCLIDEAN 1,2,3 T
			Example launching: hello.exe input.csv output.csv 2 EUCLIDEAN 1,2,3 T --debug
			Available distance algorithms: EUCLIDEAN TAXICAB COSINE
			
ORGANISATION OF DATA
====================
	
	The input file has to have each point in new line. 
	In the first line must be the number of points.
	In the second line must be the number of dimensions.
	The features should be separated by comma (,).
	If the file has cluster, it sould be at the end of the line.
	
	Example without cluster:
	3
	2
	0.1, 0.2
	0.3, 0.4
	0.5, 0.6
	
