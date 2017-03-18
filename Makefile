LIBS=--relocatable-device-code=true
ARCH=-arch=sm_20
OPTIONS= -O2

hello: main.o EuclideanDistanceMatrixGPU.o
	nvcc $(OPTIONS) $(ARCH) $(LIBS) main.o EuclideanDistanceMatrixGPU.o -o hello

main.o: main.cpp
	nvcc $(OPTIONS) $(ARCH) $(LIBS) -c main.cpp -o main.o
	
EuclideanDistanceMatrixGPU.o: euclideandistancematrixgpu.cu
	nvcc $(OPTIONS) $(ARCH) $(LIBS) -c euclideandistancematrixgpu.cu -o EuclideanDistanceMatrixGPU.o
		
clean:
	rm -rf *.o hello.*
