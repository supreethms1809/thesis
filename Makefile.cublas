#compiler
CC = icpc
NVCC = nvcc
#CC = g++ -fopenmp

#compiler options
OPT = -qopenmp -O3 -g -mkl -heap-arrays -parallel -std=c++11
OPT2 = -std=c++11 -Wall -fPIC -DADD_ -DMAGMA_SETAFFINITY -DMAGMA_WITH_MKL -DHAVE_CUBLAS -DMIN_CUDA_ARCH=350  

SRCS = magma_dpposeEstimation.cpp

MAGMA_HOME = /home/sureshm/magma/ 

DYN_LIB = -lmagma_sparse -lmagma -L/usr/local/cuda-9.0/lib64 -L/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lstdc++ -lcublas -lcusparse -lcudart -lcudadevrt

LIB = -I/usr/local/cuda-9.0/targets/x86_64-linux/include/ -I/home/sureshm/magma21/include/ -DNDEBUG -DADD_ -I/usr/local/cuda-9.0/include -I/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/include

LIN = -L/home/sureshm/magma21/lib/ 

EXE = magma_dpestimation.exe

run: all
	./magma_dpestimation.exe

all: magma_dpposeEstimation.cpp 
	$(CC) $(OPT2) $(OPT) $(LIB) $(DYN_LIB) $(LIN) $(SRCS) -o $(EXE)

clean: 
	rm ./magma_dpestimation.exe    
