#compiler
CC = icpc
NVCC = nvcc
#CC = gcc

#compiler options
OPT = -qopenmp -O3 -g -mkl -heap-arrays -ip -ipo -parallel -std=c++11
OPT2 = -Wall -fPIC -DADD_ -DMAGMA_SETAFFINITY -DMAGMA_WITH_MKL -Xlinker -shared -DHAVE_CUBLAS -DMIN_CUDA_ARCH=350  

SRCS = magma_dpposeEstimation.cpp

MAGMA_HOME = /home/sureshm/magma/ 

DYN_LIB = -lmagma -lcuda
LIB = -I/usr/local/cuda-9.0/targets/x86_64-linux/include/ -I/home/sureshm/magma-2.2.0/include/

LIN = /home/sureshm/magma-2.2.0/lib/libmagma.a

EXE = magma_dpestimation.exe

run: all
	./magma_dpestimation.exe

all: magma_dpposeEstimation.cpp 
	$(CC) $(OPT2) $(OPT) $(LIB) $(DYN_LIB) $(LIN) $(SRCS) -o $(EXE)

clean: 
	rm ./magma_dpestimation.exe    
