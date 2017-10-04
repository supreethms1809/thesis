CC = icc
#CC = gcc
OPT = -qopenmp -O3 -g -mkl -heap-arrays -ip -ipo -parallel -std=c++11

SRCS = magma_dpposeEstimation.cpp

LIB = /home/sureshm/magma/include/ 
LIB2 = /usr/local/cuda-9.0/targets/x86_64-linux/include/

LIN = /home/sureshm/magma-2.2.0/lib/

EXE = magma_dpestimation.exe

run: all
	./magma_dpestimation.exe

all: magma_dpposeEstimation.cpp 
	$(CC) $(OPT) -I $(LIB) -I $(LIB2) -L $(LIN) -lmagma $(SRCS) -o $(EXE)

clean: 
	rm ./magma_dpestimation.exe    
