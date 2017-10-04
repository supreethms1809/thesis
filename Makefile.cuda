CC = nvcc
#CC = gcc
OPT = -O3 -g -std=c++11

SRCS = cuda_dpposeEstimation.cu 

LIB = /home/sureshm/magma-2.2.0/include/

LIN = /home/sureshm/magma-2.2.0/lib/

EXE = cuda_dpestimation.exe

run: all
	./cuda_dpestimation.exe

all: cuda_dpposeEstimation.cu
	$(CC) $(OPT) -I $(LIB) -L $(LIN) $(SRCS) -o $(EXE)

clean: 
	rm ./cuda_dpestimation.exe    
