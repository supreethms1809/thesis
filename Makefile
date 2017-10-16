# compiler option
CC = icpc
NVCC = nvcc 

OPT = -qopenmp -O3 -g -mkl -heap-arrays -parallel -std=c++11

CUDA_PATH = /usr/local/cuda-9.0

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64/

INC = 

SRCS = cublas_dpposeEstimation.cpp

LIB = 

OBJ = cublas_dpposeEstimation.o cublas_inverse.o

EXE = cublas_dpposeestimation

run: all
	./cublas_dpposeestimation

all:
	$(NVCC) -O3 -g -arch=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcublas cublas_inverse.cu -c
	$(CC) -I$(INC) $(OPT) $(SRCS) $(LIB) -c  -lm
	$(CC) $(OBJ) -o $(EXE) $(LIBS) $(OPT) -lcudart -lcublas -lm 
clean: 
	rm ./cublas_dpposeestimation
