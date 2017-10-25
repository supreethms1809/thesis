# compiler option
CC = icc
NVCC = nvcc 

OPT = -qopenmp -O3 -g -mkl -heap-arrays -parallel -std=c++11

CUDA_PATH = /usr/local/cuda-9.0

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64/

INC = 

SRCS = cuda_dpposeEstimation.cpp

LIB = 

OBJ = cuda_dpposeEstimation.o gjev2.o

EXE = cuda_dpposeEstimation

run: all
	./cuda_dpposeEstimation

all:
	$(NVCC) -O3 -g -arch=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart gjev2.cu -c
	$(CC)  $(OPT) $(SRCS) $(LIB) -c  -lm
	$(CC) $(OBJ) -o $(EXE) $(LIBS) $(OPT) -lcudart -lm 
clean: 
	rm ./cuda_dpposeEstimation
