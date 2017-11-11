# compiler option
CC = icc
NVCC = nvcc 

OPT = -qopenmp -qopt-report=3 -xHost -align -O3 -g -mkl=parallel -heap-arrays -parallel -std=c++11

CUDA_PATH = /usr/local/cuda-9.0

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64/

INC = 

SRCS = cuda_dpposeEstimation.cpp

LIB = 

OBJ = cuda_dpposeEstimation.o inverse_shared.o

EXE = cuda_dpposeEstimation

run: all
	./cuda_dpposeEstimation

all:
	$(NVCC) -O3 -g  -std=c++11 -arch=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart inverse_shared.cu -c
	$(CC)  $(OPT) $(SRCS) $(LIB) -c  -lm
	$(CC) $(OBJ) -o $(EXE) $(LIBS) $(OPT) -lcudart -lm 
clean: 
	rm ./cuda_dpposeEstimation
