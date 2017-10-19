# compiler option
CC = icpc
NVCC = nvcc 

OPT = -qopenmp -O3 -g -mkl -heap-arrays -parallel -std=c++11

CUDA_PATH = /usr/local/cuda-9.0

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64/

INC = 

SRCS = cuda_dpposeEstimation.cpp

LIB = 

OBJ = cuda_dpposeEstimation.o gje.o

EXE = cuda_dpposeEstimation

run: all
	./cuda_dpposeEstimation

all:
	$(NVCC) -O3 -g -arch=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart gje.cu -c
	$(CC) -I$(INC) $(OPT) $(SRCS) $(LIB) -c  -lm
	$(CC) $(OBJ) -o $(EXE) $(LIBS) $(OPT) -lcudart -lm 
clean: 
	rm ./cuda_dpposeEstimation
