# compiler option
CC = icc
NVCC = nvcc 

OPT = -qopenmp -O3 -g -mkl -heap-arrays -parallel -ip -std=c++11

CUDA_PATH = /usr/local/cuda-9.0

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64/


SRCS = pose_estimation.cpp

LIB = 

OBJ = pose_estimation.o inverse_shared.o prox_2norm.o

EXE = estimation

run: all
	./estimation

all:
	$(NVCC) -std=c++11 -O3 -g -arch=sm_35 -gencode=arch=compute_35,code=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart inverse_shared.cu -c
	$(NVCC) -std=c++11 -O3 -g -arch=sm_35 -gencode=arch=compute_35,code=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart prox_2norm.cu -c
	$(CC)  $(OPT) $(SRCS) $(LIB) -c  -lm
	$(CC) $(OBJ) -o $(EXE) $(LIBS) $(OPT) -lcudart -lm 
clean: 
	rm ./estimation
