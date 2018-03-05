# compiler option
CC = pgc++
NVCC = nvcc 

OPT = -O3 -g 

#CUDA_PATH = /cm/extra/apps/CUDA/9.0.176_384.81/
CUDA_PATH = /usr/local/cuda-9.0/

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64/


SRCS = pose_estimation.cpp

LIB = 

#OBJ = pose_estimation.o wood_gpu_call.o
OBJ = pose_estimation.o gpu_call.o

EXE = pose_estimation

run: all
	./pose_estimation

all:
#	$(NVCC) -std=c++11 -O3 -g -arch=sm_35 -gencode=arch=compute_35,code=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart inverse_shared.cu -c
#	$(NVCC) -std=c++11 -O3 -g -arch=sm_35 -gencode=arch=compute_35,code=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart prox_2norm_old.cu -c
#	$(NVCC) -std=c++11 -O3 -g -arch=sm_35 -gencode=arch=compute_35,code=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart MatMulShared.cu -c
#	$(NVCC) -std=c++11 -O3 -g -arch=sm_35 -gencode=arch=compute_35,code=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart loop_cuda.cu -c
	$(NVCC) -std=c++11 -O3 -g -arch=sm_35 -gencode=arch=compute_35,code=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart -lcublas gpu_call.cu -c
#	$(NVCC) -std=c++11 -O3 -g -arch=sm_35 -gencode=arch=compute_35,code=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart -lcublas wood_gpu_call.cu -c
	$(CC)  $(OPT) $(SRCS) $(LIB) -c -lcudart -lcublas -lm
	$(CC) $(OBJ) -o $(EXE) $(LIBS) $(OPT) -lcudart -lcublas -lm 
clean: 
	rm ./pose_estimation
