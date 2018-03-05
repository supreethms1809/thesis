# compiler option
CC = icc

OPT = -qopenmp -O3 -g -mkl -heap-arrays -parallel -ip -std=c++11

CUDA_PATH = /usr/local/cuda-9.0

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64/


SRCS = pose_estimation.cpp

LIB = 

OBJ = pose_estimation.o 

EXE = pose_estimation

run: all
	./pose_estimation

all:
	$(CC)  $(OPT) $(SRCS) $(LIB) -c 
	$(CC) $(OBJ) -o $(EXE) $(LIBS) $(OPT) 
clean: 
	rm ./pose_estimation
