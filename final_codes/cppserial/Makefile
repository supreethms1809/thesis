# compiler option
CC = icc

OPT = -qopenmp -O3 -g -mkl -heap-arrays -std=c++11

CUDA_PATH = /usr/local/cuda-9.0

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64/


SRCS = poseEstimation.cpp

LIB = 

OBJ = poseEstimation.o 

EXE = poseEstimation

run: all
	./poseEstimation

all:
	$(CC)  $(OPT) $(SRCS) $(LIB) -c 
	$(CC) $(OBJ) -o $(EXE) $(LIBS) $(OPT) 
clean: 
	rm ./poseEstimation
