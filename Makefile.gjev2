# compiler option
CC = icpc
NVCC = nvcc 

OPT = -qopenmp -O3 -g -mkl -heap-arrays -parallel -std=c++11

CUDA_PATH = /usr/local/cuda-9.0

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64/

INC = 

SRCS = gje_inverse.cpp

LIB = 

OBJ = gje_inverse.o gjev2.o

EXE = gjev2

run: all
	./gjev2

all:
	$(NVCC) -O3 -g -arch=sm_35 -Xptxas -dlcm=cg -I$(INC) $(INCLUDES) -lcudart gjev2.cu -c
	$(CC) -I$(INC) $(OPT) $(SRCS) $(LIB) -c  -lm
	$(CC) $(OBJ) -o $(EXE) $(LIBS) $(OPT) -lcudart -lm 
clean: 
	rm ./gjev2
