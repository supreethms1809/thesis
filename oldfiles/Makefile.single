CC = icc
#CC = gcc
OPT = -qopenmp -O3 -g -mkl -heap-arrays -ip -ipo -parallel -std=c++11

SRCS = poseEstimation.cpp

LIB = 

EXE = estimation.exe

run: all
	./estimation.exe

all: poseEstimation.cpp 
	$(CC) $(OPT) $(LIB) $(SRCS) -o $(EXE)

clean: 
	rm ./estimation.exe    
