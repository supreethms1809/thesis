CC = icc
#CC = gcc
OPT = -qopenmp -O3 -g -mkl -heap-arrays -ip -ipo -parallel -std=c++11

SRCS = dpposeEstimation.cpp

LIB = 

EXE = dpestimation.exe

#run:
#	./estimation.exe

all: dpposeEstimation.cpp 
	$(CC) $(OPT) $(LIB) $(SRCS) -o $(EXE)

clean: 
	rm ./dpestimation.exe    
