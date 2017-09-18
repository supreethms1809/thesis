CC = icc
#CC = gcc
OPT = -qopenmp -O0 -g -mkl -heap-arrays 

SRCS = dpposeEstimation.cpp

LIB = 

EXE = dpestimation.exe

#run:
#	./estimation.exe

all: dpposeEstimation.cpp
	$(CC) $(OPT) $(LIB) $(SRCS) -o $(EXE)

clean: 
	rm ./dpestimation.exe    
