CC = icc
#CC = gcc
OPT = -qopenmp -O3 -g -mkl -heap-arrays -ip -ipo -parallel -std=c++11

SRCS = pre_dpposeEstimation.cpp

LIB = 

EXE = pre_dpestimation.exe

run: all
	./pre_dpestimation.exe

all: pre_dpposeEstimation.cpp 
	$(CC) $(OPT) $(LIB) $(SRCS) -o $(EXE)

clean: 
	rm ./pre_dpestimation.exe    
