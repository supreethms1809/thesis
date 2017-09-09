CC = icc
OPT = -O3 -g -mkl -heap-arrays 

SRCS = poseEstimation.cpp

LIB = 

EXE = estimation.exe

#run:
#	./estimation.exe

all: poseEstimation.cpp
	$(CC) $(OPT) $(LIB) $(SRCS) -o $(EXE)

clean: 
	rm ./estimation.exe    
