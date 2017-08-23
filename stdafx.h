// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <tchar.h>
#include <conio.h>
#include <math.h>
#include <mkl.h>
#include <mkl_lapack.h>


using namespace std;
#define LAPACK_ROW_MAJOR   101


// TODO: reference additional headers your program requires here

int readValues(string text,float *variable,int i);
void displayValues(float *variable, int i);
void rowMean(float *variable, int col, int row, float *mean);
void Scalc(float *variable, int col, int row, float *mean);
float mean_of_std_deviation(float *variable,int col,int row, float *mean);
void newScalc(float *variable,int col,int row,float a);
void initializeZero(float *variable, int col, int row);
float meanCalc(float *variable, int col, int row);
void TransposeOnCPU(float *matrix, float *matrixTranspose, int col, int row);
void cpuTransMatrixMult(float *A, float *B, float *C,  int col, int row);
void initialize(float *variable,float *variable2, int col, int row);
void calculateZ(float *Z, float *BBt,float *xy,float *E,float *T,float *B_transpose,float mu,float *M,float *Y,const int row,const int col,const int row1);
void cpuMatrixMult(float *A, float *B, float *C, int row, int col,int col2);
void scalarToMatrixMultiply(float *Temp,float *M,float mu,int row,int col);
void sumOfMatrix(float *Znum,float *temp,float *temp2,float *temp3,int row,int col);
void addScalarToDiagonal(float *Zden,float *BBt,float mu, int row, int col);
void cpuInverseOfMatrix(float *matrix, int n);
void AugmentIdentity(float *matrix, float *augmatrix, int n);
void Inverse(float *augmatrix,float *matrixInverse,int n);
void calculateQ(float *Q,float *Z,float *Y,float mu,int row,int row1);
void diferenceOfMatrix(float *diffMatrix, float *matrix1, float *matrix2, int row, int col);
//void prox_2norm(float *Q,float *M,float *C,float constant, int row, int col, int data_size);

