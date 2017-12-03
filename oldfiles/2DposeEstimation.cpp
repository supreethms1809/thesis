// 2DposeEstimation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>


int main()
{
	const int row = 2;
	const int col = 15;
	const int row1 = 384;
	const int col1 = 15;
	

	float *B = (float*)_aligned_malloc(row1*col1*sizeof(float), 16);
	float *B_transpose = (float*)_aligned_malloc(col1*row1*sizeof(float), 16);
	float *BBt = (float*)_aligned_malloc(row1*row1*sizeof(float), 16);
	float *xy = (float*)_aligned_malloc(row*col*sizeof(float), 16);
	float *one = (float*)_aligned_malloc(row*col*sizeof(float), 16);
	float *mean = (float*)_aligned_malloc(row*sizeof(float), 16);
	float *newmean = (float*)_aligned_malloc(row * sizeof(float), 16);
	float *B_mean = (float*)_aligned_malloc(row1*sizeof(float), 16);

	float a = 0.0f;
	int items = 0;
	int B_items = 0;
	int lam = 1;
	
	//read from file
	items = readValues("exp.txt",xy, items);
	//cout << "value of items " << items << endl;
	//displayValues(xy, items);

	//normalize S
	rowMean(xy, col, row, mean);
	Scalc(xy, col, row, mean);
	rowMean(xy, col, row, newmean);
	a = mean_of_std_deviation(xy, col, row, newmean);
	newScalc(xy, col, row, a);

	//ssr2D3D_wrapper
	B_items = readValues("exp1.txt", B, B_items);
	//cout << "value of B_items " << B_items << endl;
	//displayValues(B, B_items);
	rowMean(B, col1, row1, B_mean);
	Scalc(B, col1, row1, B_mean);
	int data_size = row1 / 3;
	
	//ssr2D3D_alm
	//M ==> (2*384) = 0, C ==> (1*384) = 0, E ==> (2*15) = 0, T ==> mean(W,2)
	

	float *M = (float*)_aligned_malloc(row*row1*sizeof(float), 16);
	float *C = (float*)_aligned_malloc(data_size*sizeof(float), 16);
	float *E = (float*)_aligned_malloc(row * col*sizeof(float), 16);
	float *T = (float*)_aligned_malloc(row*sizeof(float), 16);

	initializeZero(M,row1,row);
	initializeZero(C, row1, 1);
	initializeZero(E, col, row);
	displayValues(xy, items);
	//rowMean(xy, col, row, T);
	//displayValues(xy, items);

	//// auxiliary variables for ADMM
	//float *Z = (float*)_aligned_malloc(row*row1*sizeof(float), 16);
	//float *Y = (float*)_aligned_malloc(row*row1*sizeof(float), 16);
	//float *Z0 = (float*)_aligned_malloc(row*row1*sizeof(float), 16);
	//float *Q = (float*)_aligned_malloc(row*row1*sizeof(float), 16);
	//

	//float mu = 0;
	//initializeZero(Z, row1, row);
	//initializeZero(Y, row1, row);
	//
	//mu = meanCalc(xy, col, row);

	//TransposeOnCPU(B, B_transpose, row1, col);
	//cpuTransMatrixMult(B, B_transpose, BBt, row1, col);
	//
	//	initialize(Z0, Z, row1, row);
	//	calculateZ(Z, BBt,xy, E, T, B_transpose,mu,M,Y,row,col,row1);
	//	calculateQ(Q,Z,Y,mu,row,row1);
	//	//prox_2norm(Q,M,C,lam/mu,row,row1,data_size);


		//free the allocated memory
		_aligned_free(B);
		_aligned_free(B_transpose);
		_aligned_free(BBt);
		_aligned_free(xy);
		_aligned_free(one);
		_aligned_free(mean);
		_aligned_free(newmean);
		_aligned_free(B_mean);
		_aligned_free(M);
		_aligned_free(C);
		_aligned_free(E);
		_aligned_free(T);
		//_aligned_free(Z);
		//_aligned_free(Y);
		//_aligned_free(Z0);
		//_aligned_free(Q);



	cout << "Press any key to exit" << endl;
	getchar();
	return 0;
}

