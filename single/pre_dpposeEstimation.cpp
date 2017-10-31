#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_lapack.h>
#include <limits>
#include <chrono>
#include <ctime>
#include <omp.h>

#define LAPACK_ROW_MAJOR   101
#define min(a,b) ((a)>(b)?(b):(a))
#define ROW 2
#define COL 3
#define LDA COL
#define LDU ROW
#define LDVT COL


using std::string;
using namespace std;
using namespace std::chrono;

extern void gpuInverseOfMatrix(float *h_matrix,float *h_iden_mat, int col);

int readValues(char *text, float *variable, int i,int row,int col)
{
 	float temp;
	ifstream myReadFile;
	myReadFile.open(text, ios::in);
	if (myReadFile.is_open()) 
	{
		while (!myReadFile.eof())
		{
			if(i < (row*col))
			{
			myReadFile >> temp;
			variable[i] = temp;
			i++;
			}
			else
			{
				break;
			}
		}
	}
  	myReadFile.close();
	return i;
}

void dump_to_file(char *filename, float *matrix, int row, int col)
{
	ofstream fs;
	fs.open(filename, ios::out);
	for(int i = 0; i<row;i++)
	{
		for(int j = 0;j<col ;j++)
		{
		//if(i==j)
		//{
		fs << matrix[i*col+j] << "\n" ;
		//}
		}
	fs << "\n";
	}
}

void print_matrix( char *desc, MKL_INT m, MKL_INT n, float *a) 
{
        MKL_INT i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) 
	{
       		for( j = 0; j < n; j++ )
		{
		cout << a[i*n+j] << "\t";
		}
        cout << "\n" ;
	}
}

void displayValues(float *variable, int items)
{
	cout.precision(17);
	for (int i =0; i < items; i++)
	{
		cout << "Value of variable :"<< variable[i] << endl;
	}
}

void rowMean(float *variable, int col, int row , float *mean)
{
	float sum;
	for (int j = 0;j < row; j++)
	{
		sum = 0;
		for (int i = 0;i < col; i++)
		{									                
			sum += variable[(j*col)+ i];
		}
		mean[j] = sum / col;
	}
}

void Scalc(float *variable, int col, int row, float *mean)
{
	for (int j = 0;j < row;j++)
	{
		for (int i = 0;i < col;i++)
		{
			variable[(j * col) + i] = variable[(j * col) + i] - mean[j];
		}
	}
}

float mean_of_std_deviation(float *variable, int col, int row, float *mean)
{
	float std[2];
	float temp,a;
	for (int j = 0;j < row;j++)
	{
		temp = 0;
		for (int i = 0;i < col;i++)
		{
		temp += pow((variable[(j * 15) + i] - mean[j]), 2);
		}
	std[j] = sqrt(temp / (col));
	}
	a = (std[0] + std[1]) / 2;
	return a;
}

void newScalc(float *variable, int col, int row, float a)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		variable[(i * col) + j] = variable[(i * col) + j] / a;
		}
	}
}

void initializeZero(float *variable, int col, int row)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		variable[(i * col) + j] = 0.0;
		}
	}
}

float meanCalc(float *variable, int col, int row)
{
	float sum = 0;
        float mean = 0;
        float mu = 0;
	for (int i = 0;i < row; i++)
	{
		for (int j = 0;j < col; j++)
		{
		sum += fabs(variable[(i*col) + j]);
		}
	}
	mean = sum / (col*row);
	mu = 1 / mean;
	return mu;
}

void TransposeOnCPU(float *matrix, float *matrixTranspose, int row, int col)
{

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
		matrixTranspose[j*row + i] = matrix[i*col + j];
		}
	}
}


void cpuTransMatrixMult(float *A, float *B, float *C, int row, int col)
{
        float fSum;
	for (int i = 0; i < row; i++)
	{		      
		for (int j = 0; j < row; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < col; k++)
			{
			fSum += (A[(i*col) + k] * B[(k*row) + j]);
			}
		C[(i*row) + j] = fSum;
		}
	}
}

void initialize(float *variable,float *variable2, int col, int row)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		variable[(i * col) + j] = variable2[(i * col) + j];
		}
	}
}

void cpuMatrixMult(float *A, float *B, float *C, int row, int col,int col2)
{
	float fSum;
	int count = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col2; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < col; k++)
			{
			fSum += (A[(i*col) + k] * B[(k*col2) + j]);
			}
		count++;
		C[(i*col2) + j] = fSum;
		}
	}
}

void scalarToMatrixMultiply(float *Temp, float *M, float mu, int row, int col)
{
	for (int i = 0;i < row;i++)
	{		       
		for (int j = 0;j < col;j++)
		{
		Temp[(i*col) + j] = mu * M[(i*col) + j];
		}
	}
}

void sumOfMatrix(float *Znum,float *temp2, float *temp3, float *temp4, int row, int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		Znum[(i*col) + j] = temp2[(i*col) + j] + temp3[(i*col) + j] + temp4[(i*col) + j];
		}
	}
}

void addScalarToDiagonal(float *Zden, float *BBt, float mu, int row, int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
			if(i==j)
			{
			Zden[(i*col) + j] = BBt[(i*col) + j] + mu;
                        }
			else
			{
			Zden[(i*col) + j] = BBt[(i*col) + j] + 0.0;
			}
		}
	}
}

void eye(float *I, int m, int n)
{
        for(int i=0;i<n;i++)
        {
                for(int j=0;j<n;j++)
                {
                        if(i==j)
                        {
                                I[i*n+j] = 1.0;
                        }
                        else
                        {
                                I[i*n+j] = 0.0;
                        }
                }
        }
}



lapack_int matInv(float *A, int n)
{
	int ipiv[n+1];
	lapack_int ret;
	
	ret = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,n,n,A,n,ipiv);
	
	if(ret != 0)
	{
		return ret;
	}
	
	ret = LAPACKE_sgetri(LAPACK_ROW_MAJOR,n,A,n,ipiv);
	return ret;

}	
																					
void calculateZ(float *Z,float *BBt,float *xy, float *E, float *T, float *B_transpose, float mu, float *M, float *Y,const int row,const int col,const int row1)
{
	float *temp = new float [row*col];
	float *temp2 = new float [row*row1];
	float *temp3 = new float [row*row1]; 
	float *Znum = new float [row*row1];
	float *Zden = new float [row1*row1];
	int status = 0;
	high_resolution_clock::time_point t1,t2,t3,t4;

	//numerator
	//temp = (W-E-T*ones(1,p))
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		temp[(i*col) + j] = xy[(i*col) + j] - E[(i*col) + j] - T[i];
		}
	}


	
	//temp2 = temp * B'
	cpuMatrixMult(temp, B_transpose, temp2, row, col, row1);
	//displayValues(temp2,row*row1);

	//temp3 = mu*M
	scalarToMatrixMultiply(temp3, M, mu, row, row1);

	//Znum = ((W-E-T*ones(1,p))*B'+mu*M+Y) 
	sumOfMatrix(Znum,temp2, temp3, Y, row, row1);

	//denominator
	addScalarToDiagonal(Zden,BBt,mu,row1,row1);
	
	status = matInv(Zden,row1);
	
	//Z = ((W-E-T*ones(1,p))*B'+mu*M+Y)/(BBt+mu*eye(3*k))
        cpuMatrixMult(Znum, Zden, Z, row, row1, row1);	
	
	delete [] temp;	
	delete [] temp2;
	delete [] temp3;
	delete [] Znum;
	delete [] Zden;

}

void calculateZ_preZden(float *Z,float *Zden,float *xy, float *E, float *T, float *B_transpose, float mu, float *M, float *Y,const int row,const int col,const int row1)
{
	//calculateZ_preZden(Z, Zden, xy, E, T, B_transpose,mu_orig,M,Y,row,col,row1);

        float *temp = new float [row*col];
        float *temp2 = new float [row*row1];
        float *temp3 = new float [row*row1];
        float *Znum = new float [row*row1];
        int status = 0;
        high_resolution_clock::time_point t1,t2,t3,t4;

        //numerator
        //temp = (W-E-T*ones(1,p))
        for (int i = 0;i < row;i++)
        {
                for (int j = 0;j < col;j++)
                {
                temp[(i*col) + j] = xy[(i*col) + j] - E[(i*col) + j] - T[i];
                }
        }
        //temp2 = temp * B'
        cpuMatrixMult(temp, B_transpose, temp2, row, col, row1);

        //temp3 = mu*M
        scalarToMatrixMultiply(temp3, M, mu, row, row1);

        //Znum = ((W-E-T*ones(1,p))*B'+mu*M+Y) 
        sumOfMatrix(Znum,temp2, temp3, Y, row, row1);

	//Z = ((W-E-T*ones(1,p))*B'+mu*M+Y)/(BBt+mu*eye(3*k))
        cpuMatrixMult(Znum, Zden, Z, row, row1, row1);

	delete [] temp;
        delete [] temp2;
        delete [] temp3;
        delete [] Znum;

}

void differenceOfMatrix(float *diffMatrix, float *matrix1, float *matrix2, int row, int col)
{
        for (int i = 0;i < row;i++)
        {
                for (int j = 0;j < col;j++)
                {
                diffMatrix[(i*col) + j] = matrix1[(i*col) + j] - matrix2[(i*col) + j] ;
                }
        }
}

void calculateQ(float *Q, float *Z, float *Y,float mu, int row, int row1)
{
	float *temp = new float [row*row1];

	scalarToMatrixMultiply(temp, Y, 1/mu, row, row1);
	differenceOfMatrix(Q, Z, temp, row, row1);

	delete[] temp;

}


void prox_2norm(float *Q, float *M, float *C, float constant, int row, int col, int data_size)
{

	float *Q_re = new float [row*col];
	MKL_INT m = ROW, n = COL, lda = LDA, ldu = LDU, ldvt = LDVT, info;
	float superb[min(ROW,COL)-1];
	
	for(int i = 0;i < data_size;i++)
	{
		for(int j = 0;j<2;j++)
		{
			for(int k=0;k<3;k++)
			{
			
			Q_re[(i*6)+(j * 3) + k] = Q[(3 * i) + (j*col) + k];
			}
		}
	}
omp_set_num_threads(24);
#pragma omp parallel for 
	for(int i = 0;i < data_size;i++)
	{
	
	float *Qtemp = new float [6];
	float *sigma = new float [COL];
	float *u = new float [LDU*ROW];
	float *vt = new float [LDVT*COL];

	float *sigma1 = new float [ROW*ROW];
	float *vt1 = new float [ROW*COL];
	float *Qtemp1 = new float [4];
	float *Qtemp2 = new float [6];


		for(int j = 0;j<6;j++)
		{
			Qtemp[j] = Q_re[(i*6)+j];
		}
		info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, Qtemp, lda, sigma, u, ldu, vt, ldvt, superb);

		if(info > 0)
		{
			cout << "The algorithm computing SVD failed to converge" << endl;
		}
		
		if((sigma[0]+sigma[1]) <= constant )
		{
			sigma[0] = 0;
			sigma[1] = 0;
		}
		else if ((sigma[0] - sigma[1]) <= constant)
		{
			sigma[0] = ((sigma[0]+sigma[1])-constant)/2;
			sigma[1] = sigma[0];
		}
		else
		{
			sigma[0] = sigma[0] - constant;
			sigma[1] = sigma[1];
		}

		
		for(int j = 0;j<ROW;j++)
		{
			for(int k =0;k<COL;k++)
			{
				vt1[(j*COL)+k] = vt[(j*COL) + k];
			}
		}	
		for(int j = 0;j<ROW;j++)
		{
			for(int k =0;k<ROW;k++)
			{
				if(j == k)
				{
				sigma1[(j*ROW)+k] = sigma[j];
				}
				else
				{
				sigma1[(j*ROW)+k] = 0.0;
				}
			}
		}	
		cpuMatrixMult(u,sigma1,Qtemp1,ROW,ROW,ROW);
		cpuMatrixMult(Qtemp1,vt1,Qtemp2,ROW,ROW,COL);
		for(int j = 0;j<2;j++)
                {
                        for(int k=0;k<3;k++)
                        {
                        M[(3 * i) + (j*col) + k] = Qtemp2[(j * 3) + k];
                        }
                }
		

		C[i] = sigma1[0];

	delete[] Qtemp;
	delete[] sigma;
	delete[] u;
	delete[] vt;
	delete[] Qtemp1;
	delete[] Qtemp2;
	delete[] sigma1;
	delete[] vt1;

	}

	delete[] Q_re;
}


void updateDualvariable(float *Y,float mu,float *M,float *Z,int row,int row1)
{
	for(int i=0;i<row;i++)
	{
		for(int j = 0;j<row1;j++)
		{
			Y[(i*row1)+j] += mu*(M[(i*row1)+j] - Z[(i*row1)+j]); 
		}
	}
}

float febNorm(float *a, int row, int col)
{
        float norm = 0.0;
        float sum = 0.0;
        for(int i=0;i<row;i++)
        {
                for(int j=0;j<col;j++)
                {
                  sum +=(a[(i*col)+j]) * (a[(i*col)+j]);
                }
        }
        norm=sqrt(sum);
        return norm;
}

void resCalc(float *PrimRes, float *DualRes, float *M, float *Z, float *ZO,float mu, int row, int row1)
{
	float *MminusZ = new float [row*row1];
	float *ZminusZO = new float [row*row1];

	for(int i = 0; i< row ;i++)
	{
		for(int j = 0; j<row1 ; j++)
		{
			MminusZ[(i*row1)+j] = M[(i*row1)+j] - Z[(i*row1)+j];
			ZminusZO[(i*row1)+j] = Z[(i*row1)+j] - ZO[(i*row1)+j];
		}
	}
	
		
	*PrimRes = febNorm(MminusZ,row,row1)/febNorm(ZO,row,row1);
	*DualRes = mu * febNorm(ZminusZO,row,row1)/febNorm(ZO,row,row1);
	
	delete[] MminusZ;
	delete[] ZminusZO;
}

int main(void)
{
	const int iter_num = 1;
	high_resolution_clock::time_point t1[iter_num],t2[iter_num],t3,t4;
	for(int p = 0;p<iter_num;p++)
	{	//t3 = high_resolution_clock::now();

	
	const int row = 2;
	const int col = 15;
	const int row1 = 384;
	const int col1 = 15;
	float tol = 1e-04;

	float *xy = new float [row*col];
	float *mean = new float [row];
	float *B = new float [row1*col1];
	float *B_transpose = new float [col1*row1];
	float *B_mean = new float [row1];
	float *BBt = new float [row1*row1];
	int items = 0;
	float a = 0.0;
	int B_items = 0;
	int lam =1;
	bool verb = true;

	const int data_size = row1/3;	
	
	//ssr2D3D_alm
	//M => (2*384) = 0,  C ==> (1*384) = 0, E ==> (2*15) = 0, T ==> mean(W,2)
	float *M = new float [row*row1];
	float *C = new float [data_size];
	float *E = new float [row*col];
	float *T = new float [row];
	
	// auxiliary variables for ADMM
	float *Z = new float [row*row1];
	float *Y = new float [row*row1];
	float *ZO = new float [row*row1];
	float *Q = new float [row*row1];
	float mu = 0.0;
	float PrimRes;
	float DualRes;

	//allocate precomputed Zden
	float *Zden = new float [row1*row1];
	float *Zden1 = new float [row1*row1];
	float *Zden2 = new float [row1*row1];
	float *Zden3 = new float [row1*row1];
	float *Zden4 = new float [row1*row1];
        int status = 0;
	float mu_orig = 0.0;	
	float mu1 = 0.0;
	float mu2 = 0.0;
	float mu3 = 0.0;
	float mu4 = 0.0;
	float *Zden_inv = new float [row1*row1];
	float *Zden1_inv = new float [row1*row1];

	//t1[p] = high_resolution_clock::now();
	t1[p] = high_resolution_clock::now();
	items = readValues("messi2.txt",xy,items,row,col);
	rowMean(xy, col, row, mean);
        Scalc(xy, col, row, mean);
        rowMean(xy, col, row, mean);
	a = mean_of_std_deviation(xy,col,row,mean);
	newScalc(xy,col,row,a);
	//displayValues(xy,items);
	B_items = readValues("exp1.txt", B, B_items,row1,col1);
	rowMean(B,col1,row1,B_mean);
	Scalc(B, col1,row1,B_mean);
	
	initializeZero(M, row1,row);
	initializeZero(C, data_size, 1);
	initializeZero(E,col,row);
	rowMean(xy,col,row,T);

	initializeZero(Z,row1,row);
	initializeZero(Y,row1,row);
	
	//displayValues(xy,items);
	
	mu = meanCalc(xy,col,row);
	//cout << "value of mu is " << mu << endl;

        //calculation of BBt
        TransposeOnCPU(B,B_transpose,row1,col);
        cpuTransMatrixMult(B, B_transpose, BBt, row1, col);
        //Zden

	//Precompute mu
	mu_orig = mu;
	mu1 = mu * 2;
	//mu2 = mu1 * 2;
	//mu3 = mu / 2;
	//mu4 = mu3 / 2;

	addScalarToDiagonal(Zden,BBt,mu_orig,row1,row1);
	addScalarToDiagonal(Zden1,BBt,mu1,row1,row1);
	//addScalarToDiagonal(Zden2,BBt,mu2,row1,row1);
	//addScalarToDiagonal(Zden3,BBt,mu3,row1,row1);
	//addScalarToDiagonal(Zden4,BBt,mu4,row1,row1);
 	
	//dump_to_file("Z_old.txt",Z,row1,row1);				
	
	eye(Zden_inv,row1,row1);
	eye(Zden1_inv,row1,row1);
	t3 = high_resolution_clock::now();
	gpuInverseOfMatrix(Zden,Zden_inv,row1);
	gpuInverseOfMatrix(Zden1,Zden1_inv,row1);
//	status = matInv(Zden,row1);
//	status = matInv(Zden1,row1);
	//status = matInv(Zden2,row1);
	//status = matInv(Zden3,row1);
	//status = matInv(Zden4,row1);	
	
	t4 = high_resolution_clock::now();
	duration<float> time_span = duration_cast<duration<float>>(t4 - t3);
	cout << "Time in miliseconds for first section is : " << time_span.count() * 1000 << " ms" << endl;
	
	for(int iter = 0; iter < 500; iter++)
	{
		//t1 = high_resolution_clock::now();
		initialize(ZO,Z,row1,row);

		//pre_computed part
		if(mu == mu_orig)
		{
			calculateZ_preZden(Z, Zden_inv,xy, E, T, B_transpose,mu_orig,M,Y,row,col,row1);
		}
		else if(mu == mu1)
		{
			calculateZ_preZden(Z, Zden1_inv,xy, E, T, B_transpose,mu1,M,Y,row,col,row1);
		}
		else if(mu == mu2)
		{
			calculateZ_preZden(Z, Zden2,xy, E, T, B_transpose,mu2,M,Y,row,col,row1);	
		}
		else if(mu == mu3)
		{
			calculateZ_preZden(Z, Zden3,xy, E, T, B_transpose,mu3,M,Y,row,col,row1);
		}
		else if(mu == mu4)
		{
			calculateZ_preZden(Z, Zden4,xy, E, T, B_transpose,mu4,M,Y,row,col,row1);
		}
		else
		{
			calculateZ(Z, BBt,xy, E, T, B_transpose,mu,M,Y,row,col,row1);
		}
		
		calculateQ(Q,Z,Y,mu,row,row1);
	
		prox_2norm(Q,M,C,lam/mu,row,row1,data_size);
		
		updateDualvariable(Y,mu,M,Z,row,row1);
		resCalc(&PrimRes,&DualRes,M,Z,ZO,mu,row,row1);
		
		//if ((verb == true) && ((iter%10) == 0))
		//{
//			cout << "Iter "<< iter+1 <<": PrimRes = "<<PrimRes <<", DualRes = "<<DualRes<<", mu = "<< mu <<endl; 
		//}

		if((PrimRes < tol) && (DualRes < tol))
		{
		break;
		}
		else
		{
			if(PrimRes > (10*DualRes))
			{
				mu = 2 * mu;
			}
			else if(DualRes > (10*PrimRes))
			{
				mu = mu/2;
			}
			else
			{
			}
		}
		//t2 = high_resolution_clock::now();
		//duration<float> time_span = duration_cast<duration<float>>(t2 - t1);
		//cout << "Time in miliseconds: " << time_span.count() * 1000 << " ms" << endl;

	}
	t2[p] = high_resolution_clock::now();

	delete[] xy;
        delete[] mean;
	delete[] B;
	delete[] B_transpose;
	delete[] B_mean;
	delete[] BBt;
	delete[] M;
	delete[] C;
	delete[] E;
	delete[] T;
	delete[] Z;
	delete[] Y;
	delete[] ZO;
	delete[] Q;
	delete[] Zden;
	delete[] Zden1;
	delete[] Zden2;
	delete[] Zden3;
	delete[] Zden4;
	
	delete[] Zden_inv;
	delete[] Zden1_inv;

	//duration<float> time_span = duration_cast<duration<float>>(t2 - t1);
	//cout << "Time in miliseconds: " << time_span.count() * 1000 << " ms" << endl;
	}	
	duration<float> time_span;
	for(int p=0;p<iter_num;p++)
	{
		time_span += duration_cast<duration<float>>(t2[p] - t1[p]);
	}	
	cout << "Time in miliseconds: "<< (time_span.count()/iter_num) * 1000 << " ms"<<endl; 
}

