#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <limits>
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

extern void loop(float *xy,float *B,float *Bt,float *Zden,float *Z,float *ZO,float *T,float *M,float *Y,float *Q,float *Q_re,float *C,float *prim, float *dual,int row,int col,int row1,int col1,float mu,float lam,int data_size);


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

void print_matrix( char *desc, int m, int n, float *a) 
{
        int i, j;
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
		cout << "Value of variable :"<< i << " = " << variable[i] << endl;
	}
}

void normalizeS(float *variable, int row, int col,float *T)
{
	float sum;
	float std[2];
	float mean;
	float temp,a;
	for (int i = 0;i < row; i++)
	{
		sum = 0;
		mean = 0;
		for (int j = 0;j < col; j++)
		{
			sum += variable[(i*col)+ j];
		}
		mean = sum / col;
		
		for (int j = 0;j < col;j++)
		{
			variable[(i * col) + j] = variable[(i * col) + j] - mean;
		}
		
		sum = 0;
		for (int j = 0;j < col; j++)
		{
			sum += variable[(i*col)+ j];
		}
		mean = sum / col;
		temp = 0;
		for (int j = 0;j < col;j++)
		{
		temp += pow((variable[(i * col) + j] - mean), 2);
		}
	std[i] = sqrt(temp / col);
        }	
	a = (std[0] + std[1]) / 2;
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		variable[(i * col) + j] = variable[(i * col) + j] / a;
		}
	}
	
	for (int i = 0;i < row; i++)
	{
		sum = 0;
		for (int j = 0;j < col; j++)
		{									                
			sum += variable[(i*col)+ j];
		}
		T[i] = sum / col;
	}

}


void centralizeB(float *variable,int row,int col)
{
	float sum,mean;
	for (int i = 0;i < row; i++)
	{
		sum = 0;
		for (int j = 0;j < col; j++)
		{
			sum += variable[(i*col)+ j];
		}
		mean = sum / col;

		for (int j = 0;j < col;j++)
		{
			variable[(i * col) + j] = variable[(i * col) + j] - mean;
		}
	}
}

void initializeMat(float *M,float *Z,float *Y,int row,int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		M[(i * col) + j] = 0.0;
		Z[(i * col) + j] = 0.0;
		Y[(i * col) + j] = 0.0;
		}
	}

}

void initializeZero(float *variable, int row, int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		variable[(i * col) + j] = 0.0;
		}
	}
}

void initializeVec(float *variable, int col)
{
		for (int i = 0;i < col;i++)
		{
		variable[i] = 0.0;
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
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col2; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < col; k++)
			{
			fSum += (A[(i*col) + k] * B[(k*col2) + j]);
			}
		C[(i*col2) + j] = fSum;
		}
	}
}

void sumOfMatrix(float *Znum,float *temp2, float *M, float *temp4,float mu, int row, int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		Znum[(i*col) + j] = temp2[(i*col) + j] + mu * M[(i*col) + j] + temp4[(i*col) + j];
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


void svd_2_3_alt(float *a, float *u,float *sig, float *vt, int m,int n)
{
	float T = 0;
	float D = 0;
	float lam1 = 0;
	float lam2 = 0;
	float u1_norm = 0;
	float u2_norm = 0;
	float sig_inv[m*m];
	float temp[m*m];
	float u_t[m*m];
	float aat[m*m];
	float temp_var;
	float fSum;
		
	aat[0] = a[0]*a[0]+ a[1]*a[1] + a[2]*a[2];
	aat[1] = a[0]*a[3]+ a[1]*a[4] + a[2]*a[5];
	aat[2] = a[3]*a[0]+ a[4]*a[1] + a[5]*a[2];
	aat[3] = a[3]*a[3]+ a[4]*a[4] + a[5]*a[5]; 

	T = aat[0] + aat[3];
	D = aat[0] * aat[3] - aat[1] * aat[2];
	lam1 = 0.5*(T + (sqrt((T*T)-4*D)));
	lam2 = 0.5*(T - (sqrt((T*T)-4*D)));

	u[0] = aat[1];
	u[2] = lam1 - aat[0];
	u[1] = aat[1];
	u[3] = lam2 - aat[0];
	u1_norm =1/sqrt(u[0]*u[0]+u[2]*u[2]);
	u2_norm =1/sqrt(u[1]*u[1]+u[3]*u[3]);

	//final u
	u[0] = u[0]*u1_norm;
	u[2] = u[2]*u1_norm;
	u[1] = u[1]*u2_norm;
	u[3] = u[3]*u2_norm;

	//u_transpose
	u_t[0] = u[0];
	u_t[1] = u[2];
	u_t[2] = u[1];
	u_t[3] = u[3];
	
	//sigma 
	sig[0] = sqrt(lam1);
	sig[1] = 0;
	sig[2] = 0;
	sig[3] = sqrt(lam2);

	//sigma_inv
	sig_inv[0] = 1/sig[0];
	sig_inv[1] = 0;
	sig_inv[2] = 0;
	sig_inv[3] = 1/sig[3];

	//vt
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (sig_inv[(i*m) + k] * u_t[(k*m) + j]);
			}
		temp[(i*m) + j] = fSum;
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (temp[(i*m) + k] * a[(k*n) + j]);
			}
		vt[(i*n) + j] = fSum;
		}
	}
}

void prox_2norm(float *Q, float *M, float *C, float constant, int row, int col, int data_size)
{

//omp_set_num_threads(24);
//#pragma omp parallel for 
	for(int i = 0;i < data_size;i++)
	{
	
	float *Qtemp = new float [6];
	float *sigma = new float [COL];
	float *u = new float [LDU*ROW];
	float *vt = new float [ROW*COL];

	float *Qtemp1 = new float [4];
	float *Qtemp2 = new float [6];


		for(int j = 0;j<6;j++)
		{
			Qtemp[j] = Q[(i*6)+j];
		}
		svd_2_3_alt(Qtemp,u,sigma,vt,ROW,COL);
		
		if((sigma[0]+sigma[3]) <= constant )
		{
			sigma[0] = 0;
			sigma[3] = 0;
		}
		else if ((sigma[0] - sigma[3]) <= constant)
		{
			sigma[0] = ((sigma[0]+sigma[3])-constant)/2;
			sigma[3] = sigma[0];
		}
		else
		{
			sigma[0] = sigma[0] - constant;
			sigma[3] = sigma[3];
		}

		cpuMatrixMult(u,sigma,Qtemp1,ROW,ROW,ROW);
		cpuMatrixMult(Qtemp1,vt,Qtemp2,ROW,ROW,COL);
		for(int j = 0;j<2;j++)
                {
                        for(int k=0;k<3;k++)
                        {
                        M[(3 * i) + (j*col) + k] = Qtemp2[(j * 3) + k];
                        }
                }
		

		C[i] = sigma[0];

	delete[] Qtemp;
	delete[] sigma;
	delete[] u;
	delete[] vt;
	delete[] Qtemp1;
	delete[] Qtemp2;

	}

}

void reconstruct(float *R,float *B,float *C,float *M,float *xyz,float *kron,float *temp_mult,int row,int col,int row1, int col1,int data_size,int threeD_row)
{
	for(int i = 0;i<data_size;i++)
	{
		if(C[i] != 0)
		{
			for(int j = 0;j<3;j++)
                	{
				if(j == 0 || j == 1)
				{
                        		for(int k=0;k<3;k++)
                        		{
	                       		R[(3 * i) + (j*row1) + k] = M[(3 * i) + (j*row1) + k] / C[i];
        	               		}
				}
				else
				{
					
	                       	R[(3 * i) + (j*row1) + 0] = R[(3 * i) + (0*row1) + 1] * R[(3 * i) + (1*row1) + 2] -  R[(3 * i) + (0*row1) + 2] * R[(3 * i) + (1*row1) + 1] ;
	                       	R[(3 * i) + (j*row1) + 1] = R[(3 * i) + (0*row1) + 2] * R[(3 * i) + (1*row1) + 0] -  R[(3 * i) + (0*row1) + 0] * R[(3 * i) + (1*row1) + 2] ;
	                       	R[(3 * i) + (j*row1) + 2] = R[(3 * i) + (0*row1) + 0] * R[(3 * i) + (1*row1) + 1] -  R[(3 * i) + (0*row1) + 1] * R[(3 * i) + (1*row1) + 0] ;
        	               		
				}
                	}
		}
	}

	for(int i=0;i<data_size;i++)
	{
		for(int j=0;j<data_size;j++)
		{
			if(i == j)
			{
			for(int k=0;k<3;k++)
			{
				for(int l=0;l<3;l++)
				{
					if(k == l)
					{
					kron[(3*row1*j)+(3*i)+(k*row1)+l] = C[i]*1;
					}
					else
					{
					kron[(3*row1*j)+(3*i)+(k*row1)+l] = C[i]*0;
					}
				}
			}
			}
		}
	}

	cpuMatrixMult(kron, B, temp_mult, row1, row1,col);
	cpuMatrixMult(R, temp_mult, xyz, threeD_row, row1,col);

}


int main(void)
{
	
	const int row = 2;
	const int col = 15;
	const int row1 = 384;
	const int col1 = 15;
	const int threeD_row = 3;
	float tol = 1e-04;

	float *xy = new float [row*col];
	float *xyz = new float [threeD_row*col];
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
	int flag = 0;
	int iter = 0;
	int count1 = 0;

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
	float *Q_re = new float [row*row1];
	float *iden = new float [col*col];
	float *I_m = new float [row1];
	float *h_temp_Bt_mui = new float [col*row1];

	float mu = 0.0f;
	float mu_inv = 0.0f;
	float constant = 0.0f;
	float prim=0.0f;
	float dual=0.0f;

	//allocate precomputed Zden
	float *Zden = new float [row1*row1];
        int status = 0;
	float *kron = new float [row1*row1];
	float *temp_mult = new float [row1*col1];

	float *R = new float [threeD_row*row1];

	//read the 15 points from 15 point model
	items = readValues("messi2.txt",xy,items,row,col);
	
	//normalize the input
	normalizeS(xy,row,col,T);

	//read the dictionary
	B_items = readValues("B_128.txt", B, B_items,row1,col1);

	//centralize basis
	centralizeB(B,row1,col1);
	
	//initialization
	initializeMat(M,Z,Y,row,row1);
	initializeVec(C,data_size);
	initializeZero(E,row,col);
	initializeZero(R,threeD_row,col);

	mu = meanCalc(xy,col,row);

	loop(xy,B,B_transpose,Zden,Z,ZO,T,M,Y,Q,Q_re,C,&prim,&dual,row,col,row1,col1,mu,lam,data_size);

	reconstruct(R,B,C,M,xyz,kron,temp_mult,row,col,row1,col1,data_size,threeD_row);


	displayValues(xyz,threeD_row*col);
	//displayValues(temp_mult,row1);


	delete[] xy;
	delete[] xyz;
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
	delete[] Q_re;
	delete[] iden;
	delete[] Zden;
	delete[] kron;
	delete[] temp_mult;
	delete[] I_m;
	delete[] h_temp_Bt_mui;
	delete[] R;

}

