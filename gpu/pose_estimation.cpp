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

//extern void gpuInverseOfMatrix(float *h_matrix,float *h_iden_mat, int col);
extern void loop_cu(float *xy, float *B, float *B_t, float *Z, float *ZO,float *Zden, float *Y, float *Q, float *Q_re,float *M, float *C,float *E, float *T, float *iden, float *I_m, float mu, float constant, int row, int col, int row1, int col1, int data_size,float *temp_mui_B);
//extern void gpuProx_2norm(float *Q, float *M, float *C, float constant, int row, int col, int data_size);
//extern void gpuMultShared(float *h_A, float *h_B, float *h_C, const int A_rows, const int A_cols,const int B_rows,const int B_cols);

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
		cout << "Value of variable :"<< variable[i] << endl;
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

/*
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
*/

void inv_mu_i(float mu,float *I,int m)
{
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<m;j++)
		{
			if(i == j)
			{
			I[(i*m)+j] = 1 / (I[(i*m)+j] * mu);
			}
		}
	}
}

void add_iden(float *temp1,int n)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			if(i==j)
			{
			temp1[(i*n)+j] = temp1[(i*n)+j] + 1.0f;
			}
		}
	}

}

void sub_wood(float *I_n,float *temp1,float *inv,int m,int n)
{
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			inv[(i*n)+j] = I_n[(i*n)+j] - temp1[(i*n)+j];
		}
	}
}

void cpuInverseOfMatrix(float *matrix, float *I, int col)
{

	for (int m = 0; m < col; m++)
	{
		//Checking if diagonal element is 0
		if (matrix[((col) + 1)*m] == 0)
		{
			//checking if the row is last row. If it is last row add the previous row to make it non zero
                	if (m == (col - 1))
			{
				for (int i = 0; i < (col); i++)
				{					
				matrix[(m * (col)) + i] = matrix[((m - 1) * (col)) + i] + matrix[(m * (col)) + i];
				I[(m * (col)) + i] = matrix[((m - 1) * (col)) + i] + matrix[(m * (col)) + i];
				}
			}
			else	//if it is not last row, add the next row.
			{
			        for (int i = 0; i < (2 * col); i++)
				{
				matrix[(m * col) + i] = matrix[((m + 1) * col) + i] + matrix[(m * col) + i];
				I[(m * col) + i] = matrix[((m + 1) * col) + i] + matrix[(m * col) + i];
				}
			}
		}

		float initialValue = matrix[((col) + 1)*m];

		//Make the diagonal elements 1 along with the whole row(divide).
		for (int j = 0; j < (col); j++)
		{
		matrix[(m * (col)) + j] = matrix[(m * (col)) + j] / initialValue;
		I[(m * (col)) + j] = I[(m * (col)) + j] / initialValue;
		}

//omp_set_num_threads(24);
//#pragma omp parallel for
		//Making the elements of the row to zero
		for (int k = 0; k < col; k++)
		{
			float tempIni;
			tempIni = matrix[m + (k * (col))];
			if (k != m)
			{	
				for (int l = 0; l < (col); l++)
				{
					matrix[k*col+l] = matrix[k*col+l] - ((tempIni*matrix[m*col+l])/matrix[m*col+m]);
					I[k*col+l] = I[k*col+l] - ((tempIni*I[m*col+l])/matrix[m*col+m]);

				}
			}

		}

	}
}

void Zden_cacl(float *B, float *Bt, float *Zden,float mu,const int m,const int n)
{

	float mu_inv =0.0f;
        float *I_m = new float [m];
	float *temp_mui_B = new float [m*n];
	float *temp_Bt_mui = new float [m*n];
	float *temp_inv = new float [n*n];
	float *temp_I = new float [n*n];
	float *temp_mult = new float [n*m];
	float *temp_sub = new float [m*m];

	mu_inv = 1/mu;

	for(int i = 0;i<m;i++)
	{
		I_m[i] = mu_inv;
	}
	for(int i = 0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			temp_mui_B[(i*n)+j] = I_m[i] * B[(i*n)+j];
			temp_Bt_mui[(i*n)+j] = Bt[(i*n)+j] * I_m[j];
		}
	}
	
	float fSum;
        for (int i = 0; i < n; i++)
        {
                for (int j = 0; j < n; j++)
                {
                fSum = 0.0;
                        for (int k = 0; k < m; k++)
                        {
                        fSum += (Bt[(i*m) + k] * temp_mui_B[(k*n) + j]);
                        }
		if(i==j)
		{
                temp_inv[(i*n) + j] = fSum + 1;
		}
		else
		{
                temp_inv[(i*n) + j] = fSum;
		}
                }
        }	

	//matInv(temp_inv,n);
	eye(temp_I,n,n);
	cpuInverseOfMatrix(temp_inv, temp_I, n);
	
	for (int i = 0; i < n; i++)
        {
                for (int j = 0; j < m; j++)
                {
                fSum = 0.0;
                        for (int k = 0; k < n; k++)
                        {
                        fSum += (temp_I[(i*n) + k] * temp_Bt_mui[(k*m) + j]);
                        }
                temp_mult[(i*m) + j] = fSum;
                }
        }	

omp_set_num_threads(24);
#pragma omp parallel for
	for (int i = 0; i < m; i++)
        {
                for (int j = 0; j < m; j++)
                {
                fSum = 0.0;
                        for (int k = 0; k < n; k++)
                        {
                        fSum += (temp_mui_B[(i*n) + k] * temp_mult[(k*m) + j]);
                        }
		if(i==j)
		{
			Zden[(i*m)+j] = I_m[i] - fSum;
			//temp_sub[(i*m)+j];
		}
		else
		{
			Zden[(i*m)+j] = 0.0 - fSum;
			//temp_sub[(i*m)+j];
		}
                //temp_sub[(i*m) + j] = fSum;
                }
        }	
       
	delete [] I_m;
	delete [] temp_mui_B;
	delete [] temp_Bt_mui;
	delete [] temp_inv;
	delete [] temp_mult;
	delete [] temp_sub;

}


void calculateZ_preZden(float *Z,float *Zden,float *xy, float *E, float *T, float *B_transpose, float mu, float *M, float *Y,const int row,const int col,const int row1)
{
	//calculateZ_preZden(Z, Zden, xy, E, T, B_transpose,mu_orig,M,Y,row,col,row1);

        float *temp = new float [row*col];
        float *temp2 = new float [row*row1];
        float *temp3 = new float [row*row1];
        float *Znum = new float [row*row1];
        int status = 0;

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
        //CPU//
	cpuMatrixMult(temp, B_transpose, temp2, row, col, row1);
        //GPU//
	//gpuMultShared(temp, B_transpose, temp2, row, col, col, row1);

        //temp3 = mu*M

        //Znum = ((W-E-T*ones(1,p))*B'+mu*M+Y) 
        sumOfMatrix(Znum,temp2, M, Y, mu, row, row1);

	//Z = ((W-E-T*ones(1,p))*B'+mu*M+Y)/(BBt+mu*eye(3*k))
        //CPU//
	cpuMatrixMult(Znum, Zden, Z, row, row1, row1);
	//GPU//
	//gpuMultShared(Znum, Zden, Z, row, row1, row1, row1);

	delete [] temp;
        delete [] temp2;
        delete [] temp3;
        delete [] Znum;

}

void calculateQ(float *Q,float *Q_re, float *Z, float *Y,float mu, int row, int row1,int data_size)
{
	float oneovermu;
	oneovermu = 1/mu;
	for (int i = 0;i < row;i++)
        {
                for (int j = 0;j < row1;j++)
                {
                Q[(i*row1) + j] = Z[(i*row1) + j] - ((oneovermu)*Y[(i*row1) + j]) ;
                }
        }

	for(int i = 0;i < data_size;i++)
        {
                for(int j = 0;j<2;j++)
                {
                        for(int k=0;k<3;k++)
                        {

                        Q_re[(i*6)+(j * 3) + k] = Q[(3 * i) + (j*row1) + k];
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

omp_set_num_threads(24);
#pragma omp parallel for 
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

float resCalc_PrimRes(float *M, float *Z, float *ZO,float mu, int row, int row1)
{
	float *MminusZ = new float [row*row1];
	float temp = 0.0f;

	for(int i = 0; i< row ;i++)
	{
		for(int j = 0; j<row1 ; j++)
		{
			MminusZ[(i*row1)+j] = M[(i*row1)+j] - Z[(i*row1)+j];
		}
	}
	
		
	temp = febNorm(MminusZ,row,row1)/febNorm(ZO,row,row1);
	
	delete[] MminusZ;
	return temp;
}

float resCalc_DualRes(float *Z, float *ZO,float mu, int row, int row1)
{
	float *ZminusZO = new float [row*row1];
	float temp = 0.0f;

	for(int i = 0; i< row ;i++)
	{
		for(int j = 0; j<row1 ; j++)
		{
			ZminusZO[(i*row1)+j] = Z[(i*row1)+j] - ZO[(i*row1)+j];
		}
	}	
		
	temp = mu * febNorm(ZminusZO,row,row1)/febNorm(ZO,row,row1);
	
	delete[] ZminusZO;
	return temp;
}


int main(void)
{
	
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
	float *h_temp_mui_B = new float [row1*col];

	float mu = 0.0f;
	float mu_inv = 0.0f;
	float constant = 0.0f;
	float PrimRes;
	float DualRes;

	//allocate precomputed Zden
	float *Zden = new float [row1*row1];
        int status = 0;
	float *Zden_inv = new float [row1*row1];

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

	mu = meanCalc(xy,col,row);

	mu_inv = 1/mu;

	for(int i = 0;i<row1;i++)
	{
		I_m[i] = mu_inv;
	}
	
	loop_cu(xy, B, B_transpose, Z, ZO, Zden, Y, Q, Q_re, M, C, E, T, iden, I_m, mu, constant, row, col, row1, col1, data_size,h_temp_mui_B);
	dump_to_file("h_temp_mui_B",h_temp_mui_B,row1,col);

/*        //calculation of BBt
	TransposeOnCPU(B,B_transpose,row1,col);
	//cpuTransMatrixMult(B, B_transpose, BBt, row1, col);
        //gpuMultShared(B,B_transpose,BBt,row1,col,col,row1);
	//addScalarToDiagonal(Zden,BBt,mu,row1,row1);
	
	//gpu inverse
	//eye(Zden_inv,row1,row1);
	//gpuInverseOfMatrix(Zden,Zden_inv,row1);

	//cpu inverse
	//status = matInv(Zden,row1);
	//eye(Zden_inv,row1,row1);
	//cpuInverseOfMatrix(Zden, Zden_inv, row1);

	//woodburry inverse
	Zden_cacl(B,B_transpose,Zden,mu,row1,col);

	for(iter = 0; iter < 500; iter++)
	{
		count1 = iter;
		initialize(ZO,Z,row1,row);
		
		if(flag == 1)
		{
			//cpu inverse
			//addScalarToDiagonal(Zden,BBt,mu,row1,row1);
			//status = matInv(Zden,row1);
			//eye(Zden_inv,row1,row1);
			//cpuInverseOfMatrix(Zden, Zden_inv, row1);

			//gpuinverse
			//addScalarToDiagonal(Zden,BBt,mu,row1,row1);
			//eye(Zden_inv,row1,row1);
			//gpuInverseOfMatrix(Zden,Zden_inv,row1);

			//woodburry inverse
			Zden_cacl(B, B_transpose,Zden,mu,row1,col);
		}
		
		//cpu
		calculateZ_preZden(Z, Zden,xy, E, T, B_transpose,mu,M,Y,row,col,row1);

		//gpu
		//calculateZ_preZden(Z, Zden_inv,xy, E, T, B_transpose,mu,M,Y,row,col,row1);

		calculateQ(Q,Q_re,Z,Y,mu,row,row1,data_size);
	
		//cpu	
		//prox_2norm(Q_re,M,C,lam/mu,row,row1,data_size);

		//gpu
		gpuProx_2norm(Q_re,M,C,lam/mu,row,row1,data_size);	

		updateDualvariable(Y,mu,M,Z,row,row1);
		
		PrimRes = resCalc_PrimRes(M,Z,ZO,mu,row,row1);
		DualRes = resCalc_DualRes(Z,ZO,mu,row,row1);
		
		//if ((verb == true) && ((iter%10) == 0))
		//{
		//	cout << "Iter "<< iter+1 <<": PrimRes = "<<PrimRes <<", DualRes = "<<DualRes<<", mu = "<< mu <<endl; 
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
				flag = 1;
			}
			else if(DualRes > (10*PrimRes))
			{
				mu = mu/2;
				flag = 1;
			}
			else
			{
				flag = 0;
			}
		}
	}
*/
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
	delete[] Q_re;
	delete[] iden;
	delete[] Zden;
	delete[] Zden_inv;
	delete[] I_m;
	delete[] h_temp_mui_B;

}

