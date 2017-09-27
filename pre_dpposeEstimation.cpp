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

int readValues(char *text, double *variable, int i)
{
 	double temp;
	//cout << "value of text " << text << endl;
	ifstream myReadFile;
	myReadFile.open(text, ios::in);
	if (myReadFile.is_open()) 
	{
		while (!myReadFile.eof())
		{
			myReadFile >> temp;
			//cout << "value of variable " << variable[i] << endl;
			variable[i] = temp;
			i++;                                                                                                                                                                     }
	}
  	myReadFile.close();
	return i;
}

void dump_to_file(char *filename, double *matrix, int row, int col)
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

void print_matrix( char *desc, MKL_INT m, MKL_INT n, double *a) 
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

void displayValues(double *variable, int items)
{
	cout.precision(17);
	for (int i =0; i < items; i++)
	{
		cout << "Value of variable :"<< variable[i] << endl;
	}
}

void rowMean(double *variable, int col, int row , double *mean)
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
	        //cout << "value of mean " << mean[j] << endl;
	}
}

void Scalc(double *variable, int col, int row, double *mean)
{
	for (int j = 0;j < row;j++)
	{
		for (int i = 0;i < col;i++)
		{
			variable[(j * col) + i] = variable[(j * col) + i] - mean[j];
		}
	}
}

double mean_of_std_deviation(double *variable, int col, int row, double *mean)
{
	double std[2];
	double temp,a;
	for (int j = 0;j < row;j++)
	{
		temp = 0;
		for (int i = 0;i < col;i++)
		{
		temp += pow((variable[(j * 15) + i] - mean[j]), 2);
		}
	std[j] = sqrt(temp / (col));
	//cout << "value od std deviation " << std[j] << endl;
	}
	a = (std[0] + std[1]) / 2;
	//cout << "value of a " << a << endl;
	return a;
}

void newScalc(double *variable, int col, int row, double a)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		variable[(i * col) + j] = variable[(i * col) + j] / a;
		}
	}
}

void initializeZero(double *variable, int col, int row)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		variable[(i * col) + j] = 0.0;
		}
	}
}

double meanCalc(double *variable, int col, int row)
{
	double sum = 0;
        double mean = 0;
        double mu = 0;
	for (int i = 0;i < row; i++)
	{
		for (int j = 0;j < col; j++)
		{
		sum += fabs(variable[(i*col) + j]);
	//	cout << "value of varialbe is " << fabs(variable[(i*col) + j]) << endl;
		}
	}
	//cout << "value of sum is "<<sum << endl;
	mean = sum / (col*row);
	//cout << "value of mean " << mean << endl;
	mu = 1 / mean;
	return mu;
}

void TransposeOnCPU(double *matrix, double *matrixTranspose, int row, int col)
{

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
		matrixTranspose[j*row + i] = matrix[i*col + j];
		}
	}
}


void cpuTransMatrixMult(double *A, double *B, double *C, int row, int col)
{
        double fSum;
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

void initialize(double *variable,double *variable2, int col, int row)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		variable[(i * col) + j] = variable2[(i * col) + j];
		}
	}
}

void cpuMatrixMult(double *A, double *B, double *C, int row, int col,int col2)
{
	double fSum;
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
		//cout << count<<"\t" <<"value of (i*row) + j \t"<< (i*row) + j <<"\t value of i and j"<<"\t"<<i<<" and "<<j<<"\t value of fsum" << fSum << endl;
		count++;
		C[(i*col2) + j] = fSum;
		}
	}
	//cout << "count =" << count << endl;
}

void scalarToMatrixMultiply(double *Temp, double *M, double mu, int row, int col)
{
	for (int i = 0;i < row;i++)
	{		       
		for (int j = 0;j < col;j++)
		{
		Temp[(i*col) + j] = mu * M[(i*col) + j];
		}
	}
}

void sumOfMatrix(double *Znum,double *temp2, double *temp3, double *temp4, int row, int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		Znum[(i*col) + j] = temp2[(i*col) + j] + temp3[(i*col) + j] + temp4[(i*col) + j];
		}
	}
}

void addScalarToDiagonal(double *Zden, double *BBt, double mu, int row, int col)
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

void AugmentIdentity(double *matrix, double *augmatrix, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
		augmatrix[(i * 2 * n) + j] = matrix[(i*n) + j];
		augmatrix[(((2 * i) + 1)*n) + j] = 0.0;
		}
	}
	for (int i = 0; i < n; i++)
	{
	augmatrix[(((2 * i) + 1)*n) + i] = 1.0;
	}
}

void cpuInverseOfMatrix(double *matrix, int col)
{
//#pragma omp parallel for 
	for (int m = 0; m < col; m++)
	{
		//Checking if diagonal element is 0
		if (matrix[((2 * col) + 1)*m] == 0)
		{
			//checking if the row is last row. If it is last row add the previous row to make it non zero
                	if (m == (col - 1))
			{
				for (int i = 0; i < (2 * col); i++)
				{					
				matrix[(m * (2 * col)) + i] = matrix[((m - 1) * (2 * col)) + i] + matrix[(m * (2 * col)) + i];
				}
			}
			else	//if it is not last row, add the next row.
			{
			        for (int i = 0; i < (2 * col); i++)
				{
				matrix[(m * 2 * col) + i] = matrix[((m + 1) * 2 * col) + i] + matrix[(m * 2 * col) + i];
				}
			}
		}
		//Make the diagonal elements 1 along with the whole row(divide).
		double initialValue = matrix[((2 * col) + 1)*m];
		for (int j = 0; j < (2 * col); j++)
		{
		matrix[(m * (2 * col)) + j] = matrix[(m * (2 * col)) + j] / initialValue;
		}

		//Making the elements of the row to zero
		for (int k = 0; k < col; k++)
		{
			float tempIni;
			tempIni = matrix[m + (k * (2 * col))];
			if (k == m)
			{
			//Just a loop to do nothing
			}
			else
			{
				for (int l = 0; l < (2 * col); l++)
				{
				
				double tempMul, tempDiv;
				tempMul = matrix[(m * (2 * col)) + l] * tempIni;
				tempDiv = tempMul / matrix[(m * (2 * col)) + m];
				matrix[(k * 2 * col) + l] = matrix[(k * (2 * col)) + l] - tempDiv;
				}
			}

		}
	}
}

void Inverse(double *augmatrix, double *matrixInverse, int n)
{
	for (int i = 0;i < n;i++)
	{
		for (int j = 0;j < n;j++)
		{
		matrixInverse[(i*n) + j] = augmatrix[(i*2*n)+n+j];
		}
        }
}

//Determinant
double Determinant(double *a,int n)
{
	cout << "time "<<endl;
	int i,j,j1,j2;
	double det = 0.0;

        for (j1=0;j1<n;j1++)
	{
		
	double *m = new double [(n-1)*(n-1)];		
		for (i=1;i<n;i++)
         	{
			j2 = 0;
			
			for (j=0;j<n;j++)
            		{
				if (j == j1)
				{
		        	continue;
				}
				else
				{
		       		m[((i-1)*n)+j2] = a[(i*n)+j];
		       		j2++;
				}
			}
		}
		
		det += pow(-1.0,j1+2.0) * a[j1] * Determinant(m,n-1);
	
	//free the pointer
	delete[] m;
	}

	return(det);
}

lapack_int matInv(double *A, int n)
{
	int ipiv[n+1];
	lapack_int ret;
	
	ret = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,n,n,A,n,ipiv);
	
	if(ret != 0)
	{
		return ret;
	}
	
	ret = LAPACKE_dgetri(LAPACK_ROW_MAJOR,n,A,n,ipiv);
	return ret;

}	
																					
void calculateZ(double *Z,double *BBt,double *xy, double *E, double *T, double *B_transpose, double mu, double *M, double *Y,const int row,const int col,const int row1)
{
	double *temp = new double [row*col];
	double *temp2 = new double [row*row1];
	double *temp3 = new double [row*row1]; 
	double *Znum = new double [row*row1];
	double *Zden = new double [row1*row1];
	double *Zdenaug = new double [row1*row1*row1*row1];
	double *ZdenInverse = new double [row1*row1];
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
	//displayValues(temp3, row*row1);

	//Znum = ((W-E-T*ones(1,p))*B'+mu*M+Y) 
	sumOfMatrix(Znum,temp2, temp3, Y, row, row1);
	//displayValues(Znum, row*row1);

	//denominator
	addScalarToDiagonal(Zden,BBt,mu,row1,row1);
	//displayValues(Zden, row1*row1);
	
	//t3 = high_resolution_clock::now();
	status = matInv(Zden,row1);
	//cout << "value of determinant "<< status << endl;	
	//t4 = high_resolution_clock::now();
	//duration<double> time_span1 = duration_cast<duration<double>>(t4 - t3);
	//cout << "Time in miliseconds for inverse section is : " << time_span1.count() * 1000 << " ms" << endl;
	
	//Inverse calculation via guass-jordon method
	////AugmentIdentity(Zden, Zdenaug, row1);
	////t1 = high_resolution_clock::now();	
	////cpuInverseOfMatrix(Zdenaug, row1);

	//displayValues(Zdenaug, row1*row1);
	////Inverse(Zdenaug,ZdenInverse,row1);
	//displayValues(ZdenInverse, row1*row1);
	////t2 = high_resolution_clock::now();
	//duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	//cout << "Time in miliseconds for first section is : " << time_span.count() * 1000 << " ms" << endl;
	
 	dump_to_file("Zden_old.txt",Zden,row1,row1);		
	////t3 = high_resolution_clock::now();
	//Z = ((W-E-T*ones(1,p))*B'+mu*M+Y)/(BBt+mu*eye(3*k))
        cpuMatrixMult(Znum, Zden, Z, row, row1, row1);	
       
        
	////cpuMatrixMult(Znum, ZdenInverse, Z, row, row1, row1);	
	//displayValues(Z,row*row1);
	
	//t4 = high_resolution_clock::now();
	//duration<double> time_span1 = duration_cast<duration<double>>(t4 - t3);
	//cout << "Time in miliseconds for inverse section is : " << time_span1.count() * 1000 << " ms" << endl;
	
	delete [] temp;	
	delete [] temp2;
	delete [] temp3;
	delete [] Znum;
	delete [] Zden;
	delete [] Zdenaug;
	delete [] ZdenInverse;

}

void calculateZ_preZden(double *Z,double *Zden,double *xy, double *E, double *T, double *B_transpose, double mu, double *M, double *Y,const int row,const int col,const int row1)
{
	//calculateZ_preZden(Z, Zden, xy, E, T, B_transpose,mu_orig,M,Y,row,col,row1);

        double *temp = new double [row*col];
        double *temp2 = new double [row*row1];
        double *temp3 = new double [row*row1];
        double *Znum = new double [row*row1];
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
        //displayValues(temp3, row*row1);

        //Znum = ((W-E-T*ones(1,p))*B'+mu*M+Y) 
        sumOfMatrix(Znum,temp2, temp3, Y, row, row1);
        //displayValues(Znum, row*row1);

	//Z = ((W-E-T*ones(1,p))*B'+mu*M+Y)/(BBt+mu*eye(3*k))
        cpuMatrixMult(Znum, Zden, Z, row, row1, row1);
        dump_to_file("Z_pre_den.txt",Zden,row1,row1);
        //displayValues(Zden, row1*row1);

	delete [] temp;
        delete [] temp2;
        delete [] temp3;
        delete [] Znum;

}

void differenceOfMatrix(double *diffMatrix, double *matrix1, double *matrix2, int row, int col)
{
        for (int i = 0;i < row;i++)
        {
                for (int j = 0;j < col;j++)
                {
                diffMatrix[(i*col) + j] = matrix1[(i*col) + j] - matrix2[(i*col) + j] ;
                }
        }
}

void calculateQ(double *Q, double *Z, double *Y,double mu, int row, int row1)
{
	double *temp = new double [row*row1];

	scalarToMatrixMultiply(temp, Y, 1/mu, row, row1);
	differenceOfMatrix(Q, Z, temp, row, row1);

	delete[] temp;

}

void prox_2norm(double *Q, double *M, double *C, double constant, int row, int col, int data_size)
{
	MKL_INT m = ROW, n = COL, lda = LDA, ldu = LDU, ldvt = LDVT, info;
	double superb[min(ROW,COL)-1];
	//float s[COL], u[LDU*ROW], vt[LDVT*COL];
	
	double *sigma = new double [COL];
	double *u = new double [LDU*ROW];
	double *vt = new double [LDVT*COL];
	double *Qtemp = new double [6];

	double *sigma1 = new double [ROW*ROW];
	double *vt1 = new double [ROW*COL];
	double *Qtemp1 = new double [4];
	double *Qtemp2 = new double [6];

//#pragma omp parallel for
	for(int i = 0;i < data_size;i++)
	{
		for(int j = 0;j<2;j++)
		{
			for(int k=0;k<3;k++)
			{
			Qtemp[(j * 3) + k] = Q[(3 * i) + (j*col) + k];
			}
		}
		info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, Qtemp, lda, sigma, u, ldu, vt, ldvt, superb);

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
	}

	delete[] Qtemp;
	delete[] sigma;
	delete[] u;
	delete[] vt;
	delete[] Qtemp1;
	delete[] Qtemp2;
	delete[] sigma1;
	delete[] vt1;

}

void updateDualvariable(double *Y,double mu,double *M,double *Z,int row,int row1)
{
	for(int i=0;i<row;i++)
	{
		for(int j = 0;j<row1;j++)
		{
			Y[(i*row1)+j] += mu*(M[(i*row1)+j] - Z[(i*row1)+j]); 
		}
	}
}

double febNorm(double *a, int row, int col)
{
	double norm = 0.0;
	double sum = 0.0;
	double *a_transpose = new double [col*row];
	double *ata = new double [col*col];

	TransposeOnCPU(a,a_transpose,row,col);
        cpuTransMatrixMult(a_transpose, a, ata, col, row);
	for(int i=0;i<col;i++)
	{
		for(int j=0;j<col;j++)
		{
			if(i==j)
			{
		  	sum += double((ata[(i*col)+j]));
			}
		}
	}
	norm=sqrt(double(sum));

	delete[] a_transpose;
	delete[] ata;
	return double(norm);
}

        //dump_to_file("a.txt",a,row,col);
        //dump_to_file("atranspose.txt",a_transpose,col,row);
	
        //dump_to_file("ata.txt",ata,col,col);

void resCalc(double *PrimRes, double *DualRes, double *M, double *Z, double *ZO,double mu, int row, int row1)
{
	double *MminusZ = new double [row*row1];
	double *ZminusZO = new double [row*row1];

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
	const int row = 2;
	const int col = 15;
	const int row1 = 384;
	const int col1 = 15;
	double tol = 1e-10;

	double *xy = new double [row*col];
	double *mean = new double [row];
	double *B = new double [row1*col1];
	double *B_transpose = new double [col1*row1];
	double *B_mean = new double [row1];
	double *BBt = new double [row1*row1];
	int items = 0;
	double a = 0.0;
	int B_items = 0;
	int lam =1;
	bool verb = true;
	high_resolution_clock::time_point t1,t2,t3,t4;
	const int data_size = row1/3;	
	
	//ssr2D3D_alm
	//M => (2*384) = 0,  C ==> (1*384) = 0, E ==> (2*15) = 0, T ==> mean(W,2)
	double *M = new double [row*row1];
	double *C = new double [data_size];
	double *E = new double [row*col];
	double *T = new double [row];
	
	// auxiliary variables for ADMM
	double *Z = new double [row*row1];
	double *Y = new double [row*row1];
	double *ZO = new double [row*row1];
	double *Q = new double [row*row1];
	double mu = 0.0;
	double PrimRes;
	double DualRes;

	//allocate precomputed Zden
	double *Zden = new double [row1*row1];
	double *Zden1 = new double [row1*row1];
	double *Zden2 = new double [row1*row1];
	double *Zden3 = new double [row1*row1];
	double *Zden4 = new double [row1*row1];
        int status = 0;
	double mu_orig = 0.0;	
	double mu1 = 0.0;
	double mu2 = 0.0;
	double mu3 = 0.0;
	double mu4 = 0.0;
	
	//t3 = high_resolution_clock::now();
	t1 = high_resolution_clock::now();
	
	items = readValues("messi2.txt",xy,items);
	rowMean(xy, col, row, mean);
        Scalc(xy, col, row, mean);
        rowMean(xy, col, row, mean);
	a = mean_of_std_deviation(xy,col,row,mean);
	newScalc(xy,col,row,a);
	//displayValues(xy,items);
	B_items = readValues("exp1.txt", B, B_items);
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
	mu2 = mu1 * 2;
	mu3 = mu / 2;
	mu4 = mu3 / 2;

	addScalarToDiagonal(Zden,BBt,mu_orig,row1,row1);
	addScalarToDiagonal(Zden1,BBt,mu1,row1,row1);
	addScalarToDiagonal(Zden2,BBt,mu2,row1,row1);
	addScalarToDiagonal(Zden3,BBt,mu3,row1,row1);
	addScalarToDiagonal(Zden4,BBt,mu4,row1,row1);
 	
	//dump_to_file("Z_old.txt",Z,row1,row1);				

	status = matInv(Zden,row1);
	status = matInv(Zden1,row1);
	status = matInv(Zden2,row1);
	status = matInv(Zden3,row1);
	status = matInv(Zden4,row1);	
	
	//t4 = high_resolution_clock::now();
	//duration<double> time_span = duration_cast<duration<double>>(t4 - t3);
	//cout << "Time in miliseconds for first section is : " << time_span.count() * 1000 << " ms" << endl;
	

	for(int iter = 0; iter < 1; iter++)
	{
		//t1 = high_resolution_clock::now();
		initialize(ZO,Z,row1,row);
		//displayValues(Z,row1*row);
//		calculateZ(Z, BBt,xy, E, T, B_transpose,mu,M,Y,row,col,row1);
			
		//cout << "value of mu_orig "<<mu_orig<<endl;
		//cout << "value of mu1 "<<mu1<<endl;
		//cout << "value of mu2 "<<mu2<<endl;	

		//pre_computed part
		if(mu = mu_orig)
		{
			//cout << "coming inside 1"<<endl;
			calculateZ_preZden(Z, Zden,xy, E, T, B_transpose,mu_orig,M,Y,row,col,row1);
			//calculateZ(Z, BBt,xy, E, T, B_transpose,mu,M,Y,row,col,row1);
		}
		else if(mu = mu1)
		{
			//cout << "coming inside 2"<<endl;
			calculateZ_preZden(Z, Zden1,xy, E, T, B_transpose,mu1,M,Y,row,col,row1);
		}
		else if(mu = mu2)
		{
			//cout << "coming inside 3"<<endl;
			calculateZ_preZden(Z, Zden2,xy, E, T, B_transpose,mu2,M,Y,row,col,row1);	
		}
		else if(mu = mu3)
		{
			//cout << "coming inside 4"<<endl;
			calculateZ_preZden(Z, Zden3,xy, E, T, B_transpose,mu3,M,Y,row,col,row1);
		}
		else if(mu = mu4)
		{
			//cout << "coming inside 5"<<endl;
			calculateZ_preZden(Z, Zden4,xy, E, T, B_transpose,mu4,M,Y,row,col,row1);
		}
		else
		{
			//cout << "coming inside 6"<<endl;
			calculateZ(Z, BBt,xy, E, T, B_transpose,mu,M,Y,row,col,row1);
		}

	//	t1 = high_resolution_clock::now();
		calculateQ(Q,Z,Y,mu,row,row1);
	//	t2 = high_resolution_clock::now();
		//displayValues(Z,row*row1);

		//t1 = high_resolution_clock::now();
		prox_2norm(Q,M,C,lam/mu,row,row1,data_size);
		//t2 = high_resolution_clock::now();
		
		updateDualvariable(Y,mu,M,Z,row,row1);
		resCalc(&PrimRes,&DualRes,M,Z,ZO,mu,row,row1);
		//displayValues(M,row*row1);
		
		//if ((verb == true) && ((iter%10) == 0))
		//{
			cout << "Iter "<< iter+1 <<": PrimRes = "<<PrimRes <<", DualRes = "<<DualRes<<", mu = "<< mu <<endl; 
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
		//duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		//cout << "Time in miliseconds: " << time_span.count() * 1000 << " ms" << endl;

	}
	t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "Time in miliseconds: " << time_span.count() * 1000 << " ms" << endl;
	

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

}

