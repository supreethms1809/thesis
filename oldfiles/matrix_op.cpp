
//Matrix Transpose
void Transpose(double *matrix, double *matrixTranspose, int ny, int nx)
{
	
	for (int y = 0; y < ny; y++)
	{
		for (int x = 0; x < nx; x++)
		{
			matrixTranspose[x*ny + y] = matrix[y*nx + x];
		}
	}
	
}

//Matrix Multiplication
void MatrixMult(double *A, double *B, double *C,  int ny, int nx)
{
	double fSum;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			fSum = 0.0f;
			for (int k = 0; k < nx; k++)
			{
				fSum += (A[(i*nx)+k]*B[(k*nx)+j]);
			}
			C[(i*nx) + j] = fSum;
		}
	}
}

//Matrix Inverse
//Det
double Determinant(double *a,int n)
{
	int i,j,j1,j2;
	double det = 0;


	      det = 0;
        for (j1=0;j1<n;j1++)
	{
		
	double *m = new double [(n-1)*(n-1)];		
		for (i=1;i<n;i++)
         	{
			j2 = 0;
			
			for (j=0;j<n;j++)
            		{
				if (j == j1)
		        	continue;
		       		m[((i-1)*n)+j2] = a[(i*n)+j];
		       		j2++;
			}
		}
		
		det += pow(-1.0,j1+2.0) * a[j1] * Determinant(m,n-1);
	
	//free the pointer
	delete[] m;
	}

	return(det);
}

/*
   Find the cofactor matrix of a square matrix
*/
void CoFactor(double **a,int n,double **b)
{
   int i,j,ii,jj,i1,j1;
   double det;
   double **c;

   c = (double **)malloc((n-1)*sizeof(double *));
   for (i=0;i<n-1;i++)
     c[i] =( double *)malloc((n-1)*sizeof(double));

   for (j=0;j<n;j++) {
      for (i=0;i<n;i++) {

         /* Form the adjoint a_ij */
         i1 = 0;
         for (ii=0;ii<n;ii++) {
            if (ii == i)
               continue;
            j1 = 0;
            for (jj=0;jj<n;jj++) {
               if (jj == j)
                  continue;
               c[i1][j1] = a[ii][jj];
               j1++;
            }
            i1++;
         }

         /* Calculate the determinate */
         det = Determinant(c,n-1);

         /* Fill in the elements of the cofactor */
         b[i][j] = pow(-1.0,i+j+2.0) * det;
      }
   }
   for (i=0;i<n-1;i++)
      free(c[i]);
   free(c);
}

void MatrixInverse(double *A,double *InvA, int dim)//inverse of a square matrix
{
	double det;
	//double Determinant(double **a,int n)
	det=Determinant(A,dim);

	if(det<=0.0) 
	{
		cout << "Matrix is Singular " << endl;
	}
	else
	{
		//calculate inverse of the matrix
		det=1/det;
		double **adjoint;
		double **cofactor;
		adjoint=(double **)malloc(sizeof(double *)*dim);
		cofactor=(double **)malloc(sizeof(double *)*dim);
		for(int i=0;i<dim;i++)	
		{
			adjoint[i]=(double *)malloc(sizeof(double)*dim);
			cofactor[i]=(double *)malloc(sizeof(double)*dim);
		}
		//void CoFactor(double **a,int n,double **b)
		CoFactor(A,dim,cofactor);
		Transpose(cofactor,adjoint, dim, dim);

		for (int i=0;i<dim;i++)
		{
			for (int j=0;j<dim;j++)
			{
				InvA[i][j]=det*adjoint[i][j];
			}
		}

		FreeMatrix(adjoint,dim,dim);
		FreeMatrix(cofactor,dim,dim);
	
	}	


}



