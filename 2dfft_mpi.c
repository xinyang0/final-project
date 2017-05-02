
#include <stdio.h>
#include <stdlib.h>
//#include <complex.h> 
#include <fftw3.h>
#include "mpi.h"

int main(int argc, char * argv[])
{
int i, j, k, N, m, rank, size, root;
//complex double temp;
double temp;

sscanf(argv[1], "%d", &N);

MPI_Init( &argc, &argv );
MPI_Comm_rank( MPI_COMM_WORLD, &rank );
MPI_Comm_size( MPI_COMM_WORLD, &size );   // number of processor 

/* no print here? */
if ((N%size)!=0 && rank == 0) { 
	printf("Error!:matrix size N must be divisible by # of processors.");
	printf("Programm aborting....");
	MPI_Abort(MPI_COMM_WORLD,0);
}


/* Initialize the matrix and print it */

//double *matrix, *lmatrix;
fftw_complex *matrix, *lmatrix, *lmatrix_out, *newlmatrix, *newlmatrix_out; 
//complex double *matrix, *lmatrix, *lmatrix_out, *newlmatrix, *newlmatrix_out;     
/* The data is an array of type fftw_complex, which is by default a double[2] composed of
the real (in[i][0]) and imaginary (in[i][1]) parts of a complex number. */
fftw_plan p;

double *matrix_real, *matrix_imag, *lmatrix_real, *lmatrix_imag, *newlmatrix_real, *newlmatrix_imag;

if (rank == 0) {                                 
	matrix = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N*N));
	matrix_real = calloc(N*N, sizeof(double));
	matrix_imag = calloc(N*N, sizeof(double));
//	matrix = (complex double*)malloc((N*N) * sizeof(complex double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			matrix[i*N+j][0] = i + 1.0 * j;    // real part
			matrix_real[i*N+j] = i + 1.0 * j; 
			matrix[i*N+j][1] = i + 2.0 * j;    // imaginary part
			matrix_imag[i*N+j] = i + 2.0 * j; 
//			matrix[i*N+j] = (i + 1.0 * j) + (i + 2.0 * j)*I; ; 
		}
	}
    for (i = 0; i < N; i++) {
    	for(j = 0; j < N; j++) {
			printf("%f + i %f ", matrix[i*N+j][0], matrix[i*N+j][1]);
//			printf("%f + i%f\n", creal(matrix[i*N+j]), cimag(matrix[i*N+j]));
        }
      	printf("\n");
    }
   printf("\n");
}

/* Create local array. */

m = N/size;  // the number of rows in each processor
lmatrix = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (m*N));
lmatrix_real = calloc(m*N, sizeof(double));
lmatrix_imag = calloc(m*N, sizeof(double));
//lmatrix = (complex double*)malloc((m*N) * sizeof(complex double));

/* Allocate m rows of the global matrix to each processor */

root = 0;
//MPI_Scatter(&(matrix), m*N, MPI_DOUBLE_COMPLEX, &(lmatrix), m*N, MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);
MPI_Scatter(matrix_real, m*N, MPI_DOUBLE, lmatrix_real, m*N, MPI_DOUBLE, root, MPI_COMM_WORLD);
//MPI_Barrier(MPI_COMM_WORLD);
MPI_Scatter(matrix_imag, m*N, MPI_DOUBLE, lmatrix_imag, m*N, MPI_DOUBLE, root, MPI_COMM_WORLD);
//MPI_Barrier(MPI_COMM_WORLD);

for (i = 0; i < m; i++) {
	for(j = 0; j < N; j++) {
      	lmatrix[i*N+j][0] = lmatrix_real[i*N+j];
      	lmatrix[i*N+j][1] = lmatrix_imag[i*N+j];
	}
}
    
/*for (int p=0; p<size; p++) {
if (rank == p) {
	printf("Local process on rank %d is:\n", rank);
	for (i = 0; i < m; i++) {
      	for(j = 0; j < N; j++) {
           	printf("%f + i %f ", lmatrix[i*N+j][0], lmatrix[i*N+j][1]);
 //			printf("%f + i%f\n", creal(lmatrix[i*N+j]), cimag(lmatrix[i*N+j]));
         }  		
      printf("\n");
    }
    printf("\n");
}
} */


/* Use FFTW do fft for each row locally */
lmatrix_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (m*N));
//lmatrix_out = (complex double*)malloc((m*N) * sizeof(complex double));

for (i = 0; i < m; i++) {
	p = fftw_plan_dft_1d(N, &(lmatrix[i*N]), &(lmatrix_out[i*N]), FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
}

for (i = 0; i < m; i++) {
	for(j = 0; j < N; j++) {
      	lmatrix_real[i*N+j] = lmatrix_out[i*N+j][0];
      	lmatrix_imag[i*N+j] = lmatrix_out[i*N+j][1];
	}
}

/*for (int p=0; p<size; p++) {
if (rank == p) {
	printf("Local process on rank %d is:\n", rank);
	for (i = 0; i < m; i++) {
      	for(j = 0; j < N; j++) {
           	printf("%f + i %f ", lmatrix_out[i*N+j][0], lmatrix_out[i*N+j][1]);
 //			printf("%f + i %f ", lmatrix_real[i*N+j], lmatrix_imag[i*N+j]);
 //			printf("%f + i%f\n", creal(lmatrix[i*N+j]), cimag(lmatrix[i*N+j]));
         }  		
      printf("\n");
    }
    printf("\n");
}
}*/


/* Use alltoall to communicate the size blocks in each processor  */

newlmatrix = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (m*N));
//newlmatrix = (complex double*)malloc((m*N) * sizeof(complex double));
newlmatrix_real = calloc(m*N, sizeof(double));
newlmatrix_imag = calloc(m*N, sizeof(double));

for ( i = 0; i < m; i++) {
	MPI_Alltoall(&(lmatrix_real[i*N]), m, MPI_DOUBLE, &(newlmatrix_real[i*N]), m, MPI_DOUBLE, MPI_COMM_WORLD);
//	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Alltoall(&(lmatrix_imag[i*N]), m, MPI_DOUBLE, &(newlmatrix_imag[i*N]), m, MPI_DOUBLE, MPI_COMM_WORLD);
//	MPI_Barrier(MPI_COMM_WORLD);
//    MPI_Alltoall(&(lmatrix_out[i*N]), m, MPI_DOUBLE_COMPLEX, &(newlmatrix[i*N]), m, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
}

for (i = 0; i < m; i++) {
	for(j = 0; j < N; j++) {
      	newlmatrix[i*N+j][0] = newlmatrix_real[i*N+j];
      	newlmatrix[i*N+j][1] = newlmatrix_imag[i*N+j];
	}
}

/*for (int p=0; p<size; p++) {
if (rank == p) {
	printf("Local process on rank %d is:\n", rank);
	for (i = 0; i < m; i++) {
      	for(j = 0; j < N; j++) {
           	printf("%f + i %f ", newlmatrix[i*N+j][0], newlmatrix[i*N+j][1]);
//           	printf("%f + i %f ", newlmatrix_real[i*N+j], newlmatrix_imag[i*N+j]);
         }  		
      printf("\n");
    }
    printf("\n");
}
}*/

/* Transpose each m*m block */

//MPI_Barrier(MPI_COMM_WORLD);
for (k = 0; k < size; k++) {       // Loop all the blocks in each processor. block j  
	for ( i = 0; i < m; i++) {    // row i
		for (j = i+1; j < m; j++) {    // column j in each block ;
			temp = newlmatrix[i*N+k*m+j][0];
			newlmatrix[i*N+k*m+j][0] =  newlmatrix[j*N+k*m+i][0];
			newlmatrix[j*N+k*m+i][0] = temp;
     		temp = newlmatrix[i*N+k*m+j][1];
			newlmatrix[i*N+k*m+j][1] =  newlmatrix[j*N+k*m+i][1];
			newlmatrix[j*N+k*m+i][1] = temp; 
		}
	}
}


/* Do fft for each row locally again */
newlmatrix_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (m*N));
//newlmatrix_out = (complex double*)malloc((m*N) * sizeof(complex double));

for (i = 0; i < m; i++) {
	p = fftw_plan_dft_1d(N, &(newlmatrix[i*N]), &(newlmatrix_out[i*N]), FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
}

/*MPI_Barrier(MPI_COMM_WORLD);
for (int p=0; p<size; p++) {
	if (rank == p) {
		printf("Local process on rank %d is:\n", rank); */    /* little bug in printf: not in order */
/*			for (i = 0; i < m; i++) {
      			for(j = 0; j < N; j++) {
					printf(" %f + i %f ", newlmatrix_out[i*N+j][0], newlmatrix_out[i*N+j][1]);
         		}  		
      			printf("\n");
    		}
    	printf("\n");
	}
} */


/* Transform the newlmatrix_out back */

for (i = 0; i < m; i++) {
	for(j = 0; j < N; j++) {
      	lmatrix_real[i*N+j] = newlmatrix_out[i*N+j][0];
      	lmatrix_imag[i*N+j] = newlmatrix_out[i*N+j][1];
	}
}

for ( i = 0; i < m; i++) {
	MPI_Alltoall(&(lmatrix_real[i*N]), m, MPI_DOUBLE, &(newlmatrix_real[i*N]), m, MPI_DOUBLE, MPI_COMM_WORLD);
//	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Alltoall(&(lmatrix_imag[i*N]), m, MPI_DOUBLE, &(newlmatrix_imag[i*N]), m, MPI_DOUBLE, MPI_COMM_WORLD);
//	MPI_Barrier(MPI_COMM_WORLD);
}

for (i = 0; i < m; i++) {
	for(j = 0; j < N; j++) {
      	newlmatrix[i*N+j][0] = newlmatrix_real[i*N+j];
      	newlmatrix[i*N+j][1] = newlmatrix_imag[i*N+j];
	}
}

for (k = 0; k < size; k++) {       // Loop all the blocks in each processor. block j  
	for ( i = 0; i < m; i++) {    // row i
		for (j = i+1; j < m; j++) {    // column j in each block ;
			temp = newlmatrix[i*N+k*m+j][0];
			newlmatrix[i*N+k*m+j][0] =  newlmatrix[j*N+k*m+i][0];
			newlmatrix[j*N+k*m+i][0] = temp;
     		temp = newlmatrix[i*N+k*m+j][1];
			newlmatrix[i*N+k*m+j][1] =  newlmatrix[j*N+k*m+i][1];
			newlmatrix[j*N+k*m+i][1] = temp; 
		}
	}
}

MPI_Barrier(MPI_COMM_WORLD);
for (int p=0; p<size; p++) {
	if (rank == p) {
		printf("Local process on rank %d is:\n", rank);     /* little bug in printf: not in order */
			for (i = 0; i < m; i++) {
      			for(j = 0; j < N; j++) {
					printf(" %f + i %f ", newlmatrix[i*N+j][0], newlmatrix[i*N+j][1]);
         		}  		
      			printf("\n");
    		}
    	printf("\n");
	}
}


fftw_free(lmatrix);
fftw_free(newlmatrix);
fftw_free(lmatrix_out);
fftw_free(newlmatrix_out);
if (rank == 0) { 
	fftw_free(matrix);
	free(matrix_real);
	free(matrix_imag);
	}
free(lmatrix_real);
free(lmatrix_imag);
free(newlmatrix_real);
free(newlmatrix_imag);
MPI_Finalize();
return 0;
}

