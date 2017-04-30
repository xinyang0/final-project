// MPI transpose N*N square matrix. N is divisible by the number of processor.
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char * argv[])
{
int i, j, k, N, m, rank, size, root;
double temp;

sscanf(argv[1], "%d", &N);

MPI_Init( &argc, &argv );
MPI_Comm_rank( MPI_COMM_WORLD, &rank );
MPI_Comm_size( MPI_COMM_WORLD, &size );   // number of processor 

/* no print here? */
if ((N%size)!=0) { 
	printf("Error!:matrix size N must be divisible by # of processors.");
	printf("Programm aborting....");
	MPI_Abort(MPI_COMM_WORLD, 1);
}


/* Initialize the matrix and print it */

double *matrix, *lmatrix;
if (rank == 0) {                                 
	matrix = malloc(N*N*sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			matrix[i*N+j] = i+2*j;
		}
	}
    for (i = 0; i < N; i++) {
    	for(j = 0; j < N; j++) {
        	printf("%f  ", matrix[i*N+j]);
        }
      	printf("\n");
    }
   printf("\n");
}


/* Create local array. */

m = N/size;  // the number of rows in each processor
lmatrix = malloc(m*N*sizeof(double));


/* Allocate m rows of the global matrix to each processor */

root = 0;
MPI_Scatter(matrix, m*N, MPI_DOUBLE, lmatrix, m*N, MPI_DOUBLE, root, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);

/*for (int p=0; p<size; p++) {
if (rank == p) {
	printf("Local process on rank %d is:\n", rank);
	for (i = 0; i < m; i++) {
      	for(j = 0; j < N; j++) {
           	printf("%f  ", lmatrix[i*N+j]);
         }  		
      printf("\n");
    }
    printf("\n");
}
} */


/* Use alltoall to communicate the size blocks in each processor  */

double *newlmatrix =  malloc(m*N*sizeof(double));

for ( i = 0; i < m; i++) {
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Alltoall(&(lmatrix[i*N]), m, MPI_DOUBLE, &(newlmatrix[i*N]), m, MPI_DOUBLE, MPI_COMM_WORLD);
}


/* Transpose each m*m block */

MPI_Barrier(MPI_COMM_WORLD);
for (k = 0; k < size; k++) {       // Loop all the blocks in each processor. block j  
	for ( i = 0; i < m; i++) {    // row i
		for (j = i+1; j < m; j++) {    // column j in each block ;
			temp = newlmatrix[i*N+k*m+j];
			newlmatrix[i*N+k*m+j] =  newlmatrix[j*N+k*m+i];
			newlmatrix[j*N+k*m+i] = temp;
		}
	}
}

for (int p=0; p<size; p++) {
if (rank == p) {
	printf("Local process on rank %d is:\n", rank);
	for (i = 0; i < m; i++) {
      	for(j = 0; j < N; j++) {
           	printf("%f  ", newlmatrix[i*N+j]);
         }  		
      printf("\n");
    }
    printf("\n");
}
}


free(lmatrix);
free(newlmatrix);

if (rank == 0) free(matrix);
MPI_Finalize();
return 0;
}

