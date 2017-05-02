#include <fftw3.h>
int main()
{
fftw_complex *matrix, *matrix_out;
fftw_plan p;
int N=4;
int i, j;

matrix = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N*N));
for (i = 0; i < N; i++) {
	for (j = 0; j < N; j++) {
		matrix[i*N+j][0] = i + 1.0 * j;    // real part
		matrix[i*N+j][1] = i + 2.0 * j;    // imaginary part
	}
}
matrix_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N*N));
p = fftw_plan_dft_2d(N, N, matrix, matrix_out, FFTW_FORWARD, FFTW_ESTIMATE);

fftw_execute(p); /* repeat as needed */

for (i = 0; i < N; i++) {
	for (j = 0; j < N; j++) {
 	printf("%f + i %f ", matrix_out[i*N+j][0],matrix_out[i*N+j][1]);
	}
	printf("\n");
}
fftw_destroy_plan(p);
fftw_free(matrix); fftw_free(matrix_out);
}