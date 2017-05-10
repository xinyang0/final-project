#include <fftw3.h>
#include "math.h"
#include "util.h"

int main()
{
fftw_complex *in, *out;
fftw_plan p;
int N=256;
int i;

in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
/* data input */
for(i=0; i<N; i++){
   in[i][0]=cos(1.0*i);
   in[i][1]=sin(2.0*i);
   }
out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

timestamp_type time1, time2;
get_timestamp(&time1);

p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

fftw_execute(p); /* repeat as needed */

fftw_destroy_plan(p);
get_timestamp(&time2);
double elapsed = timestamp_diff_in_seconds(time1,time2);
printf("Time elapsed is %f seconds.\n", elapsed);
/*for(i=0; i<N; i++){
 	printf("out = %f + i %f ", out[i][0],out[i][1]);
	}*/

fftw_free(in); fftw_free(out);
}