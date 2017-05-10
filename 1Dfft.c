#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include <string.h>


#define PI 3.1415926535897932384626


void InitTDAndFD(double *TD_real,double *TD_imag,double *FD_real,double *FD_imag,unsigned char *data,int lWidth,int lHeight)
{
    int w=1;
    int h=1;
    int wp=0;
    int hp=0;
    double *TmpTD_real, *TmpTD_imag;
    double *TmpFD_real, *TmpFD_imag;
    
    while(w<lWidth)
    {
        w=w*2;
        wp++;
    }
    while(h<lHeight)
    {
        h=h*2;
        hp++;
    }
    
    TmpTD_real = calloc(w*h, sizeof(double));
    TmpTD_imag = calloc(w*h, sizeof(double));
    TmpFD_real = calloc(w*h, sizeof(double));
    TmpFD_imag = calloc(w*h, sizeof(double));
    
    for(int i = 0; i < h; i++)
    {
        if(i < lHeight)
        {
            for(int j = 0; j < w; j++)
            {
                if(j < lWidth)
                {
                    TmpTD_real[i*w+j]=data[i*lWidth+j];
                    TmpTD_imag[i*w+j]=0.0;
                }
                else
                {
                    TmpTD_real[i*w+j]=0.0;
                    TmpTD_imag[i*w+j]=0.0;
                }
                
            }
        }
        else
        {
            for(int j=0;j<w;j++)
            {
                TmpTD_real[i*w+j]=0.0;
                TmpTD_imag[i*w+j]=0.0;
                
            }
        }
    }
    
    for(int i = 0;i < w * h; i++)
    {
        TmpFD_real[i]=0.0;
        TmpFD_imag[i]=0.0;
    }
    
    TD_real=TmpTD_real;
    TD_imag=TmpTD_imag;
    FD_real=TmpFD_real;
    FD_imag=TmpFD_imag;
    
}

void FFT_1D(double *TD_real, double *TD_imag, double *FD_real, double *FD_imag, int Len)
{
    
    int l=1;
    int r=0;
    int p=0;
    double angle=0;
    double *W_real, *W_imag, *X1_real, *X1_imag, *X2_real, *X2_imag, *X_real, *X_imag;
    
    while(l<Len)
    {
        l=l*2;
        r++;
    }
    
    W_real = calloc(l/2, sizeof(double));
    W_imag = calloc(l/2, sizeof(double));
    X1_real = calloc(l, sizeof(double));
    X1_imag = calloc(l, sizeof(double));
    X2_real = calloc(l, sizeof(double));
    X2_imag = calloc(l, sizeof(double));
    
    for(int i=0;i<l/2;i++)
    {
        angle=-i*PI*2/l;
        W_real[i] = cos(angle);
        W_imag[i] = sin(angle);
    }
    
    memcpy(X1_real,TD_real,sizeof(double)*l);
    memcpy(X1_imag,TD_imag,sizeof(double)*l);
    
    for(int k=0;k<r;k++)
    {
        for(int j=0;j<pow(2,k);j++)
        {
            for(int i=0;i<pow(2,(r-k-1));i++)
            {
                p = j * pow(2,(r-k));
                X2_real[i+p] = X1_real[i+p] + X1_real[i+p+(int)(pow(2,(r-k-1)))];
                X2_imag[i+p] = X1_imag[i+p] + X1_imag[i+p+(int)(pow(2,(r-k-1)))];
                X2_real[i+p+(int)(pow(2,(r-k-1)))]=(X1_real[i+p]-X1_real[i+p+(int)(pow(2,(r-k-1)))]) * W_real[i*(int)(pow(2,k))]-(X1_imag[i+p]-X1_imag[i+p+(int)(pow(2,(r-k-1)))]) * W_imag[i*(int)(pow(2,k))];
                X2_imag[i+p+(int)(pow(2,(r-k-1)))]=(X1_real[i+p]-X1_real[i+p+(int)(pow(2,(r-k-1)))]) * W_imag[i*(int)(pow(2,k))]+(X1_imag[i+p]-X1_imag[i+p+(int)(pow(2,(r-k-1)))]) * W_real[i*(int)(pow(2,k))];
            }
        }
        X_real=X1_real;
        X_imag=X1_imag;
        X1_real=X2_real;
        X1_imag=X2_imag;
        X2_real=X_real;
        X2_imag=X_imag;
    }
    
    
    for(int j=0;j<l;j++)
    {
        p=0;
        for(int i=0;i<r;i++)
        {
            if(j&(int)(pow(2,i)))
            {
                p+=pow(2,(r-i-1));
            }
        }
        FD_real[j]=X1_real[p];
        FD_imag[j]=X1_imag[p];
    }
    
    free(W_real);
    free(W_imag);
    free(X1_real);
    free(X1_imag);
    free(X2_real);
    free(X2_imag);
    
}
