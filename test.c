#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include "neuron.h"
#include "randtool.h"
// test simple neuron
void test1()
{
	double t=1;

	NEURON *N;
	N = NEURON_New(1,0);
	double a[LEN] = {1,2,3};
	double *x[LEN];
	double w[LEN] = {1,1,1};
	double b = 1;
	x[0]=&a[0];
	x[1]=&a[1];
	x[2]=&a[2];

	NEURON_Init(N,x,w,3,b,sigmoid);
	NEURON_Calc(N);
	printf("%lf %lf\n",N->z,N->a);
	return;
}

// test uniform random generator
void test2()
{
	double a[10000];
	int i,j,n;
	double EX=0;
	n=8000;
	for (i=0;i<n;i++)
	{
		a[i]=rand_uniform(-1000,1000);
		EX+=a[i];
	}
	EX/=n;
	printf("EX=%lf\n",EX);
	int s[10];
	memset(s,0,sizeof(s));
	for (i=0;i<n;i++)
		s[(int)abs(a[i]/50)]++;
	for (i=0;i<10;i++)
		printf("s[%d]=%d ",i,s[i]);
	printf("\n");
	return;
}
// test layer
void test3()
{
    NEURON_LAYER *L0,*L1,*L2,*L3,*LA[10];
    double Input[LEN];
    int num[]={3,5,3},num0,num_layer;
    int i,j,k;

    num_layer=sizeof(num)/sizeof(int);
    // input layer L0
    num0=num[0];
    for (i=0;i<num0;i++)
        Input[i]=rand_uniform(-5,10);
    i=0;
    LA[i] = LAYER_New(i,num[i]);
    LAYER_Init(LA[i],NULL,num[i],NULL,NULL,Input);
    // layer Li
    for (i=1;i<num_layer;i++)
    {
        LA[i]=LAYER_New(i,num[i]);
        LAYER_Init(LA[i],LA[i-1],num[i],NULL,tanh,NULL);
        LAYER_Connect(LA[i-1],LA[i]);
    }
    //------finish: initialization------


    Forward(LA[1]);
    ShowLayer(LA[0]);
    return;
}

typedef void (*PF)(void);

void TEST()
{
    PF T[]={
        test1,
        test2,
        test3,
    };
    int i,n=sizeof(T)/sizeof(PF);
    for (i=0;i<n;i++)
        T[i]();
    return;
}