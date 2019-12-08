#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "neuron.h"
#include "randtool.h"

//---------------Simple Function-------------
double equal(double x) { return x;}
double sigmoid(double z)
{
	return 1/(1+exp(-z));
}

double tanh(double x)
{
	double e1 = exp(x);
	double e2 = exp(-x);
	return (e1-e2)/(e1+e2);
}

// tanh的导数
double tanh_p(double x)
{
	double t = tanh(x);
	return 1 - t*t;
}

static double Xavier_Generator(int n1, int n2)
{
	double left,right;
	right = sqrt(6)/sqrt(n1+n2);
	left=-right;
	return rand_uniform(left,right);
}

//------------Neuron Initalization-------------

NEURON *NEURON_New(int level,int index)
{
	NEURON *p;
	p = (NEURON*)malloc(sizeof(NEURON));
	if (!p) return NULL;
	memset(p,0,sizeof(p));
	p->level=level;
	p->index=index;

	return p;
}

void NEURON_Del(NEURON* p)
{
	if (!p) return;
	free(p);

	return;
}

void NEURON_Init(NEURON *N,
				double *x[LEN],
				double w[LEN],
				double num, // num of weights
				double b,
				ACT_PF act_pf)
{
	int i;
	N->f = act_pf;
	N->num=num;
	for (i=0;i<num;i++)
	{
		N->x[i+1]=x[i];
		N->w[i+1]=w[i];
	}
	N->w[0]=b;
	// assume *N->x[0]=1; but you should notice that x[0] is a wild pointer now.
	N->z=0;
	N->a=0;

	return;
}

void NEURON_Calc(NEURON *N)
{
	int i;
	double sum = N->w[0];
	if (N->level==0) return;
	for (i=0;i<N->num;i++)
		sum+=N->w[i+1] * (*N->x[i+1]);
	N->z=sum;
	N->a=N->f(sum);

	return;
}

NEURON_LAYER *LAYER_New(int level,int num)
{
	NEURON_LAYER *p;
	int i;
	p=(NEURON_LAYER*)malloc(sizeof(NEURON_LAYER));
	if (!p) return NULL;
	p->num=num;
	p->level=level;
	p->prev_layer=NULL;
	p->next_layer=NULL;
	for (i=0;i<num;i++)
		p->units[i]=NEURON_New(level,i);

	return p;
}

void LAYER_Del(NEURON_LAYER *p)
{
	int i;
	if (!p) return;
	for (i=0;i<p->num;i++)
		NEURON_Del(p->units[i]);
	free(p);

	return;
}
void LAYER_Connect(NEURON_LAYER *Curr, NEURON_LAYER *Next)
{
	if (!Curr || !Next) return;
	Curr->next_layer = Next;
	return;
}
void LAYER_Init(NEURON_LAYER *L,NEURON_LAYER *Pre, 
                int neuron_num, 
				double (*para_gen)(int p1,int p2), 
				ACT_PF act_pf, 
				double data[])
{
	int i,j;
	double w[LEN];
	double *x[LEN];
	double bias;
	if (!L) return;
	L->prev_layer=Pre;
	// use LAYER_Connect to define next layer
	L->next_layer=NULL;
	// default activication function is equal(no function)
	if (act_pf == NULL) act_pf=equal;	
	// the input level has no parameters
	if (L->level != 0)
	{
		// default para_gen is Xavier_Generator
		if (para_gen==NULL) para_gen=Xavier_Generator;

		// Get input from previous layer's output
		for (i=0;i<Pre->num;i++)
			x[i]=&Pre->units[i]->a;
		// randomly normalization the initial parameters
		for (j=0;j<L->num;j++)
		{
			// Notice: we are 0,1,2 ; but we need 1 2 3 
			for (i=0;i<Pre->num;i++)
				w[i]=para_gen(i+1,j+1);
			bias=para_gen(i+1,j+1);
			NEURON_Init(L->units[j],x,w,Pre->num,bias,act_pf);
		}
	}
	else if (data!=NULL)
	{
		// TODO: deal with level 0
		// For Level 0 (input) , just need z.
		// inital neuron manually
		for (i=0;i<L->num;i++)
		{
			L->units[i]->z=data[i];
			L->units[i]->a=act_pf(data[i]);
			L->units[i]->num=0;
			L->units[i]->f=equal;
		}
	}
	return;
}
// void LAYER_FullConnectInput(NEURON_LAYER *L,double x[],int num)
// {
// 	int i,j;
// 	for (i=0;i<num;i++)
// 		for (j=0;j<L->units[i]->num;j++)
// 			L->units[i]->x[j+1]=&x[j];
// 	return;
// }

// Calculate start at level 1, level 1 could get input from level 0
void Forward(NEURON_LAYER *L1)
{
	int i,n;
	NEURON_LAYER *L;
	L = L1;
	while (L)
	{
		n=L->num;
		for (i=0;i<n;i++)
			NEURON_Calc(L->units[i]);
		L=L->next_layer;
	}

	return;
}

void ShowLayer(NEURON_LAYER *L1)
{
	int i,n,m,j;
	NEURON_LAYER *L;
	NEURON *N;
	L=L1;
	printf("\n---------Show Neural Network----------\n");
	while (L)
	{
		n=L->num;
		printf("\n[LAYER]\nlevel=%d, num=%d \n",L->level,L->num);
		for (i=0;i<n;i++)
		{
			N=L->units[i];
			m=N->num;
			printf("[NEURON]level=%d, num=%d \n",N->level,N->num);
			printf("[NRURON]z=%lf, a=%lf\n",N->z,N->a);
			for (j=0;j<m;j++)
				printf("%lf ",N->w[j]);
			printf("\n");
		}
		L=L->next_layer;
	}
	return;
}
// 最小二乘法, Ordinary Least Squares
double OLS(double *output,double *result, int n)
{
	int i;
	double sum=0;

	for (i=0;i<n;i++)
		sum+=(result[i]-output[i])*(result[i]-output[i]);
	sum/=n;
	sum*=0.5;

	return sum;
}


// TODO: finish backpropagation
void BackPropagation(NEURON_LAYER *Network[],
					 int num_layer,  
					 double Y[],	     // 数据结果  				
					 LOSS_PF Cost_pf,	 // 损失函数
					 ACT_PF act, 		 // 激活函数
					 ACT_PF act_p,	     // 激活函数的导数
					 double rate	     // 学习速率
					 )
{
	double **grad,deviation[LEN],d1[LEN],d2[LEN];
	double cost=0,Output[LEN],tmp,sum;
	NEURON_LAYER *L;
	int level,i,j,l,n,num;

	grad = malloc(sizeof(double*)*LEN);
	for (i=0;i<LEN;i++)
		grad[i] = malloc(sizeof(double)*LEN);
	// the Output Layer
	level = num_layer-1;
	L = Network[level];
	for (i=0;i<L->num;i++)
		Output[i]=L->units[i]->a;
	// calculate the cost
	cost = Cost_pf(Output,Y,L->num);

	n = L->num;
	for (i=0;i<n;i++)
		d1[i]=(L->units[i]->a-Y[i])*act_p(L->units[i]->z);
	for (i=0;i<n;i++)	// for each neuron unit
		for (j=0;j<L->units[i]->num;j++)	// for each weight
		{
			// calculate the last layer gradiant, but use it later.
			grad[i][j]=d1[i] * L->prev_layer->units[j]->a;
		}

	// the Hidden Layer	
	for (l=level-1;l>0;l--)
	{
		L = Network[l];
		n = L->num;

		// 1. calculate the new deviation
		for (i=0;i<n;i++)
		{
			d2[i] = 0;
			for (j=0;j<L->next_layer->num;j++)
				d2[i] += d1[j] * L->next_layer->units[j]->w[i+1] * act_p(L->units[i]->z);
			
		}

		// 2. update the weight ( we need the old to cal d2)
		for (i=0;i<L->next_layer->num;i++)
		{
			L->next_layer->units[i]->w[0] -= d1[i];
			for (j=0; j < L->next_layer->units[i]->num;j++)
				L->next_layer->units[i]->w[j+1] -= rate*grad[i][j];
		}	
		// 3. transfer d2 to d1 (for loop)
		for (i=0;i<n;i++)
			d1[i]=d2[i];

		// 4. calculate new grad
		for (i=0;i<n;i++)	// for each neuron unit
			for (j=0;j<L->units[i]->num;j++)	// for each weight
			{
				// calculate the last layer gradiant, but use it later.
				grad[i][j]=d1[i] * L->prev_layer->units[j]->a;
			}	

	}
	
	printf("COST = %lf\n",cost);

	free(grad);
	return;
}