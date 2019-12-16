#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "neuron.h"
#include "randtool.h"

//---------------Switches---------------
const int ADAM_ON = 1;
const int L2_ON   = 0;



//---------------Simple Function-------------
double equal(double x) { return x;}
double equal_p(double x) {return 1;}
double pixel_scale(double x) {return x/255.0;}
double pixel_scale_p(double x) {return 1.0/255;}
double sigmoid(double z) {return 1/(1+exp(-z));}
double sigmoid_p(double z) {return sigmoid(z)*(1-sigmoid(z));}

// double tanh(double x)
// {
// 	double e1 = exp(x);
// 	double e2 = exp(-x);
// 	return (e1-e2)/(e1+e2);
// }

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

void softmax(double * activations, int length)
{
    int i;
    double sum, max;

    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
        activations[i] = exp(activations[i] - max);
        sum += activations[i];
    }

    for (i = 0; i < length; i++) {
        activations[i] /= sum;
    }
	return;
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
				int    num, // num of weights
				double b,
				ACT_PF act_f,
				ACT_PF act_fp)
{
	int i;
	N->f = act_f;
	N->fp= act_fp;
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

LAYER *LAYER_New(int level,int num)
{
	LAYER *p;
	int i;
	p=(LAYER*)malloc(sizeof(LAYER));
	if (!p) return NULL;
	p->num=num;
	p->level=level;
	p->prev_layer=NULL;
	p->next_layer=NULL;
	for (i=0;i<num;i++)
		p->units[i]=NEURON_New(level,i);

	return p;
}

void LAYER_Del(LAYER *p)
{
	int i;
	if (!p) return;
	for (i=0;i<p->num;i++)
		NEURON_Del(p->units[i]);
	free(p);

	return;
}
void LAYER_Connect(LAYER *Curr, LAYER *Next)
{
	if (!Curr || !Next) return;
	Curr->next_layer = Next;
	return;
}
void LAYER_Init(LAYER *L,
				LAYER *Pre, 
                int neuron_num, 
				double (*para_gen)(int p1,int p2), 
				ACT_PF act_f,
				ACT_PF act_fp)
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
	if (act_f == NULL) act_f=equal;	
	if (act_fp== NULL) act_fp=equal_p;
	L->num = neuron_num;
	// notice the input level has no parameters
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
			// Notice: Xavier初始化
			for (i=0;i<Pre->num;i++)
				w[i]=para_gen(Pre->num,L->num);
			bias=para_gen(i+1,j+1);
			NEURON_Init(L->units[j],x,w,Pre->num,bias,act_f,act_fp);
		}
	}
	else // level 0
	{
		// inital neuron manually
		// call LAYER_SetInput(L,data,L->num) later
	}
	// else 
	// {
	// 	printf("[LAYER_Init]level == 0 but data == NULL!\n");
	// }
	return;
}
void LAYER_SetInput(LAYER *L,double x[],int num)
{
	int i;
	if (L->level!=0)
	{
		printf("[LAYER_SetInput]You should not set level(%d) besides level 0.\n",L->level);
		return;
	}
	for (i=0;i<L->num;i++)
	{
		L->units[i]->num=0;
		L->units[i]->f=pixel_scale;
		L->units[i]->fp=pixel_scale_p;
		L->units[i]->z=x[i];
		L->units[i]->a=L->units[i]->f(x[i]);
	}
	return;
}

// Calculate start at level 1, level 1 could get input from level 0
void Forward(LAYER *L1)
{
	int i,j,n;
	double z[LEN];
	LAYER *L, *last;
	L = L1;
	// the output layer use softmax
	while (L)
	{
		n=L->num;
		for (i=0;i<n;i++)
			NEURON_Calc(L->units[i]);
		if (L->next_layer == NULL) last = L;
		L=L->next_layer;
	}
	// calculate the last layer manually
	// use softmax
	L = last;
	n=L->num;
	for (i=0;i<n;i++) 
	{
		// z[i]=L->units[i]->w[0];
		// for (j=0;j<L->units[i]->num;j++)
		// 	z[i] += L->units[i]->w[j+1] * (*L->units[i]->x[j+1]);
		// L->units[i]->z = z[i];
		z[i] = L->units[i]->a;
	}
	softmax(z,n);

	for (i=0;i<n;i++)
		L->units[i]->a=z[i];
	return;
}

void ShowLayer(LAYER *L1)
{
	int i,n,m;
	LAYER *L;
	NEURON *N;
	L=L1;
	printf("\n---------Show Neural Network----------\n");
	// while (0)
	{
		n=L->num;
		printf("\n[LAYER]\nlevel=%d, num=%d \n",L->level,L->num);
		for (i=0;i<n;i++)
		{
			N=L->units[i];
			m=N->num;
			printf("[NEURON]index=%d, num=%d \n",i,N->num);
			printf("[NRURON]z=%lf, a=%lf\n",N->z,N->a);
			// for (j=0;j<m;j++)
			// 	printf("%lf ",N->w[j]);
			printf("\n");
		}
		//L=L->next_layer;
	}
	return;
}
// 均方误差, Mean Squared Error
double MSE(double *output,double *result, int n)
{
	int i;
	double sum=0;

	for (i=0;i<n;i++)
		sum+=(result[i]-output[i])*(result[i]-output[i]);
	sum/=n;
	sum*=0.5;

	return sum;
}

// 交叉熵函数， Cross entropy loss
double CEL(double *output,double *result, int n)
{
	int i;
	double sum=0;

	for (i=0;i<n;i++)
		sum += result[i] * log(output[i]); // + (1-result[i]) * log(1-output[i]);
	
	return -sum;
}


double L2_Addtion(double lamda, double w)
{
	return lamda*w;
}

/********* 
 * I changed the last layer, and now the output used softmax, 
 * and the loss function is not MSE but CLE
 *******/
void BackPropagation(NETWORK *net,
					 double Y[],	     // 数据结果
					 GRADIANT *grad
					 )
{
	LAYER **Network;
	double d1[LEN],d2[LEN];
	LAYER *L;
	int num_layer,i,j,l,n;
	double tmp;

	Network = net->layer_header;
	num_layer = net->num_layers;

	for (l=num_layer-1;l>=1;l--)
	{
		L = Network[l];
		n = L->num;
		if (l==num_layer-1)
		{
			// softmax层直接计算传递误差
			for (i=0;i<n;i++)	// for each neuron unit
				d1[i] = L->units[i]->a - Y[i];
		}
		else 
		{
			// 计算隐含层传递误差
			for (i=0;i<n;i++)
			{
				d2[i] = 0;
				for (j=0;j<L->next_layer->num;j++)
					d2[i] += d1[j] * L->next_layer->units[j]->w[i+1] * L->next_layer->units[j]->fp(L->next_layer->units[j]->z);
			}
			// 3. transfer d2 to d1 (for loop)
			for (i=0;i<n;i++)
				d1[i]=d2[i];
		}


		// 更新梯度
		for (i=0;i<n;i++)	// for each neuron unit
		{
			grad->b[l][i] += d1[i];
			tmp = L->units[i]->fp(L->units[i]->z);
			for (j=0;j<L->units[i]->num;j++)	// for each weight
			{
				// calculate the last layer gradiant, but use it later.
				grad->W[l][i][j] += d1[i] * L->prev_layer->units[j]->a * tmp + L2_Addtion(net->L2_lamda, L->units[i]->w[j+1]);
			}
		}
	}

	return;
}

NETWORK *NETWORK_New()
{
	NETWORK *net;
	net = malloc(sizeof(NETWORK));
	memset(net,0,sizeof(net));
	return net;
}
/***
 * activation function is tanh
 ***/
void NETWORK_Init(NETWORK *net, int num_layer[],int n)
{
	int i;
	LAYER *L;
	if (n<0) return;
	net->num_layers=n; 		// the output use softmax layer
	// record input
	
	// initialize neural layer
	// initialize input layer (level 0)
	net->layer_header=calloc(n,sizeof(LAYER*));
	net->layer_header[0]=LAYER_New(0,num_layer[0]);
	L=net->layer_header[0];
	LAYER_Init(L,NULL,num_layer[0],NULL,equal,equal_p);
	// initialize rest layers (level 1~n-2), the n-1 use softmax
	for (i=1;i<n;i++)
	{
		net->layer_header[i]=LAYER_New(i,num_layer[i]);
		L=net->layer_header[i];
		if (i==n-1)
			LAYER_Init(net->layer_header[i],
					net->layer_header[i-1],
					num_layer[i],
					Xavier_Generator,
					tanh,
					tanh_p);
		else
			LAYER_Init(net->layer_header[i],
				net->layer_header[i-1],
				num_layer[i],
				Xavier_Generator,
				equal,
				equal_p);

		LAYER_Connect(net->layer_header[i-1],net->layer_header[i]);
	}

	// initialize the L2 normalization lamda (for weight decay)
	if (L2_ON == 1)
		net->L2_lamda = 0.0001;
	else
		net->L2_lamda = 0;


	printf("The network size is :%d ",num_layer[0]);
	for (i=1;i<n;i++)
		printf(" * %d",num_layer[i]);
	printf("\n");
	printf("ADAM = %d, L2_Normalization = %d\n", ADAM_ON, L2_ON);

	return;
}

double network_loss(NETWORK *net,LOSS_PF loss_f,double *Y)
{
	LAYER *L;
	int i,n,j;
	double Output[LEN],cost;
	int L2_normalization_on = 1;
	double L2_lamda = net->L2_lamda;
	double w_2 = 0;

	n = net->num_layers;
	L = net->layer_header[n-1];
	
	for (i=0;i<L->num;i++)
		Output[i]=L->units[i]->a;
	// calculate the cost
	cost = loss_f(Output,Y,L->num);
	// L2 normalization additional
	
	for (n=1; n<net->num_layers; n++)
	{
		L = net->layer_header[n];
		for (i=0;i<L->num;i++)
			for (j=0;j<L->units[i]->num;j++)
				w_2 += L->units[i]->w[j+1]*L->units[i]->w[j+1];
	}
	cost += w_2 * net->L2_lamda;

	// printf("w_2 = %lf\n",w_2);

	return cost;
}

double NETWORK_Training_OneStep(NETWORK *net,double *in,int in_size,int result,GRADIANT *grad)
{
	LAYER *level0,*level1;
	double loss;
	double Y[10];
	int i;

	level0 = net->layer_header[0];
	level1 = net->layer_header[1];
	LAYER_SetInput(level0, in, in_size);
	Forward(level1);
	for (i=0;i<10;i++)
		if (result==i) Y[i]=1; else Y[i]=0;
	BackPropagation(net, Y, grad);
	loss = network_loss(net, CEL, Y);

	return loss;
}

#define MAX_STEP 2000
double adam(ADAM_PARA *para, double gt, int t)
{
	static int init = 0;
	static double lut_b1[MAX_STEP];
	static double lut_b2[MAX_STEP];
	static double m0 = 0;
	static double v0 = 0;

	double a, b1, b2;
	double mt,vt;
	int i;

	a = para->alpha;
	b1 = para->beta1;
	b2 = para->beta2;
	// accelerate 
	if (init == 0)
	{
		init = 1;
		lut_b1[0]=lut_b2[0]=1;
		for (i=1; i<MAX_STEP; i++)
		{
			lut_b1[i]=lut_b1[i-1]*b1;
			lut_b2[i]=lut_b2[i-1]*b2;
		}
	}

	gt = gt;
	m0 = b1*m0 + (1-b1)*gt;
	v0 = b2*v0 + (1-b2)*gt*gt;
	mt = m0/(1-lut_b1[t]);
	vt = v0/(1-lut_b2[t]);
	// return delta : W[i] -= delta
	return a * mt / (sqrt(vt)+para->epsilon);
}

double Update_Inf(double gt, ADAM_PARA *para, int t, int sw)
{
	// default rate is 0.1
	if (sw==0) return gt*0.1;
	return adam(para,gt,t);
}

void NETWORK_Update_Gradiant(NETWORK *net, GRADIANT *grad, int set_size, ADAM_PARA *para)
{
	static int t = 0;
	int i,j,l;
	LAYER *L;
	int adam_on = ADAM_ON; // switch of adam
	// start at level 1
	if (adam_on == 1) t++;

	for (l=1; l<net->num_layers; l++)
	{
		L = net->layer_header[l];
		for (i=0;i<L->num;i++)
		{
			L->units[i]->w[0] -= Update_Inf(grad->b[l][i]/set_size, para, t, adam_on);
			for (j=0;j<L->units[i]->num; j++)
				L->units[i]->w[j+1] -= Update_Inf(grad->W[l][i][j]/set_size, para, t, adam_on);
		}
	}
	return;
}
//  if correct return 1 else return 0
int NETWORK_Calculate_Accuracy(NETWORK *net,double *in,int in_size, int result)
{
	LAYER *level0,*level1,*last;
	// double loss;
	double m;
	int i,predict;

	i=net->num_layers-1;
	level0 = net->layer_header[0];
	level1 = net->layer_header[1];
	last   = net->layer_header[i];
	LAYER_SetInput(level0, in, in_size);
	Forward(level1);
	predict=0;
	m=last->units[0]->a;
	for (i=1;i<last->num;i++)
		if (last->units[i]->a > m)
		{
			predict=i;
			m=last->units[i]->a;
		}
	if (predict == result) return 1;
	return 0;
}

GRADIANT *GRADIANT_New()
{
	GRADIANT *g;
	g = malloc(sizeof(GRADIANT));
	memset(g,0,sizeof(GRADIANT));
	return g;
}

void ADAM_PARA_Init(ADAM_PARA *para)
{
	para->alpha = 0.002;
	para->beta1 = 0.9;
	para->beta2 = 0.999;
	para->epsilon = 1E-8;
	return;
}

void Debug_ShowOutput(NETWORK *net, int result)
{
	int n=net->num_layers;
	LAYER *L = net->layer_header[n-1];
	int i;
	printf("\n layer output: %d\n", result);
	i = result;
	for (i=0;i<L->num;i++)
		printf("%.3lf\t", L->units[i]->a);
	printf("\n");
	return;
}