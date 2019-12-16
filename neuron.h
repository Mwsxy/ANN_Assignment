#ifndef NEURON_H_
#define NEURON_H_

#define LEN 785
#define _MAX_LAYER_ 5


//---------Type Declaration-------
typedef double (*ACT_PF)(double z);
typedef double (*LOSS_PF)(double *output,double *result, int n);
/***
 *     \
 *    ---w1
 *    ---w2----(a   |   f(a)=z) ----
 *     /
 **/
typedef struct tag_neuron
{
	int level,index;
	double z,a;
	// f is activation function, fp is the derivative of f
	ACT_PF f,fp;
	// x is input, w is weight, w[0] is bias, assume x[0] = 1
	// num is valid elements numbers of weight without w[0]
	int num;// num of x and w
	double *x[LEN],w[LEN];
	// DO NOT TOUCH X[0], I WILL NERVER DEFINE IT. AND I THINK X[0] EQUALS 1, WITH W[0] IS bias
} NEURON;


typedef struct tag_layer
{
	int num,level;  // num of units
	struct tag_layer *prev_layer,*next_layer;
	NEURON *units[LEN];
} LAYER;

typedef struct tag_network
{
	int num_layers;
	// an array of (LAYER *)
	LAYER **layer_header;
	// L2 normalization parameters
	double L2_lamda;
} NETWORK;

typedef struct tag_gradiant
{
	double W[_MAX_LAYER_][LEN][LEN];
	double b[_MAX_LAYER_][LEN];
} GRADIANT;

typedef struct tag_adam_para
{
	// parameters for admam 
	double alpha,beta1,beta2,epsilon;
} ADAM_PARA;
//--------------Function Declaration--------------
double equal(double x);
double sigmoid(double z);
double sigmoid_p(double z);
double tanh(double x);
double tanh_p(double x);
double MSE(double *output,double *result, int n);
double CEL(double *output,double *result, int n);
NEURON *NEURON_New(int level,int index);
void NEURON_Del(NEURON* p);
void NEURON_Init(NEURON *N,
				double *x[LEN],
				double w[LEN],
				int num,
				double b,
				ACT_PF act_f,
				ACT_PF act_fp);
void NEURON_Calc(NEURON *N);
LAYER *LAYER_New(int level,int num);
void LAYER_Del(LAYER *p);
void LAYER_Connect(LAYER *Curr, LAYER *Next);
void LAYER_Init(LAYER *L,
				LAYER *Pre, 
				int neuron_num, 
				double (*para_gen)(int p1,int p2), 
				ACT_PF act_f,   // activation function
				ACT_PF act_fp);
void LAYER_SetInput(LAYER *L,double x[],int num);
void Forward(LAYER *L1);
void BackPropagation(NETWORK *net,
					double Y[],	     // 数据结果  				
					GRADIANT *grad
					);
void ShowLayer(LAYER *L1);
NETWORK *NETWORK_New();
void NETWORK_Init(NETWORK *net, int num_layer[],int n);
double NETWORK_Training_OneStep(NETWORK *net,double *in,int size,int result, GRADIANT *grad);
void NETWORK_Update_Gradiant(NETWORK *net, GRADIANT *grad, int set_size, ADAM_PARA *para);
int NETWORK_Calculate_Accuracy(NETWORK *net,double *in,int in_size, int result);

GRADIANT *GRADIANT_New();
void ADAM_PARA_Init(ADAM_PARA *para);

void Debug_ShowOutput(NETWORK *net, int result);

#endif /* NEURON_H_ */
