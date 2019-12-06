#ifndef NEURON_H_
#define NEURON_H_

#define LEN 1000


//---------Type Declaration-------
typedef double (*ACT_PF)(double z);
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
	ACT_PF f;
	// x is input, w is weight, w[0] is bias, assume x[0] = 1
	// num is valid elements numbers of weight without w[0]
	int num;// num of x and w
	double *x[LEN],w[LEN];
	// DO NOT TOUCH X[0], I WILL NERVER DEFINE IT. AND I THINK X[0] EQUALS 1, WITH W[0] IS bias
} NEURON;


typedef struct tag_layer
{
	int num,level;  // num of units
	struct tag_layer * next_layer;
	NEURON *units[LEN];
} NEURON_LAYER;




//--------------Function Declaration--------------
double equal(double x);
double sigmoid(double z);
double tanh(double x);
NEURON *NEURON_New(int level,int index);
void NEURON_Del(NEURON* p);
void NEURON_Init(NEURON *N,
				double *x[LEN],
				double w[LEN],
				double num,
				double b,
				ACT_PF act_pf);
void NEURON_Calc(NEURON *N);
NEURON_LAYER *LAYER_New(int level,int num);
void LAYER_Del(NEURON_LAYER *p);
void LAYER_Connect(NEURON_LAYER *Curr, NEURON_LAYER *Next);
void LAYER_Init(NEURON_LAYER *L,NEURON_LAYER *Pre, int neuron_num, 
				double (*para_gen)(int p1,int p2), 
				ACT_PF act_pf,
				double data[]);
// void LAYER_FullConnectInput(NEURON_LAYER *L,double x[],int num);
void Forward(NEURON_LAYER *L1);
void ShowLayer(NEURON_LAYER *L1);


#endif /* NEURON_H_ */
