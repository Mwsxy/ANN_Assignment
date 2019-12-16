#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include "neuron.h"
#include "randtool.h"
#include "mnist_file.h"
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

	NEURON_Init(N,x,w,3,b,sigmoid,sigmoid_p);
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
    // LAYER *L0,*L1,*L2,*L3,*LA[10];
    // double Input[LEN],Output[LEN];
    // int num[]={3,5,3},num0,num_layer;
    // int i,j,k;
	// int iter_n; // 迭代次数
	// GRADIANT *grad;

    // num_layer=sizeof(num)/sizeof(int);
    // // input layer L0
    // num0=num[0];
    // for (i=0;i<num0;i++)
	// {
	// 	Input[i]=rand_uniform(-5,10);
	// 	Output[i] = rand_uniform(0,15);
	// }

	// // Input[0]=Input[1]=Input[2]=1;
	// // Output[0]=Output[1]=Output[2]=2;


    // i=0;
    // LA[i] = LAYER_New(i,num[i]);
    // LAYER_Init(LA[i],NULL,num[i],NULL,NULL,NULL);
	// LAYER_SetInput(LA[i],Input,num0);
    // // layer Li
    // for (i=1;i<num_layer;i++)
    // {
    //     LA[i]=LAYER_New(i,num[i]);
    //     LAYER_Init(LA[i],LA[i-1],num[i],NULL,tanh,tanh_p);
    //     LAYER_Connect(LA[i-1],LA[i]);
    // }
    // //------finish: initialization------

    // Forward(LA[1]);
    // ShowLayer(LA[0]);
	// iter_n = 10;
	// grad = malloc(sizeof(GRADIANT));
	// memset(grad,0,sizeof(GRADIANT));
	// for (int  i = 0; i < iter_n; i++)
	// {
	// 	BackPropagation(LA,num_layer,Output,MSE,grad);
	// 	Forward(LA[1]);
	// }
	// free(grad);
    return;
}

double Training_dataset_Accuracy(mnist_dataset_t *dataset, NETWORK *net)
{
	double input[LEN];
	int result;
	double corr = 0, acc;
	int j,k;

	for (j=0; j<(int)dataset->size; j++)
	{
		result=dataset->labels[j];
		for (k=0;k<MNIST_IMAGE_SIZE;k++)
			input[k]=dataset->images[j].pixels[k];
		corr+=NETWORK_Calculate_Accuracy(net, input, MNIST_IMAGE_SIZE, result);
	}
	acc = corr / (double)dataset->size;
	return acc;
}

// test load data and test train
#define STEPS 10
#define BATCH_SIZE 64

void test4()
{
	const char * train_images_file = "data\\train-images.idx3-ubyte";
	const char * train_labels_file = "data\\train-labels.idx1-ubyte";
	const char * test_images_file = "data\\t10k-images.idx3-ubyte";
	const char * test_labels_file = "data\\t10k-labels.idx1-ubyte";
	mnist_dataset_t * train_dataset, * test_dataset, * restruct_dataset;
    mnist_dataset_t batch;
    double loss, total_loss;
    int i, j, k, batches, result, step;

/**
 * 	95%			784*128*64*10
 *  90%			784*       10
 * 	90%			784*512*512*10
 * 
 * */
	NETWORK *net;
	double rate = 0.5;
	int layer_size[]={28*28, 128, 128, 10},num_l;
	double input[LEN];
	ADAM_PARA para;
	GRADIANT *grad;
	int corr;
	double acc;
	double stop_predict = 1.0;

	num_l = sizeof(layer_size)/sizeof(int);

	// init network
	net = NETWORK_New();
	NETWORK_Init(net, layer_size, num_l);
	// init gradiant
	grad = GRADIANT_New();
	ADAM_PARA_Init(&para);

    // Read the datasets from the files
    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    // Calculate how many batches (so we know when to wrap around)
    batches = train_dataset->size / BATCH_SIZE;

	
	for (step=0; step<STEPS; step++)
	{
		// Mix the training data
		k = train_dataset->size;
		for (i=0;i<k;i++)
		{
			j = rand() % k;
			mnist_image_t t1;
			uint8_t t2;
			t1 = train_dataset->images[i];
			t2 = train_dataset->labels[i];
			train_dataset->images[i]=train_dataset->images[j];
			train_dataset->labels[i]=train_dataset->labels[j];
			train_dataset->images[j]=t1;
			train_dataset->labels[j]=t2;
		}
		
		total_loss = 0;
		for (i=0; i<batches; i++)
		{
			// Initialise a new batch
			mnist_batch(train_dataset, &batch, BATCH_SIZE, i % batches);
			/* * * * * * * * * * * * * * 
			* TODO: batch-normalization
			* * * * * * * * * * * * * **/
			loss = 0;
			memset(grad,0,sizeof(GRADIANT));
			for (j=0; j<(int)batch.size; j++)
			{
				result = batch.labels[j];
				for (k=0;k<MNIST_IMAGE_SIZE;k++)
					input[k]=batch.images[j].pixels[k];
				
				loss += NETWORK_Training_OneStep(net,input, MNIST_IMAGE_SIZE, result, grad);
			}
			loss/=(double)batch.size;
			total_loss+=loss;
			NETWORK_Update_Gradiant(net, grad, batch.size, &para);
		}
		total_loss = total_loss / (double)batches;
		acc = Training_dataset_Accuracy(train_dataset, net);
		printf("Step %02d, Loss = %.4lf, accuracy in training set = %.4lf\%\n", step, total_loss, acc*100);
		acc = Training_dataset_Accuracy(test_dataset, net);
		printf("ACC in test set = %.4lf\%\n", acc*100);
	}
	free(grad);
	// Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);
	return;
}

typedef void (*PF)(void);

void TEST()
{
    PF T[]={
        // test1,
        // test2,
        // test3,
		test4,
    };
    int i,n=sizeof(T)/sizeof(PF);

    for (i=0;i<n;i++)
        T[i]();
    return;
}
