#include<math.h>
#include<stdlib.h>
#include<time.h>

double rand_uniform(double left, double right)
{
    static int state = 0;
    // init seed, once
    if (state==0)
    {
        srand((unsigned int)time(NULL));
        state++;
    }

    double range = (right-left)*1024;

    if (range<0) return left;
    return left + ((double)rand())*range/RAND_MAX/1024;
}

