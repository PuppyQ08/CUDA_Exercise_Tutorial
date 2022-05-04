#include <iostream>
#include <vector>
#include <ctime>
#include <math.h>

#define PI 3.1415926535897931f
static int samples = 30e6;


int main(void){
    std::clock_t start;
    double duration;
    start = std::clock();
    int count = 0;
    for (int i = 0;i < samples; ++i){
        float x = float(rand()) / RAND_MAX;
        float y = float(rand()) / RAND_MAX;
        if (x*x + y*y < 1)
            count++;
    }
    double error = fabs(PI - 4.0 * count/samples);   

    duration = (std::clock()-start)/(double) CLOCKS_PER_SEC;
    printf("CPU time took %7.5f sec and error of %.8f", duration, error);
}
