#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <bm.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

#define SIZE 100
#define AMP 0.3989422804014327 
#define THRESHOLD 0.01
#define DEFAULT_STEP 0.01

#define CONV_SIZE 100
// TODO: dynamic step size?

#define RSTD1 1.0
#define RSTD2 1.0

double gauss_conv(double x, double y, double max_r)
{
    int i, j;
    double x1, x2, y1, y2;
    double sum, step = max_r * 2 / SIZE;
    for (i = 0; i < SIZE; ++i)
        for (j = 0; j < SIZE; ++j)
        {
            x1 = step * j - max_r;
            y1 = step * i - max_r;
            x2 = x1 - x;
            y2 = y1 - y;
            sum += exp(-(x1 * x1 + y1 * y1) * 0.5 * RSTD1 * RSTD1 - 0.5 * RSTD2 * RSTD2 * (x2 * x2 + y2 * y2));
        }
    
    return sum * step * step * AMP * AMP * RSTD1 * RSTD2;
}

int main(int argc, char* argv[])
{
    double max_r = 0, y = 1, rstd = MIN(RSTD1, RSTD2);

    while (AMP * rstd * exp(-max_r * max_r * rstd * rstd) > THRESHOLD)
        max_r += DEFAULT_STEP;

    double step_size = max_r * 2 / SIZE;

    printf("Integral to %.16f in %.16f steps\n", max_r, step_size);

    double conv[SIZE * SIZE];

    struct BM_Data bm;
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    int i, j;
    #pragma parallel for private(i,j)
    for (i = 0; i < SIZE; ++i)
        for (j = 0; j < SIZE; ++j)
            conv[i * SIZE + j] = gauss_conv(j * step_size - max_r, i * step_size - max_r, max_r);

    // TODO: restart timer here

    double sum;
    for (i = 0; i < SIZE; ++i)
        for (j = 0; j < SIZE; ++j)
            sum += step_size * step_size * conv[i * SIZE + j];

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    printf("result = %f\n", sum);

    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}