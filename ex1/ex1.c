#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <bm.h>

#define SQ(X) ((X) * (X))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

#define SIZE 1000
#define AMP 0.3989422804014327 
#define THRESHOLD 1e-10
#define DEFAULT_STEP 0.1

#define CONV_SIZE 20
// TODO: dynamic step size?

#define MAX_TASKS 20
#define STRIDE SIZE / MAX_TASKS

#define RSTD1 2.5066282746310002 
#define RSTD2 2.5066282746310002 

double gauss_conv(const double x, const double y, const double max_r)
{
    double tx, ty;
    const double step = max_r * 2.0 / CONV_SIZE;
    double sum = 0;
    for (tx = 0.5 * step - max_r; tx < max_r; tx += step)
        for (ty = 0.5 * step - max_r; ty < max_r; ty += step)
            sum += exp(-0.5 * SQ(RSTD1) * (SQ(tx) + SQ(ty)) - 0.5 * SQ(RSTD2) * (SQ(tx - x) + SQ(ty - y)));
    
    return sum * SQ(step) * SQ(AMP) * RSTD1 * RSTD2;
}

int main(int argc, char* argv[])
{
    double tmp = 0, rstd = MIN(RSTD1, RSTD2);

    while (AMP * rstd * exp(-SQ(tmp) * SQ(rstd)) > THRESHOLD)
        tmp += DEFAULT_STEP;
    
    const double max_r = tmp;
    const double step = max_r * 2.0 / SIZE;

    printf("Integral to %.16f in %d %.16f steps\n", max_r, SIZE, step);

    double conv[SQ(SIZE)];
    struct BM_Data bm;

#pragma region Convolution - single thread
    printf("Convolution (seriel)\n");
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    int i, j;
    double y;
    for (i = 0; i < SIZE; ++i)
    {
        y = (i + 0.5) * step - max_r;
        for (j = 0; j < SIZE; ++j)
            conv[i * SIZE + j] = gauss_conv((j + 0.5) * step - max_r, y, max_r);
    }

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
#pragma endregion

#pragma region Convolution - multi thread
    printf("Convolution (parallel)\n");
    bm.state = READY;
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    #pragma omp parallel for private(i,j,y)
    for (i = 0; i < SIZE; ++i)
    {
        y = (i + 0.5) * step - max_r;
        for (j = 0; j < SIZE; ++j)
            conv[i * SIZE + j] = gauss_conv((j + 0.5) * step - max_r, y, max_r);
    }

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
#pragma endregion

#pragma region Convolution - tasks
    /* FIXME: breaks for large SIZE
    printf("Convolution (tasks)\n");
    bm.state = READY;
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
        
    for (i = 0; i < SIZE; ++i)
    {
        y = (i + 0.5) * step - max_r;
        #if (SIZE > MAX_TASKS)
            for (int k = 0; k < MAX_TASKS; ++k)
                #pragma omp task
                    for (j = k; j < SIZE; j += STRIDE)
                        conv[i * SIZE + j] = gauss_conv((j + 0.5) * step - max_r, y, max_r);
        #elif
            #pragma omp task
                for (j = 0; j < SIZE; ++j)
                    conv[i * SIZE + j] = gauss_conv((j + 0.5) * step - max_r, y, max_r);
        #endif
    }
    #pragma omp taskwait

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    */
#pragma endregion

#pragma region Integral - ordered
    bm.state = READY;
    printf("Integral\n");
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    double sum = 0;
    for (i = 0; i < SIZE; ++i)
        for (j = 0; j < SIZE; ++j)
            sum += conv[i * SIZE + j];

    sum *= step * step;

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
#pragma endregion

#pragma region Integral - unordered
    bm.state = READY;
    printf("Integral (cache miss)\n");
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    sum = 0;
    for (j = 0; j < SIZE; ++j)
        for (i = 0; i < SIZE; ++i)
            sum += conv[i * SIZE + j];

    sum *= step * step;

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
#pragma endregion

    printf("result = %f\n", sum);
    return EXIT_SUCCESS;
}