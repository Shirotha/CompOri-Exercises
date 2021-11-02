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
// TODO: dynamic step size? use symmetry?

#define MAX_TASKS 1000
#define STRIDE MAX_TASKS

#define BATCH_SIZE 1000

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

void init_array(int* a, int len)
{
    for (int i = 0; i < len; ++i)
    a[i] = rand() % 1000;
}

void quicksort(int *a, int len, int param) {
    if (len < 2) return;
    
    int pivot = a[len / 2];
    
    int i, j;
    for (i = 0, j = len - 1; ; i++, j--) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
    
        if (i >= j) break;
    
        int temp = a[i];
        a[i]     = a[j];
        a[j]     = temp;
    }

    #pragma omp task if(param > 0)
    quicksort(a, i, param - 1);
    #pragma omp task if(param > 0)
    quicksort(a + i, len - i, param - 1);
    //#pragma omp taskwait
}

static int param = 0;

int quicksort_test(int i)
{
    int array[SIZE];
    init_array(array, SIZE);
    #pragma omp parallel
    #pragma omp single
    quicksort(array, SIZE, param);
    #pragma omp taskwait

    return EXIT_SUCCESS;
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

    int i = 0, j = 0;
    double y = 0.0;

#pragma region quicksort
    /*
    printf("quicksort\n");

    struct BM_Data bms[SIZE];

    for (int p = 0; p < 16; ++p)
    {
        for (i = 0; i < SIZE; ++i)
            bms[i].state = READY;

        param = p;
        bm_batch(bms, SIZE, quicksort_test);
        
        if (bm_print_batch(bms, SIZE) != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
    */
#pragma endregion

#pragma region Convolution - single thread
    printf("Convolution (seriel)\n");
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

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

#pragma region Convolution - multi thread (static)
    printf("Convolution (parallel - static)\n");
    bm.state = READY;
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    #pragma omp parallel for schedule(static) default(none) shared(conv, step, max_r) private(i, j, y)
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

#pragma region Convolution - multi thread (dynamic)
    printf("Convolution (parallel - dynamic)\n");
    bm.state = READY;
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    #pragma omp parallel for schedule(dynamic) default(none) shared(conv, step, max_r) private(i, j, y)
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
    printf("Convolution (%d tasks)\n", MAX_TASKS);
    bm.state = READY;
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    #pragma omp parallel
    #pragma omp single
    for (int k = 0; k < MAX_TASKS; ++k)
        #pragma omp task default(none) shared(conv, step, max_r) private(k, i, j, y)
            for (i = k; i < SIZE; i += STRIDE)
            {
                y = (i + 0.5) * step - max_r;
                for (j = 0; j < SIZE; ++j)
                    conv[i * SIZE + j] = gauss_conv((j + 0.5) * step - max_r, y, max_r);
            }
    #pragma omp taskwait

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
#pragma endregion

// TODO: plot convolution results

    printf("Batch size for time measure = %d\n", BATCH_SIZE);

    double sum = 0;
    int k;
#pragma region Integral - ordered
    bm.state = READY;
    printf("Integral (forwards)\n");
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    for (k = 0; k < BATCH_SIZE; ++k)
    {
        sum = 0;
        for (i = 0; i < SIZE; ++i)
            for (j = 0; j < SIZE; ++j)
                sum += conv[i * SIZE + j];

        sum *= step * step;
    }

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
#pragma endregion

#pragma region Integral - ordered (backwards)
    bm.state = READY;
    printf("Integral (backwards)\n");
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    for (k = 0; k < BATCH_SIZE; ++k)
    {
        sum = 0;
        for (i = SIZE - 1; i >= 0; --i)
            for (j = SIZE - 1; j >= 0; --j)
                sum += conv[i * SIZE + j];

        sum *= step * step;
    }

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
#pragma endregion

#pragma region Integral - parallel
    bm.state = READY;
    printf("Integral (parallel)\n");
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;


    double pSum;
    for (k = 0; k < BATCH_SIZE; ++k)
    {
        pSum = 0;
        #pragma omp parallel for default(none) shared(conv, sum)  private(i, j) reduction(+: pSum)
        for (i = 0; i < SIZE; ++i)
            for (j = 0; j < SIZE; ++j)
                pSum += conv[i * SIZE + j];

        pSum *= step * step;
        sum = pSum;
    }

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
#pragma endregion

#pragma region Integral - unordered
    bm.state = READY;
    printf("Integral (force cache miss)\n");
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    for (k = 0; k < BATCH_SIZE; ++k)
    {
        sum = 0;
        for (j = 0; j < SIZE; ++j)
            for (i = 0; i < SIZE; ++i)
                sum += conv[i * SIZE + j];

        sum *= step * step;
    }

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
#pragma endregion

    printf("result = %f\n", sum);

    return EXIT_SUCCESS;
}