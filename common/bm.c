#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "bm.h"

int bm_start(struct BM_Data* data)
{
    if (data->state != READY)
    {
        printf("bm not ready");
        return EXIT_FAILURE;
    }

    data->start = omp_get_wtime();
    data->state = RUNNING;

    return EXIT_SUCCESS;
}

int bm_end(struct BM_Data* data)
{
    if (data->state != RUNNING)
    {
        printf("can't stop unless bm is running");
        return EXIT_FAILURE;
    }

    data->end = omp_get_wtime();
    data->state = FINISHED;

    return EXIT_SUCCESS;
}

int bm_batch(struct BM_Data* data, int size, int (*func)(int))
{
    if (size < 1)
    {
        printf("no data given");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < size; ++i)
    {
        if (bm_start(data) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        if (func(i) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        if (bm_end(data) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        if (i < size - 1)
            ++data;
    }

    return EXIT_SUCCESS;
}

int bm_print(struct BM_Data* data)
{
    if (data->state != FINISHED)
    {
        printf("can't print non finished bm");
        return EXIT_FAILURE;
    }

    double ms = (data->end - data->start) * 1e3,
           dms = omp_get_wtick() * 1e3;

    printf("Measured time: %.6fms +/- %.6fms\n", ms, dms);

    return EXIT_SUCCESS;
}

int bm_print_batch(struct BM_Data* data, int size)
{
    if (size < 1)
    {
        printf("no data to print");
        return EXIT_FAILURE;
    }

    double sum = 0;
    for (int i = 0; i < size; ++i)
    {
        if (data->state != FINISHED)
        {
            printf("can't print non finished bm");
            return EXIT_FAILURE;
        }

        printf("%d ", i);
        bm_print(data);

        sum += data->end - data->start;

        if (i < size - 1)
            ++data;
    }

    double ms = (sum / size) * 1e3,
           dms = (sqrt(size) / size) * omp_get_wtick() * 1e3;

    printf("Average time: %.6fms +/- %.9fms\n", ms, dms);

    return EXIT_SUCCESS;
}