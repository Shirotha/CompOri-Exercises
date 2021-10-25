#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <bm.h>

#define SIZE 10

void read(int* data)
{
    printf("read data\n");
    *data = 1;
}

void process(int* data)
{
    printf("process_data\n");
    (*data)++;
}

int batch_test(int i)
{
    int sum;
    for (int k = 1; k <= SIZE * SIZE; ++k)
        sum += (i + 1) * k;

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    printf("processor count = %d\n", omp_get_num_procs());
    printf("number of threads = %d\n", omp_get_max_threads());

    struct BM_Data bm;
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    {
        int count = 0;
        #pragma omp parallel num_threads(SIZE)
        {
            #pragma omp atomic
            ++count;
        }
        printf("atomic count: %d\n", count);
    }

    {
        int i, max, a[SIZE];

        for (i = 0; i < SIZE; ++i)
        {
            a[i] = rand();
            printf("a[%d] = %d\n", i, a[i]);   
        }

        max = a[0];
        #pragma omp parallel for num_threads(4)
        for (i = 1; i < SIZE; ++i)
            if (a[i] > max)
                #pragma omp critical
                if (a[i] > max)
                    max = a[i];

        printf("maximum a = %d\n", max);
    }

    {
        int data, flag = 0;

        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                printf("Thread %d: ", omp_get_thread_num());
                read(&data);
                #pragma omp flush(data)
                flag = 1;
                #pragma omp flush(flag)
            }
            #pragma omp section
            {
                while (!flag)
                {
                    #pragma omp flush(flag)
                }
                #pragma omp flush(data)

                printf("Thread %d: ", omp_get_thread_num());
                process(&data);
                printf("data = %d\n", data);
            }
        }
    }

    {
        int i, nRet = 0, nSum = 0, nStart = 1, nEnd = SIZE;
        int nThreads = 0, nTmp = nStart + nEnd;
        unsigned uTmp = ((unsigned)(abs(nStart - nEnd) + 1) * (unsigned)(abs(nTmp))) / 2;
        int nSumCalc = uTmp;

        if (nTmp < 0)
            nSumCalc = -nSumCalc;

        omp_set_num_threads(4);

        #pragma omp parallel default(none) private(i) shared(nSum, nThreads, nStart, nEnd)
        {
            #pragma omp master
            nThreads = omp_get_num_threads();

            #pragma omp for reduction(+:nSum)
            for (i = nStart; i <= nEnd; ++i)
                nSum += i;
        }

        printf("%d Threads were used\nThe sum of %d through %d is %d\n", nThreads, nStart, nEnd, nSum);
    }

    {
        int i, a[SIZE];

        #pragma omp parallel
        {
            #pragma omp for
            for (i = 0; i < SIZE; i++)
                a[i] = i * i;

            #pragma omp master
            for (i = 0; i < SIZE; i++)
                printf("a[%d] = %d\n", i, a[i]);

            #pragma omp barrier

            #pragma omp for
            for (i = 0; i < SIZE; ++i)
                a[i] += i;
        }
    }

    {
        int i;
        // FIXME: how does this actually work?
        #pragma omp parallel for schedule(static) ordered
        for (i = 0; i < SIZE; ++i)
            printf("iter %d\n", i);
    }

    {
        for (int i = 0; i < SIZE; ++i)
            #pragma omp task if(i % 2 == 0)
            printf("task %d\n", i);

        #pragma omp taskwait
        printf("all finished\n");
    }

    {
        int data;
        #pragma omp task depend(out:data)
        read(&data);
        #pragma omp task depend(in:data)
        process(&data);
    }

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;


    printf("batch test\n");
    struct BM_Data batch_bm[SIZE];
    for (int i = 0; i < SIZE; ++i)
        batch_bm[i].state = READY;

    if (bm_batch(batch_bm, SIZE, batch_test) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (bm_print_batch(batch_bm, SIZE) != EXIT_SUCCESS)
        return EXIT_FAILURE;



    return EXIT_SUCCESS;
}