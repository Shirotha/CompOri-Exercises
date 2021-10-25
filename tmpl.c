#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <bm.h>

int main(int argc, char* argv[])
{
    struct BM_Data bm;
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // TODO: code goes here

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}