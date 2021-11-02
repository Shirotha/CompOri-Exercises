#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <slepceps.h>

#include "../common/bm.h"

static PetscErrorCode ierr;
#define CHK(X) do { X; CHKERRQ(ierr); } while(0)
#define PRINT(format, args...) CHK(PetscPrintf(PETSC_COMM_WORLD, format "\n", args));

int main(int argc, char* argv[])
{
    struct BM_Data bm;
    if (bm_start(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    {
        CHK(SlepcInitialize(&argc, &argv, (char*)0, (char*)0));

        PetscInt n=30;
        CHK(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
        PRINT("Matrix Size = %d", n);

        Mat H;
        CHK(MatCreate(PETSC_COMM_WORLD, &H));
        CHK(MatSetSizes(H, PETSC_DECIDE, PETSC_DECIDE, n, n));
        CHK(MatSetFromOptions(H));
        CHK(MatSetUp(H));

        PetscInt i, begin, end;
        CHK(MatGetOwnershipRange(H, &begin, &end));
        for (i = begin; i < end; ++i)
        {
            if (i > 0)
                CHK(MatSetValue(H, i, i - 1, -1.0, INSERT_VALUES));
            if (i < n - 1)
                CHK(MatSetValue(H, i, i + 1, -1.0, INSERT_VALUES));

            CHK(MatSetValue(H, i, i, 2.0, INSERT_VALUES));
        }

        CHK(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
        CHK(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));

        Vec xr, xi;
        CHK(MatCreateVecs(H, NULL, &xr));
        CHK(MatCreateVecs(H, NULL, &xi));

        EPS eps;
        CHK(EPSCreate(PETSC_COMM_WORLD, &eps));

        CHK(EPSSetOperators(eps, H, NULL));
        CHK(EPSSetProblemType(eps, EPS_HEP));

        CHK(EPSSetFromOptions(eps));

        CHK(EPSSolve(eps));

        {
            PetscInt iterations;
            CHK(EPSGetIterationNumber(eps, &iterations));
            PRINT("Iterations = %d", iterations);
        }

        {
            EPSType type;
            CHK(EPSGetType(eps, &type));
            PRINT("Type = %d", type);
        }

        {
            PetscInt dimension;
            CHK(EPSGetDimensions(eps, &dimension, NULL, NULL));
            PRINT("Result Dimension = %d", dimension);
        }

        {
            PetscInt iterations;
            PetscReal tolerance;
            CHK(EPSGetTolerances(eps, &tolerance, &iterations));
            PRINT("Tolerance = %.4f, Max. Iterations = %d", tolerance, iterations);
        }

        {
            PetscInt convergedEigenvalues;
            CHK(EPSGetConverged(eps, &convergedEigenvalues));
            PRINT("Converged Eigenvalues = %d", convergedEigenvalues);

            if (convergedEigenvalues > 0)
            {
                PetscScalar kr, ki;
                PetscReal re, im, error;
                for (i = 0; i < convergedEigenvalues; ++i)
                {
                    CHK(EPSGetEigenpair(eps, i, &kr, &ki, xr, xi));
                    re = PetscRealPart(kr);
                    im = PetscImaginaryPart(ki);

                    CHK(EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error));

                    PRINT("%9f + %9fi +/- %12f", re, im, error);
                }
            }
        }

        CHK(EPSDestroy(&eps));
        CHK(MatDestroy(&H));
        CHK(VecDestroy(&xr));
        CHK(VecDestroy(&xi));
        CHK(SlepcFinalize());

    }

    if (bm_end(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    
    if (bm_print(&bm) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    return ierr;
}