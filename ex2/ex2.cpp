#include "../common/bm.hpp"
#include "../common/la.hpp"

int main(int argc, char* argv[])
{
    bm::BM bm;
    bm.start();
    MPI_Init(&argc, &argv);
    E(SlepcInitialize(&argc, &argv, (char*)0, (char*)0));
    {
        PetscMPIInt rank, size;
        MPI_Comm_size(PETSC_COMM_WORLD, &size);
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        PRINT("MPI Size = %i, MPI Rank = %i", size, rank);
        PetscInt n=30;
        E(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
        PRINT("Matrix Size = %d", n);

        auto grid = std::make_shared<la::Grid>(DM_BOUNDARY_GHOSTED, n, 1, 1, -1.0, 1.0);
        auto H = std::make_shared<la::Matrix>(grid);
        
        H->apply_cfd<2, 2>(INSERT_VALUES, -0.5);

        H->assemble(MAT_FINAL_ASSEMBLY);

        auto solver = std::make_shared<la::EigenSolver>(H, EPS_HEP);

        bm.lap();

        solver->solve();

        bm.lap();

        PRINT("Iterations = %i", solver->getNumIterations());
        PRINT("Type = %i", solver->getType());
        PRINT("Result Dimension = %i", solver->getDimension());
        

        {
            auto [iter, tol] = solver->getTolerance();
            PRINT("Tolerance = %.4E, Max. Iterations = %i", iter, tol);
        }

        {
            int i, n = solver->getNumEigenpairs();
            PRINT("Converged Eigenvalues = %i", n);

            if (n > 0)
            {
                PetscScalar ev;
                PetscReal error;
                auto x = std::make_shared<la::Vector>(grid);

                for (i = 0; i < n; ++i)
                {
                    solver->getEigenpair(i, ev, *x, error);
                    PRINT("%9E + %9Ei +/- %12E", ev.real(), ev.imag(), error);
                }
            }
        }
    }
    bm.stop();

    PRINT("Setup = %9E +/- %9E", bm[0], bm.getError());
    PRINT("Solve = %9E +/- %9E", bm[1], bm.getError());
    PRINT("Total = %9E +/- %9E", bm.getTotal(), bm.getError());
    
    E(SlepcFinalize());

    return ierr;
}