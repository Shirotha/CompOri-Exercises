#include "../common/bm.hpp"
#include "../common/la.hpp"

std::function<PetscScalar(PetscReal)> harmonic(PetscReal omega)
{
    double c = 0.5 * omega * omega;
    return [c](PetscReal x) { return c * x * x; };
}

std::shared_ptr<la::EigenSolver> solve_schrödinger(bm::BM& bm, const PetscInt n, const PetscReal range, const std::function<PetscScalar(PetscReal)>&& potential)
{
    PRINT("Range = %5E", range);

    auto grid = std::make_shared<la::Grid>(DM_BOUNDARY_GHOSTED, n, 1, 1, -range, range);
    auto H = std::make_shared<la::Matrix>(grid);
    PRINT("Step = %5E", grid->step);
    
    H->apply_cfd<2, 2>(ADD_VALUES, -0.5);
    H->apply_diagonal(ADD_VALUES, std::move(potential));

    H->assemble(MAT_FLUSH_ASSEMBLY);


    H->assemble(MAT_FINAL_ASSEMBLY);
    //MatScale(H->mat, omega * grid->step * grid->step * 0.25);

    auto solver = std::make_shared<la::EigenSolver>(H, EPS_HEP);

    bm.lap();
    
    solver->solve();

    bm.lap();

    return solver;
}

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
        PetscInt n=199;
        E(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
        PRINT("Matrix Size = %d", n);

        PetscReal omega = 4.0;
        auto solver = solve_schrödinger(bm, n, 4.0 / sqrt(omega), harmonic(omega));

        PRINT("Iterations = %i", solver->getNumIterations());
        //PRINT("Type = %i", solver->getType());
        //PRINT("Result Dimension = %i", solver->getDimension());
        
        /*{
            auto [iter, tol] = solver->getTolerance();
            PRINT("Tolerance = %.4E, Max. Iterations = %i", iter, tol);
        }*/

        {
            int i, n = solver->getNumEigenpairs();
            PRINT("Converged Eigenvalues = %i", n);

            if (n > 0)
            {
                PetscScalar ev;
                PetscReal error;
                auto x = std::make_shared<la::Vector>(solver->matrix->grid);
                
                for (i = 0; i < n; ++i)
                {
                    solver->getEigenpair(i, ev, *x, error);
                    PRINT("%9E + %9Ei +/- %12E", ev.real(), ev.imag(), error);
                    /*
                    PetscScalar* vs = (PetscScalar*)malloc(sizeof(PetscScalar) * grid->size);
                    VecGetArray(x->vec, &vs);
                    for (int j = 0; j < grid->size; ++j)
                    {
                        PRINT("%9E", vs[j].real() * vs[j].real() + vs[j].imag() * vs[j].imag());
                    }
                    VecRestoreArray(x->vec, &vs);
                    free(vs);
                    */
                }
            }
        }
    }
    bm.stop();
    /*
    PRINT("Setup = %9E +/- %9E", bm[0], bm.getError());
    PRINT("Solve = %9E +/- %9E", bm[1], bm.getError());
    PRINT("Total = %9E +/- %9E", bm.getTotal(), bm.getError());
    */
    E(SlepcFinalize());

    return ierr;
}