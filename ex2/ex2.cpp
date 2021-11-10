#include <sstream>
#include <limits>
#include <sciplot/sciplot.hpp>
namespace plt = sciplot;

#include "../common/bm.hpp"
#include "../common/la.hpp"

std::shared_ptr<la::EigenSolver> solve_schrödinger1d(bm::BM& bm, const PetscInt n, const PetscInt dimension, const la::SpecializedPotential& potential)
{
    PRINT("Range = %5E..%5E", potential.range.first.x, potential.range.second.x);
    
    auto grid = std::make_shared<la::Grid>(DM_BOUNDARY_GHOSTED, n, 1, 1, potential.range.first.x, potential.range.second.x);
    auto H = std::make_shared<la::Matrix>(grid);
    PRINT("Step = %5E", grid->step.x);
    
    H->apply_cfd<2, 2>(ADD_VALUES, -0.5);
    H->apply_diagonal(ADD_VALUES, std::move(potential.potential));

    H->assemble(MAT_FINAL_ASSEMBLY);

    auto solver = std::make_shared<la::EigenSolver>(H, EPS_HEP, dimension);

    bm.lap();
    
    solver->solve();

    bm.lap();

    return solver;
}

template<typename T>
void plot_eigenvectors(const std::shared_ptr<la::EigenSolver> solver, const T& potential, bool logScale=false)
{
    int i, j, n = solver->getNumEigenpairs();
    PRINT("Converged Eigenvalues = %i", n);

    if (n > 0)
    {
        plt::Plot plot;
        plot.size(1300, 650);

        plot.xlabel("x");
        plot.ylabel("E - V_{min}");

        plot.xrange(potential.range.first.x, potential.range.second.x);
        
        plot.legend()
            .atOutsideBottom()
            .displayHorizontal()
            .displayHorizontalMaxCols(2);
        
        auto points = solver->matrix->grid->getPoints();
        plt::Vec xs(solver->matrix->grid->size);
        for (i = 0; i < xs.size(); ++i)
            xs[i] = points[i].real();

        std::stringstream label;
        label.setf(label.scientific, label.floatfield);

        EPSConvergedReason reason;
        PetscScalar ev;
        PetscReal error;
        auto x = std::make_shared<la::Vector>(solver->matrix->grid);
        plt::Vec pot(solver->matrix->grid->size);
        double ymin = std::numeric_limits<double>::max(), ymax = 0, tmp, max, pmin = std::numeric_limits<double>::max();
        for (i = 0; i < xs.size(); ++i)
        {
            tmp = potential.potential(xs[i]).real();
            /*
            if (tmp > ymax)
                ymax = tmp;
            if (tmp < ymin)
                ymin = tmp;
            */
           if (tmp < pmin)
            pmin = tmp;
            pot[i] = tmp;
        }
        for (i = 0; i < pot.size(); ++i)
            pot[i] -= pmin;

        if (logScale)
            for (i = 0; i < pot.size(); ++i)
                pot[i] = log10(pot[i]);


        plot.drawCurveFilled(xs, pot).labelNone().above().fillIntensity(0.3);

        plt::Vec ys(solver->matrix->grid->size);
        for (i = 0; i < n; ++i)
        {
            solver->getEigenpair(i, ev, *x, error);
            auto vec = x->getArray();
            max = 0;
            for (j = 0; j < ys.size(); ++j)
            {
                tmp = pow(std::__complex_abs(vec[j]), 2);
                //tmp = vec[j].real();
                if (tmp > max)
                    max = tmp;
                ys[j] = tmp;
            }
            max = 1.0 / max;
            for (j = 0; j < ys.size(); ++j)
            {
                tmp = ys[j] * max + ev.real() - pmin;
                if (logScale)
                    tmp = log10(tmp);
                if (tmp > ymax)
                    ymax = tmp;
                if (tmp < ymin)
                    ymin = tmp;
                ys[j] = tmp;
            }

            label << ev.real() << " + " << ev.imag() << "i +/- " << error;

            E(EPSGetConvergedReason(solver->eps, &reason));
            PRINT("%9E + %9Ei +/- %9E (%i)", ev.real(), ev.imag(), error, reason);
            plot.drawCurve(xs, ys).label(label.str());

            label.str("");
        }
        
        plot.yrange(ymin - 0.01 * (ymax - ymin), ymax);
        plot.show();
    }
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
        PetscInt n=200;
        E(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
        PRINT("Matrix Size = %d", n);
        //if (0)
        {
            auto potential = la::harmonic(1.0);
            
            auto solver = solve_schrödinger1d(bm, n, 4, potential);
            PRINT("Iterations = %i", solver->getNumIterations());
            plot_eigenvectors(solver, potential);
        }
        //if (0)
        {
            auto potential = la::coulomb(8.0);

            auto solver = solve_schrödinger1d(bm, n, 8, potential);
            PRINT("Iterations = %i", solver->getNumIterations());
            plot_eigenvectors(solver, potential/*, true*/);
        }
        //if (0)
        {
            auto potential = la::morse(2.0, 4.0);

            auto solver = solve_schrödinger1d(bm, n, 8, potential);
            plot_eigenvectors(solver, potential);
        }
        //if (0)
        {
            // TODO: find better parameters
            auto potential = la::doubleWell(1.1, 0.2);

            auto solver = solve_schrödinger1d(bm, n, 4, potential);
            plot_eigenvectors(solver, potential);
        }
    }
    bm.stop();
    /*
    PRINT("Setup = %9E +/- %9E", bm[0], bm.getError());
    PRINT("Solve = %9E +/- %9E", bm[1], bm.getError());
    */
    PRINT("Total = %9E +/- %9E", bm.getTotal(), bm.getError());
    E(SlepcFinalize());

    return ierr;
}