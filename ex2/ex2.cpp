#include <string>
#include <sstream>
#include <limits>
#include <sciplot/sciplot.hpp>
namespace plt = sciplot;

#include "../common/bm.hpp"
#include "../common/la.hpp"

/*
 * Analytical solutions in N dimensions
 * E_n = omega * (n + N/2)
 */
struct Harmonic : la::PotentialFamily<la::Potential1d, PetscReal>
{
    la::Potential1d get(PetscReal omega)
    {
        const double c = 0.5 * omega * omega;
        return [c](PetscReal x) 
        { 
            return c * x * x; 
        };
    }
    la::Interval range(PetscReal omega)
    {
        constexpr PetscReal threshold = 4.0;
        PetscReal x = threshold / sqrt(omega);
        return { -x, x };
    }
} harmonic;
/*
 * Analytical solutions
 * E_n = -amp^2/2 1/n^2
 */
struct Coulomb : la::PotentialFamily<la::Potential1d, PetscReal>
{
    la::Potential1d get(PetscReal amp)
    {
        auto r = range(amp);
        PetscReal min = -200.0 / (r.second.x - r.first.x);
        return [amp, min](PetscReal x) 
        { 
            PetscReal y = -amp / abs(x);
            if (y < min)
                return min;
            return y;
        };
    }
    la::Interval range(PetscReal amp)
    {
        constexpr PetscReal threshold = 1e-1;
        PetscReal x = sqrt(1/threshold * amp);
        return { -x, x };
    }
} coulomb;

struct Morse : la::PotentialFamily<la::Potential1d, PetscReal, PetscReal>
{
    la::Potential1d get(PetscReal r0, PetscReal depth)
    {
        PetscReal a = 1.0 / sqrt(2 * depth);
        return [r0, depth, a](PetscReal x)
        {
            PetscReal e = 1 - exp(-a * (x - r0));
            return depth * e * e;
        };
    }
    la::Interval range(PetscReal r0, PetscReal depth)
    {
        constexpr PetscReal threshold = 0.99;
        return { 0.0, sqrt(2 * depth) * log(1.0 / (1 - sqrt(threshold))) };
    }
} morse;

struct DoubleWell : la::PotentialFamily<la::Potential1d, PetscReal, PetscReal>
{
    la::Potential1d get(PetscReal h, PetscReal c)
    {
        PetscReal h4_4 = 0.25 * h * h * h * h, c2_2 = 0.5 * c * c;
        return [h4_4, c2_2](PetscReal x)
        {
            PetscReal x2 = x * x;
            return c2_2 * x2 * x2 - h4_4 * x2;
        };
    }
    la::Interval range(PetscReal h, PetscReal c)
    {
        constexpr PetscReal threshold = 0.6;
        PetscReal x = threshold * sqrt(1 + sqrt(3)) * h * h / c;
        return { -x, x };
    }
} doubleWell;

std::shared_ptr<la::EigenSolver> solve_schrödinger(const std::string& name, bm::BM& bm, const PetscInt n, const PetscInt dimension, const la::SpecializedPotential<la::Potential1d>& potential)
{
    PRINT("== %s ==", name.c_str());
    PRINT("Matrix Size = %i", n);
    PRINT("Range = %5E..%5E", potential.range.first.x, potential.range.second.x);
    
    auto bnd = la::option("-boundary", DM_BOUNDARY_GHOSTED);

    auto grid = std::make_shared<la::Grid>(la::option("-boundaryX", bnd), n, 1, 1, potential.range.first.x, potential.range.second.x);
    auto H = std::make_shared<la::Matrix>(grid);
    PRINT("Step = %5E", grid->step.x);
    
    H->apply_cfd<2, 2>(ADD_VALUES, -0.5);
    H->apply_diagonal(ADD_VALUES, std::move(potential.potential));

    H->assemble(MAT_FINAL_ASSEMBLY);

    auto solver = std::make_shared<la::EigenSolver>(H, EPS_HEP, dimension);

    bm.lap();
    
    solver->solve();

    bm.lap();

    PRINT("Iterations = %i", solver->getNumIterations());

    return solver;
}

std::shared_ptr<la::EigenSolver> solve_schrödinger(const std::string& name, bm::BM& bm, const PetscInt n, const PetscInt m, const PetscInt dimension, const la::SpecializedPotential<la::Potential2d>& potential)
{
    PRINT("== %s ==", name.c_str());
    PRINT("Matrix Size = %i (%i, %i)", n * m, n, m);
    PRINT("X Range = %5E..%5E", potential.range.first.x, potential.range.second.x);
    PRINT("Y Range = %5E..%5E", potential.range.first.y, potential.range.second.y);
    
    auto bnd = la::option("-boundary", DM_BOUNDARY_GHOSTED);

    auto grid = std::make_shared<la::Grid>(la::option("-boundaryX", bnd), la::option("-boundaryY", bnd), n, m, 1, 1, potential.range.first.x, potential.range.second.x, potential.range.first.y, potential.range.second.y);
    
    auto H = std::make_shared<la::Matrix>(grid);
    PRINT("X Step = %5E", grid->step.x);
    PRINT("Y Step = %5E", grid->step.y);
    
    H->apply_cfd<2, 2, X>(ADD_VALUES, -0.5);
    H->apply_cfd<2, 2, Y>(ADD_VALUES, -0.5);
    H->apply_diagonal(ADD_VALUES, std::move(potential.potential));
    
    H->assemble(MAT_FINAL_ASSEMBLY);

    auto solver = std::make_shared<la::EigenSolver>(H, EPS_HEP, dimension);

    bm.lap();
    
    solver->solve();

    bm.lap();

    PRINT("Iterations = %i", solver->getNumIterations());

    return solver;
}

std::shared_ptr<la::EigenSolver> solve_schrödinger(const std::string& name, bm::BM& bm, const PetscInt n, const PetscInt m, const PetscInt o, const PetscInt dimension, const la::SpecializedPotential<la::Potential3d>& potential)
{
    PRINT("== %s ==", name.c_str());
    PRINT("Matrix Size = %i (%i, %i, %i)", n * m * o, n, m, o);
    PRINT("X Range = %5E..%5E", potential.range.first.x, potential.range.second.x);
    PRINT("Y Range = %5E..%5E", potential.range.first.y, potential.range.second.y);
    PRINT("Z Range = %5E..%5E", potential.range.first.z, potential.range.second.z);
    
    auto bnd = la::option("-boundary", DM_BOUNDARY_GHOSTED);

    auto grid = std::make_shared<la::Grid>(la::option("-boundaryX", bnd), la::option("-boundaryY", bnd), la::option("-boundaryZ", bnd), n, m, o, 1, 1, potential.range.first.x, potential.range.second.x, potential.range.first.y, potential.range.second.y, potential.range.first.z, potential.range.second.z);
    
    auto H = std::make_shared<la::Matrix>(grid);
    PRINT("X Step = %5E", grid->step.x);
    PRINT("Y Step = %5E", grid->step.y);
    PRINT("Z Step = %5E", grid->step.z);

    H->apply_cfd<2, 2, X>(ADD_VALUES, -0.5);
    H->apply_cfd<2, 2, Y>(ADD_VALUES, -0.5);
    H->apply_cfd<2, 2, Z>(ADD_VALUES, -0.5);
    H->apply_diagonal(ADD_VALUES, std::move(potential.potential));
    
    H->assemble(MAT_FINAL_ASSEMBLY);

    auto solver = std::make_shared<la::EigenSolver>(H, EPS_HEP, dimension);

    bm.lap();
    
    solver->solve();

    bm.lap();

    PRINT("Iterations = %i", solver->getNumIterations());

    return solver;
}

void dump_eigenvalues(const std::string& name, const std::shared_ptr<la::EigenSolver> solver)
{
    int n = solver->getNumEigenpairs();
    PRINT("Converged Eigenvalues = %i", n);

    PetscScalar ev;
    auto x = std::make_shared<la::Vector>(solver->matrix->grid);
    PetscReal error;
    for (int i = 0; i < n; ++i)
    {
        solver->getEigenpair(i, ev, *x, error);
        PRINT("%9E + %9Ei +/- %9E", ev.real(), ev.imag(), error);
    }
}

void plot_eigenvectors(const std::string& name, const std::shared_ptr<la::EigenSolver> solver, const la::SpecializedPotential<la::Potential1d>& potential, bool logScale=false)
{
    size_t i, j, n = solver->getNumEigenpairs();

    if (n > 0)
    {
        plt::Plot plot;
        plot.size(1300, 650);
        // TODO: set window title to name
        plot.xlabel("x");
        plot.ylabel("E - V_{min}");

        plot.xrange(potential.range.first.x, potential.range.second.x);
        
        plot.legend()
            .atOutsideBottom()
            .displayHorizontal()
            .displayHorizontalMaxCols(2);
        
        auto ptr = solver->matrix->grid->getPoints();
        PetscScalar* points = (PetscScalar*)ptr.get();
        plt::Vec xs(solver->matrix->grid->size);
        for (i = 0; i < xs.size(); ++i)
            xs[i] = points[i].real();


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
            plot.ytics().logscale();

        plot.drawCurveFilled(xs, pot).labelNone().above().fillIntensity(0.3);

        std::stringstream label;
        label.setf(label.scientific, label.floatfield);

        PetscScalar ev;
        auto x = std::make_shared<la::Vector>(solver->matrix->grid);
        PetscReal error;
        plt::Vec ys(solver->matrix->grid->size);
        for (i = 0; i < n; ++i)
        {
            solver->getEigenpair(i, ev, *x, error);
            auto vec = x->getArray();
            max = 0;
            for (j = 0; j < ys.size(); ++j)
            {
                tmp = pow(std::abs(vec[j]), 2);
                //tmp = vec[j].real();
                if (tmp > max)
                    max = tmp;
                ys[j] = tmp;
            }
            max = 1.0 / max;
            for (j = 0; j < ys.size(); ++j)
            {
                tmp = ys[j] * max + ev.real() - pmin;
                if (tmp > ymax)
                    ymax = tmp;
                if (tmp < ymin)
                    ymin = tmp;
                ys[j] = tmp;
            }

            label << ev.real() << " + " << ev.imag() << "i +/- " << error;
            plot.drawCurve(xs, ys).label(label.str());
            label.str("");
        }
        
        plot.yrange(ymin - 0.01 * (ymax - ymin), ymax);
        
        std::vector<std::vector<plt::PlotVariant>> plots(1);
        plots[0].push_back(plot);
        plt::Figure fig(plots);
        fig.size(1300, 650);
        fig.title(name);
        fig.show();
    }
}

void plot_eigenvectors(const std::string& name, const std::shared_ptr<la::EigenSolver> solver, const la::SpecializedPotential<la::Potential2d>& potential, bool logScale=false)
{
    size_t i, j, n = solver->getNumEigenpairs();
    size_t rows = sqrt(n);
    if (n > 0)
    {
        std::vector<std::vector<plt::PlotVariant>> plots(rows);

        auto ptr = solver->matrix->grid->getPoints();
        PetscScalar** points = (PetscScalar**)ptr.get();

        size_t mx = solver->matrix->grid->size.x, my = solver->matrix->grid->size.y;
        plt::Vec xs(mx * my);
        plt::Vec ys(xs.size());
        for (j = 0; j < my; ++j)
            for (i = 0; i < mx; ++i)
            {
                xs[j * mx + i] = points[j][2 * i].real();
                ys[j * mx + i] = points[j][2 * i + 1].real();
            }
            
        PetscReal zmin = std::numeric_limits<double>::max(), zmax = 0, max, pmin = std::numeric_limits<double>::max(), tmp;
        plt::Vec pot(xs.size());
        for (i = 0; i < xs.size(); ++i)
        {
            tmp = potential.potential(xs[i], ys[i]).real();
            if (tmp < pmin)
                pmin = tmp;
            pot[i] = tmp;
        }
        for (i = 0; i < pot.size(); ++i)
            pot[i] -= pmin;
        std::stringstream label;
        label.setf(label.scientific, label.floatfield);

        PetscScalar ev;
        auto x = std::make_shared<la::Vector>(solver->matrix->grid);
        PetscReal error;
        plt::Vec zs(xs.size());
        for (i = 0; i < n; ++i)
        {
            plt::Plot3D plot;
            plot.size(1300, 650);

            plot.xlabel("x");
            plot.ylabel("y");
            plot.zlabel("E - V_{min}");

            plot.xrange(potential.range.first.x, potential.range.second.x);
            plot.yrange(potential.range.first.y, potential.range.second.y);

            plot.legend()
                .atOutsideBottom()
                .displayHorizontal()
                .displayHorizontalMaxCols(2);

            plot.gnuplot("set hidden3d");

            label << "set dgrid3d " << mx << "," << my << " qnorm 2";
            plot.gnuplot(label.str());
            label.str("");

            //plot.drawCurve(xs, ys, pot).labelNone();

            solver->getEigenpair(i, ev, *x, error);
            auto vec = x->getArray();
            max = 0;
            for (j = 0; j < zs.size(); ++j)
            {
                tmp = std::pow(std::abs(vec[j]), 2);
                if (tmp > max)
                    max = tmp;
                zs[j] = tmp;
            }
            max = 1.0 / max;
            for (j = 0; j < zs.size(); ++j)
            {
                tmp = zs[j] * max /*+ ev.real() - pmin*/;
                if (logScale)
                    tmp = log10(tmp);
                if (tmp > zmax)
                    zmax = tmp;
                if (tmp < zmin)
                    zmin = tmp;
                zs[j] = tmp;
            }

            label << ev.real() << " + " << ev.imag() << "i +/- " << error;
            plot.drawCurve(xs, ys, zs).label(label.str());
            label.str("");

            plot.zrange(0.0, 1.0);

            plots[i / rows].push_back(plot);
        }

        plt::Figure fig(plots);
        fig.size(1300, 650);
        fig.title(name);
        fig.show();
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
        /*
        PetscInt test = 1;
        E(PetscOptionsGetInt(NULL, "pre", "-test", &test, NULL));
        PRINT("")
        */
        auto n1 = la::option("-n1", 200);
        auto n2 = la::option("-n2", 20);
        auto n3 = la::option("-n3", 8);
        auto do1 = la::option("-d1", PETSC_FALSE);
        auto do2 = la::option("-d2", PETSC_FALSE);
        auto do3 = la::option("-d3", PETSC_FALSE);
        auto plot = la::option("-plot", PETSC_FALSE);
        auto plot1 = la::option("-plot1", plot);
        auto plot2 = la::option("-plot2", plot);

        if (la::option("-harm", do1))
        {
            const std::string name("Harmonic");

            auto n = la::option("-harm_n", n1);
            auto d = la::option("-harm_d", 4);
            auto omega = la::option("-harm_omega", 1.0);

            auto potential = harmonic(omega);
            auto solver = solve_schrödinger(name, bm, n, d, potential);
            dump_eigenvalues(name, solver);
            if (la::option("-harm_plot", plot1))
                plot_eigenvectors(name, solver, potential);
        }
        if (la::option("-h", do1))
        {
            const std::string name("H-Atom");

            auto n = la::option("-h_n", n1);
            auto d = la::option("-h_d", 8);
            auto amp = la::option("-h_amp", 2.65);

            auto potential = coulomb(amp);
            auto solver = solve_schrödinger(name, bm, n, d, potential);
            dump_eigenvalues(name, solver);
            if (la::option("-h_plot", plot1))
                plot_eigenvectors(name, solver, potential, la::option("-h_log", PETSC_FALSE));
        }
        if (la::option("-morse", do1))
        {
            const std::string name("Morse");

            auto n = la::option("-morse_n", n1);
            auto d = la::option("-morse_d", 8);
            auto r0 = la::option("-morse_r0", 2.0);
            auto depth = la::option("-morse_depth", 4.0);

            auto potential = morse(r0, depth);
            auto solver = solve_schrödinger(name, bm, n, d, potential);
            dump_eigenvalues(name, solver);
            if (la::option("-morse_plot", plot1))
                plot_eigenvectors(name, solver, potential);
        }
        if (la::option("-dw", do1))
        {
            const std::string name("Double-Well");

            auto n = la::option("-dw_n", n1);
            auto d = la::option("-dw_d", 4);
            auto h = la::option("-dw_h", 1.1);
            auto c = la::option("-dw_c", 0.2);

            auto potential = doubleWell(h, c);
            auto solver = solve_schrödinger(name, bm, n, d, potential);
            dump_eigenvalues(name, solver);
            if (la::option("-dw_plot", plot1))
                plot_eigenvectors(name, solver, potential);
        }
        if (la::option("-harm2", do2))
        {
            const std::string name("2D-Harmonic");

            auto n = la::option("-harm2_n", n2);
            auto d = la::option("-harm2_d", 4);
            auto omega = la::option("-harm2_omega", 1.0);

            auto potential = la::radial(harmonic(omega));
            auto solver = solve_schrödinger(name, bm, n, n, d, potential);
            dump_eigenvalues(name, solver);
            if (la::option("-harm2_plot", plot2))
                plot_eigenvectors(name, solver, potential);
        }
        if (la::option("-h2", do2))
        {
            const std::string name("2D-H-Atom");

            auto n = la::option("-h2_n", n2);
            auto d = la::option("-h2_d", 4);
            auto amp = la::option("-h2_amp", 2.65);

            auto potential = la::radial(coulomb(amp));
            auto solver = solve_schrödinger(name, bm, n, n, d, potential);
            dump_eigenvalues(name, solver);
            if (la::option("-h2_plot", plot2))
                plot_eigenvectors(name, solver, potential);
        }
        if (la::option("-morse2", do2))
        {
            const std::string name("2D-Morse");

            auto n = la::option("-morse2_n", n2);
            auto d = la::option("-morse2_d", 4);
            auto r0 = la::option("-morse2_r0", 2.0);
            auto depth = la::option("-morse2_depth", 4.0);

            auto potential = la::radial(morse(r0, depth));
            auto solver = solve_schrödinger(name, bm, n, n, d, potential);
            dump_eigenvalues(name, solver);
            if (la::option("-morse2_plot", plot2))
                plot_eigenvectors(name, solver, potential);
        }
        if (la::option("-dw2", do2))
        {
            const std::string name("Double-Well");

            auto n = la::option("-dw2_n", n2);
            auto d = la::option("-dw2_d", 4);
            auto h = la::option("-dw2_h", 1.1);
            auto c = la::option("-dw2_c", 0.2);

            auto potential = la::radial(doubleWell(h, c));
            auto solver = solve_schrödinger(name, bm, n, n, d, potential);
            dump_eigenvalues(name, solver);
            if (la::option("-dw2_plot", plot2))
                plot_eigenvectors(name, solver, potential);
        }
        if (la::option("-harm3", do3))
        {
            const std::string name("3D-Harmonic");

            auto n = la::option("-harm3_n", n3);
            auto d = la::option("-harm3_d", 8);
            auto omega = la::option("-harm3_omega", 1.0);

            auto potential = la::spherical(harmonic(omega));
            auto solver = solve_schrödinger(name, bm, n, n, n, d, potential);
            dump_eigenvalues(name, solver);
        }
    }
    bm.stop();
    /*
    PRINT("Setup = %9E +/- %9E", bm[0], bm.getError());
    PRINT("Solve = %9E +/- %9E", bm[1], bm.getError());
    */
    PRINT("Total Time = %9E +/- %9E", bm.getTotal(), bm.getError());
    E(SlepcFinalize());

    return ierr;
}