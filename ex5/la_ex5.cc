#include <string>
#include <sstream>
#include <sciplot/sciplot.hpp>
namespace plt = sciplot;
#include "../common/la.hpp"

struct FHNDiffCtx : la::DynamicsContext
{
    PetscScalar alpha = 0.0;
    PetscScalar gamma = 0.5;
    PetscScalar epsilon = 1e-2;
    PetscScalar iApp = 1.0;
    PetscScalar diff = 1.0;

    FHNDiffCtx() : la::DynamicsContext(
        std::make_shared<la::Grid>(DM_BOUNDARY_GHOSTED, 100, 2, 1, 0.0, 1.0), 
        TSRK, TS_NONLINEAR)
    {
        setup(10.0, 1e-2, TS_EXACTFINALTIME_STEPOVER);
    }

    void initialCondition(void* x)
    {
        PetscScalar** xs = (PetscScalar**)x;

        auto begin = this->state.grid->begin;
        auto end = this->state.grid->end;

        int iSpike = (begin.x + end.x) / 2;

        for (int i = begin.x; i < end.x; ++i)
        {
            xs[i][0] = exp(-0.1 * SQ(i - iSpike));
            xs[i][1] = 0;
        }
    }

    void calcRHS(PetscReal time, la::GridPointsR x, la::GridPoints y)
    {
        const PetscScalar** xs = (const PetscScalar**)x.get();
        PetscScalar** ys = (PetscScalar**)y.get();

        auto begin = this->state.grid->begin;
        auto end = this->state.grid->end;
        auto step = this->state.grid->step;
        for (int i = begin.x; i < end.x; ++i)
        {
            ys[i][0] = (xs[i][0] * (1 - xs[i][0]) * (xs[i][0] - alpha) - xs[i][1] + iApp + 
                la::Vector::cfd_coeff<1, 2, 2>(x, i, step, 0, diff)) / epsilon;
            ys[i][1] = xs[i][0] - gamma * xs[i][1];
        }
    }

    virtual void monitor(PetscInt step, PetscReal time, la::GridPointsR x)
    {
        const PetscScalar** xs = (const PetscScalar**)x.get();

        PRINT("v(0.5, %9E) = %9E", time, xs[50][0]);
    }
};

void plotSpacialDOF(la::Vector state, size_t dof)
{
    PetscScalar* x = (PetscScalar*)state.grid->getPoints().get();
    const PetscScalar** y = (const PetscScalar**)state.readArrayGlobal().get();

    plt::Plot plot;
    plot.size(1200, 600);

    plot.xlabel("x");
    plot.ylabel("y");

    size_t i, j;

    plt::Vec xs(state.grid->size.x);
    for (i = 0; i < xs.size(); ++i)
        xs[i] = PetscRealPart(x[i]);

    std::stringstream label;
    label.setf(label.scientific, label.floatfield);

    plt::Vec ys(xs.size());
    for (j = 0; j < dof; ++j)
    {
        for (i = 0; i < ys.size(); ++i)
            ys[i] = PetscRealPart(y[i][j]);

        label << "DOF " << (j + 1);
        plot.drawCurve(xs, ys).label(label.str());
        label.str("");
    }

    plot.show();
}

int main(int argc, char** argv)
{
    E(PetscInitialize(&argc, &argv, NULL, NULL));
    {
        FHNDiffCtx ctx;

        plotSpacialDOF(ctx.state, 1);

        ctx.solve();

        plotSpacialDOF(ctx.state, 1);
    }
    E(PetscFinalize()); 

    return ierr;
}