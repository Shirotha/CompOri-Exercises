#include <string>
#include <sstream>
#include <sciplot/sciplot.hpp>
namespace plt = sciplot;
#include "../common/la.hpp"
#include "lagrange_lerp.hpp"


struct AppCtx : la::OptimizerContext
{
    AppCtx() : la::OptimizerContext(TAOIPM, 2, 1, 2)
    { 
        setup();
    }

    void initialize()
    {
        state.setValues(0.0);
        lowerBound.setValues(-1.0);
        upperBound.setValues(2.0);
    }

    void lateInitialize()
    {
        configureSolvers(KSPPREONLY, PCLU, MATSOLVERSUPERLU);
        //configureTolerance(1e-10, 1e-10, 1e-10, 1000);
    }

    // f(x, y) = (x - 2)^2 + (y - 2)^2 - 2(x + y)
    PetscScalar calcValue(la::ScalarAR x)
    {
        return SQ(x[0] - 2.0) + SQ(x[1] - 2.0) - 2.0 * (x[0] + x[1]);
    }

    // df = (2(x - 2) - 2) dx + (2(y - 2) - 2) dy
    void calcGradient(la::ScalarAR x, la::ScalarA g)
    {
        g[0] = 2.0 * (x[0] - 2.0) - 2.0;
        g[1] = 2.0 * (x[1] - 2.0) - 2.0;
    }
    
    void calcHessian(la::ScalarAR x, la::ScalarA h/*, ScalarA precon*/)
    {
        Vec dualEq;
        Vec dualIeq;

        E(TaoGetDualVariables(tao, &dualEq, &dualIeq));
        auto de = la::Vector::readArray(dualEq);
        auto di = la::Vector::readArray(dualIeq);

        h[0] = 2.0 * (1 + de[0] + di[0] - di[1]);
        h[3] = 2.0;
    }
    
    // x^2 + y = 2
    void calcEqs(la::ScalarAR x, la::ScalarA c)
    {
        c[0] = SQ(x[0]) + x[1] - 2.0;
    }

    // x^2 - 1 <= y <= x^2
    void calcIeqs(la::ScalarAR x, la::ScalarA c)
    {
        c[0] = SQ(x[0]) - x[1];
        c[1] = 1.0 - SQ(x[0]) + x[1];
    }

    void calcEqJacobian(la::ScalarAR x, la::ScalarA j/*, ScalarA precon*/)
    {
        j[0] = 2 * x[0];
        j[1] = 1.0;
    }

    void calcIeqJacobian(la::ScalarAR x, la::ScalarA j/*, ScalarA precon*/)
    {
        j[0] = 2.0 * x[0];
        j[1] = -1.0;
        j[2] = -2.0 * x[0];
        j[3] = 1.0;
    }
};

struct SimpleCtx : la::OptimizerContext
{
    SimpleCtx() : la::OptimizerContext(TAOLMVM, 1, 0, 0)
    {
        setup();
    }

    PetscScalar calcValue(la::ScalarAR x)
    {
        return SQ(x[0] - 1.0) - 1.0;
    }

    void calcGradient(la::ScalarAR x, la::ScalarA g)
    {
        g[0] = 2.0 * (x[0] - 1.0);
    }
};

struct LJCtx : la::OptimizerContext
{
    PetscReal a = 1.0;
    PetscReal b = 1.0;

    LJCtx() : la::OptimizerContext(TAOBLMVM, 1, 0, 0)
    {
        setup();
    }

    void initialize()
    {
        state.setValues(100.0);
        lowerBound.setValues(0.01);
        upperBound.setValues(100.0);
    }

    PetscScalar calcValue(la::ScalarAR x)
    {
        return (a / (x[0]) - b) / x[0];
    }

    void calcGradient(la::ScalarAR x, la::ScalarA g)
    {
        g[0] = (b - 2 * a / x[0]) / (x[0] * x[0]);
    }
};

template<int N>
struct LLGravityCtx : LagrangeLerpContext<N>
{
    const PetscReal gravity = 9.81;

    LLGravityCtx() : LagrangeLerpContext<N>(1.0, {0.0, 1.0}, {10.0, 0.0})
    { }

    PetscScalar calcPotential() 
    {
        PetscScalar s{}; 

        for (int i = 0; i < N; ++i)
        {
            s += this->ys[i] + this->ys[i + 1];
        }

        s *= 0.5 * this->mass * gravity;

        return s;
    }

    void calcPotentialGradient(la::ScalarA g)
    {
        
        for (int i = 0; i < N; ++i)
            g[i + N - 1] += this->mass * gravity;
        
    }
};

template<typename T>
void solve()
{
    T ctx;

    auto reason = ctx.solve();

    auto x = ctx.state.readArray();
    auto n = ctx.state.size();

    std::stringstream str;
    str.setf(str.scientific, str.floatfield);

    str << "f(";
    for (int i = 0; i < n.x; ++i)
    {
        str << x[i];
        if (i < n.x - 1)
            str << ", ";
    }

    str << ") = " << ctx.calcValue(x) << " (" << TaoConvergedReasons[reason] << ")";
    PRINT("%s", str.str().c_str());
}

template<typename T>
void plotL(T ctx)
{
    plt::Plot plot;
    plot.size(1200, 600);

    plot.xlabel("x");
    plot.ylabel("y");

    plot.legend().show(false);

    plt::Vec xs(ctx.xs, ctx.n + 1);
    plt::Vec ys(ctx.ys, ctx.n + 1);

    plot.drawCurveWithPoints(xs, ys);
    /*
    plt::Vec x2s = plt::linspace(0, 10, 100);
    plt::Vec y2s(x2s.size());
    for (size_t i = 0; i < y2s.size(); ++i)
        y2s[i] = 5 - 0.05 * SQ(x2s[i]);

    plot.drawCurve(x2s, y2s);
    */
    plot.show();
}

int main(int argc, char** argv)
{
    E(PetscInitialize(&argc, &argv, NULL, NULL));
    {
        LLGravityCtx<100> ctx;

        ctx.solve();

        plotL(ctx);
    }
    E(PetscFinalize()); 

    return ierr;
}