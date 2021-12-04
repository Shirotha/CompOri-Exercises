#include "../common/la.hpp"


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

    // x^2 <= y <= x^2 - 1 ??
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

int main(int argc, char** argv)
{
    E(PetscInitialize(&argc, &argv, NULL, NULL));
    {
        AppCtx ctx;
        
        auto reason = ctx.solve();

        auto x = ctx.state.readArray();
        PRINT("f(%9E, %9E) = %9E (%s)", x[0], x[1], ctx.calcValue(x), TaoConvergedReasons[reason]);        
    }
    {
        SimpleCtx ctx;
        
        auto reason = ctx.solve();

        auto x = ctx.state.readArray();
        PRINT("f(%9E) = %9E (%s)", x[0], ctx.calcValue(x), TaoConvergedReasons[reason]);   
    }
    E(PetscFinalize());

    return ierr;
}