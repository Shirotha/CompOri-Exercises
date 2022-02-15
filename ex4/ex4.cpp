#include <string>
#include <sstream>
#include <sciplot/sciplot.hpp>
namespace plt = sciplot;
#include <petsc.h>
#include <petsctao.h>

#define E(X) ierr = X; CHKERRQ(ierr);
#define SQ(X) ((X) * (X))

struct PetscScalar2
{
    PetscScalar x;
    PetscScalar y;

    static PetscScalar2 create(const PetscScalar x, const PetscScalar y)
    {
        return { x, y };
    }

    constexpr PetscScalar norm2() const
    {
        return x * x + y * y;
    }

    constexpr PetscScalar2 operator+ (const PetscScalar2& rhs) const
    {
        return {x + rhs.x, y + rhs.y};
    }
    constexpr PetscScalar2 operator- (const PetscScalar2& rhs) const
    {
        return {x - rhs.x, y - rhs.y};
    }
    constexpr PetscScalar2 operator* (const PetscScalar& rhs) const
    {
        return {x * rhs, y * rhs};
    }
    constexpr PetscScalar2 operator/ (const PetscScalar& rhs) const
    {
        return {x / rhs, y / rhs};
    }
};
constexpr PetscScalar2 operator* (const PetscScalar& lhs, const PetscScalar2& rhs)
{
    return { lhs * rhs.x, lhs * rhs.y };
}

constexpr PetscScalar2 min(const PetscScalar2& a, const PetscScalar2& b)
{
    return {
        PetscRealPart(a.x - b.x) > 0 ? b.x : a.x,
        PetscRealPart(a.y - b.y) > 0 ? b.y : a.y
    };
}

constexpr PetscScalar2 max(const PetscScalar2& a, const PetscScalar2& b)
{
    return {
        PetscRealPart(a.x - b.x) < 0 ? b.x : a.x,
        PetscRealPart(a.y - b.y) < 0 ? b.y : a.y
    };
}

constexpr PetscScalar2 lerp(const PetscScalar2 a, const PetscScalar2 b, PetscReal x)
{
    return a + (b - a) * x;
}

struct TAOContext
{
    virtual PetscErrorCode calc(const void* coords, PetscReal* f, PetscScalar* grad) const = 0;
};

PetscErrorCode TAO1ObjectiveAndGradientRountine(Tao tao, Vec coords, PetscReal* f, Vec grad, void* ptr)
{
    PetscErrorCode ierr = 0;

    TAOContext* ctx = (TAOContext*)ptr;

    const PetscScalar* coordsData;
    PetscScalar* gradData;

    PetscFunctionBegin;

    E(VecGetArrayRead(coords, &coordsData))
    E(VecGetArray(grad, &gradData))

    E(ctx->calc(coordsData, f, gradData))

    E(VecRestoreArray(grad, &gradData))
    E(VecRestoreArrayRead(coords, &coordsData))

    PetscFunctionReturn(ierr);
}

struct LagrangeLerpContext : public TAOContext
{
    PetscInt knots = 10;
    PetscScalar2 begin = {0.0, 2.0};
    PetscScalar2 end = {10.0, 0.0};
    PetscReal mass = 1.0;

    LagrangeLerpContext()
    {
        PetscOptionsGetInt(NULL, NULL, "-knots", &knots, NULL);
        PetscOptionsGetReal(NULL, NULL, "-mass", &mass, NULL);

        PetscInt max = 2;
        PetscOptionsGetScalarArray(NULL, NULL, "-begin", &begin.x, &max, NULL);
        max = 2;
        PetscOptionsGetScalarArray(NULL, NULL, "-end", &end.x, &max, NULL);
    }

    virtual PetscScalar2 min() const
    {
        return ::min(begin, end) - PetscScalar2::create(10, 10);
    }

    virtual PetscScalar2 max() const
    {
        return ::max(begin, end) + PetscScalar2::create(10, 10);
    }

    PetscScalar2 get(const void* coords, const PetscInt i) const
    {
        const PetscScalar2 zero = { 0, 0 };
        const PetscScalar* x = (const PetscScalar*)coords;
        if (i < 0)
            return i == -1 ? begin : zero;
        if (i >= knots)
            return i == knots ? end : zero;
        
        return { x[2 * i], x[2 * i + 1] };
    }

    void set(PetscScalar* grad, const PetscInt i, const PetscScalar2 value) const
    {
        if (i < 0 || i >= knots)
            return;
            
        grad[2 * i] = value.x;
        grad[2 * i + 1] = value.y;
    }

    void add(PetscScalar* grad, const PetscInt i, const PetscScalar2 value) const
    {
        if (i < 0 || i >= knots)
            return;
            
        grad[2 * i] += value.x;
        grad[2 * i + 1] += value.y;
    }

    virtual PetscErrorCode calc(const void* coords, PetscReal* f, PetscScalar* grad) const
    {
        PetscErrorCode ierr = 0;

        *f = 0;
        for (int i = -1; i <= knots; ++i)
        {
            *f += PetscRealPart((get(coords, i + 1) - get(coords, i)).norm2());
        }
        *f *= 0.5 * SQ(knots + 1) * mass;

        for (int i = 0; i < knots; ++i)
            set(grad, i, SQ(knots + 1) * mass * (2 * get(coords, i) - get(coords, i + 1) - get(coords, i - 1)));

        return ierr;
    }
};

struct LagrangeLerpGravityContext : public LagrangeLerpContext
{
    PetscReal gravity = 9.81;

    LagrangeLerpGravityContext()
    {
        PetscOptionsGetReal(NULL, NULL, "-gravity", &gravity, NULL);
    }

    PetscErrorCode calc(const void* coords, PetscReal* f, PetscScalar* grad) const
    {
        PetscErrorCode ierr = 0;

        LagrangeLerpContext::calc(coords, f, grad);

        PetscReal s = 0;
        for (int i = 0; i <= knots; ++i)
            s += PetscRealPart(get(coords, i).y + get(coords, i + 1).y);

        *f += 0.5 * this->mass * gravity * s;
        
        for (int i = 0; i < knots - 1; ++i)
            add(grad, i, { 0, this->mass * gravity });

        return ierr;
    }
};

PetscErrorCode plot_curve(const LagrangeLerpContext& ctx, Vec coords)
{
    PetscErrorCode ierr = 0;

    plt::Plot plot;

    plot.size(1300, 650);

    plot.xlabel("x");
    plot.ylabel("y");

    plot.legend().show(false);

    plt::Vec xs(ctx.knots + 2);
    plt::Vec ys(xs.size());

    {
        const PetscScalar* x;

        E(VecGetArrayRead(coords, &x))

        PetscScalar2 a;
        for (PetscInt i = -1; i <= ctx.knots; ++i)
        {
            a = ctx.get(x, i);
            xs[i + 1] = PetscRealPart(a.x);
            ys[i + 1] = PetscRealPart(a.y);
        }

        E(VecRestoreArrayRead(coords, &x))
    }

    plot.drawCurveWithPoints(xs, ys);

    plot.show();

    return ierr;
}

PetscErrorCode run_tao(LagrangeLerpContext& ctx)
{
    PetscErrorCode ierr = 0;

    Tao tao;
    Vec coords, min, max;

    {
        E(VecCreate(PETSC_COMM_WORLD, &coords))
        E(VecSetType(coords, VECSEQ))
        E(VecSetSizes(coords, PETSC_DECIDE, 2 * ctx.knots))
        E(VecAssemblyBegin(coords))
        E(VecAssemblyEnd(coords))

        E(VecDuplicate(coords, &min))
        E(VecDuplicate(coords, &max))

        {
            PetscScalar* data, *minData, *maxData;

            E(VecGetArray(coords, &data))
            E(VecGetArray(min, &minData))
            E(VecGetArray(max, &maxData))

            PetscScalar2 knot, a = ctx.min(), b = ctx.max();
            for (int i = 0; i < ctx.knots; ++i)
            {
                knot = lerp(ctx.begin, ctx.end, (i + 1) / (ctx.knots + 1.0));
                data[2 * i] = knot.x;
                data[2 * i + 1] = a.x;
                minData[2 * i] = a.x;
                minData[2 * i + 1] = a.y;
                maxData[2 * i] = b.x;
                maxData[2 * i + 1] = b.y;
            }

            E(VecRestoreArray(max, &maxData))
            E(VecRestoreArray(min, &minData))
            E(VecRestoreArray(coords, &data))
        }

        E(TaoCreate(PETSC_COMM_WORLD, &tao))
        E(TaoSetType(tao, TAOLMVM))
        E(TaoSetVariableBounds(tao, min, max))
        E(TaoSetInitialVector(tao, coords))
        E(TaoSetObjectiveAndGradientRoutine(tao, TAO1ObjectiveAndGradientRountine, &ctx))

        PetscReal norm = 1e-10, relNorm = 1e-10, progNorm = 1e-10;
        PetscInt iterations = 10000;

        E(TaoSetTolerances(tao, norm, relNorm, progNorm))        
        E(TaoSetMaximumIterations(tao, iterations))
        E(TaoSetMaximumFunctionEvaluations(tao, iterations))
        
        E(TaoSetFromOptions(tao))

        E(TaoSolve(tao))

        PetscBool dump_points = PETSC_FALSE;        
        E(PetscOptionsGetBool(NULL, NULL, "-dump_points", &dump_points, NULL))
        if (dump_points)
        {
            const PetscScalar* data;

            E(VecGetArrayRead(coords, &data))

            PetscScalar2 x;
            for (PetscInt i = -2; i <= ctx.knots + 1; ++i)
            {
                x = ctx.get(data, i);
                E(PetscPrintf(PETSC_COMM_WORLD, "(%5E, %5E)\n", x.x, x.y))
            }

            E(VecRestoreArrayRead(coords, &data))
        }

        E(plot_curve(ctx, coords))
    }
    E(VecDestroy(&coords))
    E(TaoDestroy(&tao))

    return ierr;
}

int main(int argv, char** argc)
{
    PetscErrorCode ierr = 0;
    E(PetscInitialize(&argv, &argc, NULL, NULL))

    LagrangeLerpGravityContext ctx;

    E(run_tao(ctx))

    E(PetscFinalize())

    return ierr;
}