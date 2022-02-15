#pragma once

#include <tuple>
#include <string>
#include <sstream>

#include <sciplot/sciplot.hpp>
namespace plt = sciplot;

#include <petsc.h>
#include <petsctao.h>
#include <petscts.h>

#define E(X) ierr = X; CHKERRQ(ierr);
#define SQ(X) ((X) * (X))

template<typename T>
constexpr T sqr(const T x)
{
    return x * x;
}

struct PetscScalar2
{
    PetscScalar x;
    PetscScalar y;

    constexpr PetscScalar2(const PetscScalar2& original) : x(original.x), y(original.y)
    { }
    constexpr PetscScalar2(const PetscScalar x, const PetscScalar y) : x(x), y(y)
    { }
    constexpr PetscScalar2(const PetscScalar x) : x(x), y(x)
    { }
#ifdef PETSC_USE_COMPLEX
    constexpr PetscScalar2(const PetscReal x, const PetscReal y) : x(x), y(y)
    { }
    constexpr PetscScalar2(const PetscReal x) : x(x), y(x)
    { }
#endif
    constexpr PetscScalar2() : x(0), y(0)
    { }

    constexpr PetscScalar norm2() const
    {
        return x * x + y * y;
    }
    constexpr PetscScalar norm() const
    {
        return PetscSqrtScalar(norm2());
    }

    constexpr PetscScalar2 operator+ (const PetscScalar2& rhs) const
    {
        return {x + rhs.x, y + rhs.y};
    }
    constexpr PetscScalar2 operator+= (const PetscScalar2&& rhs)
    {
        x += rhs.x; y += rhs.y;
        return *this;
    }
    constexpr PetscScalar2 operator- (const PetscScalar2& rhs) const
    {
        return {x - rhs.x, y - rhs.y};
    }
    constexpr PetscScalar2 operator-= (const PetscScalar2&& rhs)
    {
        x -= rhs.x; y -= rhs.y;
        return *this;
    }
    constexpr PetscScalar2 operator* (const PetscScalar& rhs) const
    {
        return {x * rhs, y * rhs};
    }
    constexpr PetscScalar2 operator*= (const PetscScalar&& rhs)
    {
        x *= rhs; y *= rhs;
        return *this;
    }
    constexpr PetscScalar2 operator/ (const PetscScalar& rhs) const
    {
        return {x / rhs, y / rhs};
    } 
    constexpr PetscScalar2 operator/= (const PetscScalar&& rhs)
    {
        x /= rhs; y /= rhs;
        return *this;
    }
    constexpr void operator= (const PetscScalar2& value)
    {
        x = value.x; y = value.y;
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

constexpr PetscReal clamp(const PetscReal x, const PetscReal a, const PetscReal b)
{
    if (x < a)
        return a;
    if (x > b)
        return b;

    return x;
}

// TODO: remove rest of hardcoded 2D
typedef PetscScalar2 Point;

class Spline
{
    int _Segments;
    int _ControlPoints;
    int _Stride;
    int _FixedPoints;

    Point* _PointBuffer;

public:
    Spline(int segments, int controlPoints, int stride, int fixedPoints) : _Segments(segments), _ControlPoints(controlPoints), _Stride(stride), _FixedPoints(fixedPoints)
    {
        _PointBuffer = new Point[_Segments * _Stride + (_ControlPoints - _Stride)];
    }

    Spline(Spline& original) = delete;
    Spline(Spline&& original) = delete;

    ~Spline()
    {
        delete[] _PointBuffer;
    }

    constexpr int Segments() const noexcept
    {
        return _Segments;
    }

    constexpr int Order() const noexcept
    {
        return _ControlPoints - 1;
    }

    constexpr int TotalControlPoints() const noexcept
    {
        return _Segments * _Stride + (_ControlPoints - _Stride);
    }
    constexpr int FreeControlPoints() const noexcept
    {
        return _Segments * _Stride + (_ControlPoints - _Stride) - 2 * _FixedPoints;
    }

protected:
    Point* SegmentDataBegin(const int segment)
    {
        if (PetscDefined(USE_DEBUG))
        {
            if (segment < 0 || segment >= _Segments)
                throw "segment out of range";
        }
        return _PointBuffer + segment * _Stride;
    }
    const Point* SegmentDataBegin(const int segment) const
    {
        if (PetscDefined(USE_DEBUG))
        {
            if (segment < 0 || segment >= _Segments)
                throw "segment out of range";
        }
        return _PointBuffer + segment * _Stride;
    }

public:
    Point& ControlPoint(const int i)
    {
        if (PetscDefined(USE_DEBUG))
        {
            if (i < 0 || i >= TotalControlPoints())
                throw "control point out of range";
        }
        return _PointBuffer[i];
    }
    const Point& ControlPoint(const int i) const
    {
        if (PetscDefined(USE_DEBUG))
        {
            if (i < 0 || i >= TotalControlPoints())
                throw "control point out of range";
        }
        return _PointBuffer[i];    
    }
    Point& LocalControlPoint(const int segment, const int i)
    {
        if (PetscDefined(USE_DEBUG))
        {
            if (segment < 0 || segment >= _Segments)
                throw "segment out of range";
            if (i < 0 || i >= _ControlPoints)
                throw "control point out of range";
        }

        return _PointBuffer[segment * _Stride + i];
    }
    const Point& LocalControlPoint(const int segment, const int i) const
    {
        if (PetscDefined(USE_DEBUG))
        {
            if (segment < 0 || segment >= _Segments)
                throw "segment out of range";
            if (i < 0 || i >= _ControlPoints)
                throw "control point out of range";
        }

        return _PointBuffer[segment * _Stride + i];
    }

protected:
    virtual constexpr PetscScalar BasisFunction(const PetscInt controlPoint, const PetscReal x) const noexcept = 0;
    virtual constexpr PetscScalar BasisGradient(const PetscInt controlPoint, const PetscReal x) const noexcept = 0;

public:
    virtual constexpr void InterpolateBetween(Point a, Point b) = 0;

    constexpr Point At(const PetscReal x) const
    {
        PetscInt segment = clamp(PetscFloorReal(x), 0, _Segments - 1);
        
        Point value{};
        for (PetscInt c = 0; c < _ControlPoints; ++c)
            value += LocalControlPoint(segment, c) * BasisFunction(c, x - segment);

        return value;
    }

    constexpr Point Grad(const PetscReal x) const
    {
        PetscInt segment = clamp(PetscFloorReal(x), 0, _Segments - 1);
        
        Point value{};
        for (PetscInt c = 0; c < _ControlPoints; ++c)
            value += LocalControlPoint(segment, c) * BasisGradient(c, x - segment);

        return value;
    }
    
    constexpr PetscScalar Var(const int controlPoint, const PetscReal x) const
    {
        PetscInt segment = clamp(PetscFloorReal(x), 0, _Segments - 1);
        
        PetscInt c = controlPoint - segment * _Stride;
        if (c < 0 || c >= _ControlPoints)
            return 0.0;

        return BasisFunction(c, x - segment);
    }
    
    constexpr PetscScalar GradVar(const int controlPoint, const PetscReal x) const
    {
        PetscInt segment = clamp(PetscFloorReal(x), 0, _Segments - 1);

        PetscInt c = controlPoint - segment * _Stride;
        if (c < 0 || c >= _ControlPoints)
            return 0.0;

        return BasisGradient(c, x - segment);    
    }

    PetscErrorCode ReadControlPointsFromTAO(const PetscScalar* data)
    {
        return PetscMemcpy(_PointBuffer + _FixedPoints, data, sizeof(Point) * FreeControlPoints());
    }

    PetscErrorCode WriteControlPointsToTAO(PetscScalar* data) const
    {
        return PetscMemcpy(data, _PointBuffer + _FixedPoints, sizeof(Point) * FreeControlPoints());
    }
};

class Potential
{
public:
    constexpr Potential() { }

    Potential(Potential& original) = delete;
    Potential(Potential&& original) = delete;

    virtual constexpr PetscScalar Value(const Point x, const Point v, const PetscReal t) const noexcept = 0;
    virtual constexpr Point XDerivative(const Point x, const Point v, const PetscReal t) const noexcept = 0;
    virtual constexpr Point VDerivative(const Point x, const Point v, const PetscReal t) const noexcept = 0;
};

PetscErrorCode LagrangeObjectiveTSRHSFunction(TS ts, PetscReal lambda, Vec state, Vec rhs, void* ptr);
PetscErrorCode LagrangeGradientTSRHSFunction(TS ts, PetscReal lambda, Vec state, Vec rhs, void* ptr);

struct TAOContext
{
    PetscReal InitialSpeed = 1.0;
    PetscReal Mass = 1.0;
    PetscReal TimeConversion = 1.0;
    PetscReal InverseTimeConversion = 1.0;
    PetscReal BoundaryRadius = 5.0;

private:
    TS objectiveTS;
    Vec objectiveState;

    TS gradientTS;
    Vec gradientState;
public:
    Spline* spline;
    Potential* potential;

    TAOContext()
    {
        PetscOptionsGetReal(NULL, NULL, "-initial_speed", &InitialSpeed, NULL);
        PetscOptionsGetReal(NULL, NULL, "-mass", &Mass, NULL);

        if (InitialSpeed < PETSC_MACHINE_EPSILON)
            throw "speed has to be positive";
    }

    PetscErrorCode UpdateTimeConversion()
    {
        TimeConversion = PetscRealPart(spline->Grad(0).norm()) / InitialSpeed;
        InverseTimeConversion = 1 / TimeConversion;
        return 0;
    }

    PetscErrorCode SetBoundary(PetscScalar* min, PetscScalar* max)
    {
        Point a = spline->At(0);
        Point b = spline->At(spline->Segments());

        Point Min = ::min(a, b);
        Point Max = ::max(a, b);

        for (int i = 0; i < spline->TotalControlPoints(); ++i)
        {
            min[2 * i] = Min.x;
            min[2 * i + 1] = Min.y;
            max[2 * i] = Max.x;
            max[2 * i + 1] = Max.y;
        }

        return 0;
    }
    // TODO: only use one TS and switch between them
    PetscErrorCode InitTS(PetscInt steps)
    {
        PetscErrorCode ierr = 0;

        E(VecCreateSeq(PETSC_COMM_WORLD, 1, &objectiveState))
        E(VecSetFromOptions(objectiveState))
        E(VecAssemblyBegin(objectiveState))
        E(VecAssemblyEnd(objectiveState))

        E(TSCreate(PETSC_COMM_WORLD, &objectiveTS))
        E(TSSetType(objectiveTS, TSEULER))
        E(TSSetProblemType(objectiveTS, TS_NONLINEAR))
        E(TSSetMaxTime(objectiveTS, spline->Segments()))
        E(TSSetTimeStep(objectiveTS, spline->Segments() / (PetscReal)steps))
        E(TSSetExactFinalTime(objectiveTS, TS_EXACTFINALTIME_MATCHSTEP))
        E(TSSetRHSFunction(objectiveTS, objectiveState, LagrangeObjectiveTSRHSFunction, this))
        E(TSSetFromOptions(objectiveTS))

        E(VecCreateSeq(PETSC_COMM_WORLD, 2 * spline->TotalControlPoints(), &gradientState))
        E(VecSetFromOptions(gradientState))
        E(VecAssemblyBegin(gradientState))
        E(VecAssemblyEnd(gradientState))

        E(TSCreate(PETSC_COMM_WORLD, &gradientTS))
        E(TSSetType(gradientTS, TSEULER))
        E(TSSetProblemType(gradientTS, TS_NONLINEAR))
        E(TSSetMaxTime(gradientTS, spline->Segments()))
        E(TSSetTimeStep(gradientTS, spline->Segments() / (PetscReal)steps))
        E(TSSetExactFinalTime(gradientTS, TS_EXACTFINALTIME_MATCHSTEP))
        E(TSSetRHSFunction(gradientTS, gradientState, LagrangeGradientTSRHSFunction, this))
        E(TSSetFromOptions(gradientTS))

        return ierr;
    }

    PetscErrorCode RunObjectiveTS(PetscScalar* result)
    {
        PetscErrorCode ierr = 0;

        E(VecSet(objectiveState, 0.0))
        E(TSSetTime(objectiveTS, 0.0))
        E(TSSetSolution(objectiveTS, objectiveState))

        E(TSSolve(objectiveTS, objectiveState))
        {
            Vec sol;
            PetscInt i = 0;
            E(TSGetSolution(objectiveTS, &sol))
            E(VecGetValues(sol, 1, &i, result))
        }

        TSConvergedReason reason;
        E(TSGetConvergedReason(objectiveTS, &reason))

        if (reason < 0)
            ierr = reason;
        
        return ierr;
    }

        PetscErrorCode RunGradientTS(PetscScalar* result)
    {
        PetscErrorCode ierr = 0;

        E(VecSet(gradientState, 0.0))
        E(TSSetTime(gradientTS, 0.0))
        E(TSSetSolution(gradientTS, gradientState))

        E(TSSolve(gradientTS, gradientState))
        {
            Vec sol;
            PetscInt is[2 * spline->TotalControlPoints()];
            for (int i = 0; i < 2 * spline->TotalControlPoints(); ++i)
                is[i] = i;

            E(TSGetSolution(gradientTS, &sol))
            E(VecGetValues(sol, 2 * spline->TotalControlPoints(), is, result))
        }

        TSConvergedReason reason;
        E(TSGetConvergedReason(gradientTS, &reason))

        if (reason < 0)
            ierr = reason;
        
        return ierr;
    }
};

PetscErrorCode LagrangeObjectiveTSRHSFunction(TS ts, PetscReal lambda, Vec state, Vec rhs, void* ptr)
{
    PetscErrorCode ierr = 0;

    TAOContext* ctx = (TAOContext*)ptr;

    PetscScalar* rhsData;

    PetscFunctionBegin;

    E(VecGetArray(rhs, &rhsData))

    PetscReal time = lambda * ctx->InverseTimeConversion;
    Point x = ctx->spline->At(lambda);
    Point v = ctx->spline->Grad(lambda);
    PetscScalar V = ctx->potential->Value(x, v * ctx->TimeConversion, time);

    rhsData[0] = 0.5 * ctx->Mass * v.norm2() * ctx->TimeConversion +
                 V                           * ctx->InverseTimeConversion;

    //E(PetscPrintf(PETSC_COMM_WORLD, "S(%5E) = %7E\n", lambda, rhsData[0]))

    E(VecRestoreArray(rhs, &rhsData))

    PetscFunctionReturn(ierr);
}
// FIXME: line search failure (even with g = 0)
PetscErrorCode LagrangeGradientTSRHSFunction(TS ts, PetscReal lambda, Vec state, Vec rhs, void* ptr)
{
    PetscErrorCode ierr = 0;

    TAOContext* ctx = (TAOContext*)ptr;

    PetscScalar* rhsData;

    PetscFunctionBegin;

    E(VecGetArray(rhs, &rhsData))

    PetscReal time = lambda * ctx->InverseTimeConversion;
    Point x = ctx->spline->At(lambda);
    Point v = ctx->spline->Grad(lambda);
    Point dx_V = ctx->potential->XDerivative(x, v * ctx->TimeConversion, time);
    Point dv_V = ctx->potential->VDerivative(x, v * ctx->TimeConversion, time);
    PetscScalar dx, dv;
    Point value;

    for (int i = 0; i < ctx->spline->TotalControlPoints(); ++i)
    {
        dx = ctx->spline->Var(i, lambda);
        dv = ctx->spline->GradVar(i, lambda);

        value = ctx->Mass * v * dv      * ctx->TimeConversion +
                dx_V * dx               * ctx->InverseTimeConversion +
                dv_V * dv               * 1;

        rhsData[2 * i] = value.x;
        rhsData[2 * i + 1] = value.y;
    }

    E(VecRestoreArray(rhs, &rhsData))

    PetscFunctionReturn(ierr);
}

PetscErrorCode LagrangeTAOObjectiveAndGradientRountine(Tao tao, Vec coords, PetscReal* f, Vec grad, void* ptr)
{
    PetscErrorCode ierr = 0;

    TAOContext* ctx = (TAOContext*)ptr;

    const PetscScalar* coordsData;
    PetscScalar* gradData;

    PetscFunctionBegin;

    E(VecGetArrayRead(coords, &coordsData))
    E(VecGetArray(grad, &gradData))

    ctx->spline->ReadControlPointsFromTAO(coordsData);
    ctx->UpdateTimeConversion();
    {
        PetscScalar sol;
        E(ctx->RunObjectiveTS(&sol))
        *f = PetscRealPart(sol);
    }
    ctx->RunGradientTS(gradData);

    E(VecRestoreArray(grad, &gradData))
    E(VecRestoreArrayRead(coords, &coordsData))

    PetscFunctionReturn(ierr);
}
// TODO: draw real curve and not only lerp
PetscErrorCode LagrangePlotCurve(const TAOContext& ctx)
{
    PetscErrorCode ierr = 0;

    plt::Plot plot;

    plot.size(1300, 650);

    plot.xlabel("x");
    plot.ylabel("y");

    plot.legend().show(false);

    plt::Vec xs(ctx.spline->Segments() + 1);
    plt::Vec ys(xs.size());

    {
        Point a;
        for (PetscInt i = 0; i <= ctx.spline->Segments(); ++i)
        {
            a = ctx.spline->At(i);
            xs[i] = PetscRealPart(a.x);
            ys[i] = PetscRealPart(a.y);
        }
    }
    
    plot.drawCurveWithPoints(xs, ys);

    plot.show();

    return ierr;
}

PetscErrorCode LagrangeRunTAO(TAOContext& ctx)
{
    PetscErrorCode ierr = 0;

    Tao tao;
    Vec coords, min, max;

    {
        E(TaoCreate(PETSC_COMM_WORLD, &tao))
        E(TaoSetType(tao, TAOLMVM))

        E(VecCreate(PETSC_COMM_WORLD, &coords))
        E(VecSetType(coords, VECSEQ))
        E(VecSetSizes(coords, PETSC_DECIDE, 2 * ctx.spline->FreeControlPoints()))
        E(VecAssemblyBegin(coords))
        E(VecAssemblyEnd(coords))

        E(VecDuplicate(coords, &min))
        E(VecDuplicate(coords, &max))

        {
            PetscScalar* data, *minData, *maxData;

            E(VecGetArray(coords, &data))

            ctx.spline->WriteControlPointsToTAO(data);

            E(VecRestoreArray(coords, &data))

            E(VecGetArray(min, &minData))
            E(VecGetArray(max, &maxData))

            ctx.SetBoundary(minData, maxData);

            E(VecRestoreArray(max, &maxData))
            E(VecRestoreArray(min, &minData))
        }
        E(TaoSetVariableBounds(tao, min, max))
        E(TaoSetInitialVector(tao, coords))
        E(TaoSetObjectiveAndGradientRoutine(tao, LagrangeTAOObjectiveAndGradientRountine, &ctx))

        PetscReal norm = 1e-6, relNorm = 1e-6, progNorm = 1e-6;
        PetscInt iterations = 100;

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

            Point x;
            for (PetscInt i = 0; i <= ctx.spline->Segments(); ++i)
            {
                x = ctx.spline->At(i);
                E(PetscPrintf(PETSC_COMM_WORLD, "(%5E, %5E)\n", x.x, x.y))
            }

            E(VecRestoreArrayRead(coords, &data))
        }

        E(LagrangePlotCurve(ctx))
    }
    E(VecDestroy(&coords))
    // FIXME: double free
    E(TaoDestroy(&tao))

    return ierr;
}