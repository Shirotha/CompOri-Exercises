#include <vector>
#include <string>
#include <sstream>

#include <petsc.h>
#include <petscts.h>

#define E(X) ierr = X; CHKERRQ(ierr);
#define SQ(X) ((X) * (X))

constexpr PetscInt lerp(const PetscInt a, const PetscInt b, const PetscReal x)
{
    return a + (PetscInt)PetscFloorReal(x * (b - a));
}

constexpr PetscReal unlerp(const PetscReal a, const PetscReal b, const PetscReal x)
{
    return (x - a) / (b - a);
}

constexpr PetscInt remap(const PetscReal a1, const PetscReal b1, const PetscInt a2, const PetscInt b2, const PetscReal x1)
{
    return lerp(a2, b2, unlerp(a1, b1, x1));
}

constexpr PetscInt ThreeWayCompare(const PetscInt a, const PetscInt b)
{
    if (a < b)
        return -1;
    if (a > b)
        return 1;
    return 0;
}

PetscInt BalancedTernaryConvert(const std::vector<PetscInt> digits)
{
    if (digits.size() == 0)
        return 0;

    PetscInt result = 0;
    PetscInt p = 1;
    for (auto d = digits.begin(); d < digits.end(); ++d)
    {
        result += p * *d;
        p *= 3;
    }
    return result;
}

struct TSDMContext
{
    DM dm;
    DMDALocalInfo localInfo;

    Vec localState;
    Vec localRHS;

    virtual PetscErrorCode InitialState(void* state) { return 0; };
    virtual PetscErrorCode RHS(PetscReal time, const void* state, void* rhs) = 0;
    virtual PetscErrorCode Monitor(PetscInt step, PetscReal time, const void* state) { return 0; }

    TSDMContext(DM dm) : dm(dm)
    {
        DMDAGetLocalInfo(dm, &localInfo);

        DMCreateLocalVector(dm, &localState);
        DMCreateLocalVector(dm, &localRHS);
    }

    ~TSDMContext()
    {
        VecDestroy(&localRHS);
        VecDestroy(&localState);
    }
};

PetscErrorCode TSDMRHSFunction(TS ts, PetscReal time, Vec state, Vec rhs, void* ptr)
{
    PetscErrorCode ierr = 0;

    TSDMContext* ctx = (TSDMContext*)ptr;

    const void* stateData;
    void* rhsData;

    PetscFunctionBegin;
    // NOTE: interlocking multiple transforms on same DM doesn't work
    E(DMGlobalToLocal(ctx->dm, state, INSERT_VALUES, ctx->localState))
    E(DMDAVecGetArrayDOFRead(ctx->dm, ctx->localState, &stateData))
    
    E(DMGlobalToLocal(ctx->dm, rhs, INSERT_VALUES, ctx->localRHS))
    E(DMDAVecGetArrayDOF(ctx->dm, ctx->localRHS, &rhsData))

    E(ctx->RHS(time, stateData, rhsData))

    E(DMDAVecRestoreArrayDOF(ctx->dm, ctx->localRHS, &rhsData));
    E(DMLocalToGlobalBegin(ctx->dm, ctx->localRHS, INSERT_VALUES, rhs))

    E(DMDAVecRestoreArrayDOFRead(ctx->dm, ctx->localState, &stateData))

    E(DMLocalToGlobalEnd(ctx->dm, ctx->localRHS, INSERT_VALUES, rhs))

    PetscFunctionReturn(ierr);
}

PetscErrorCode TSDMMonitor(TS ts, PetscInt step, PetscReal time, Vec state, void* ptr)
{
    PetscErrorCode ierr = 0;

    TSDMContext* ctx = (TSDMContext*)ptr;

    const void* stateData;

    PetscFunctionBegin;

    E(DMDAVecGetArrayDOFRead(ctx->dm, state, &stateData))

    E(ctx->Monitor(step, time, stateData))

    E(DMDAVecRestoreArrayDOFRead(ctx->dm, state, &stateData))

    PetscFunctionReturn(ierr);
}

struct FHNDiffContext : TSDMContext
{
    PetscReal alpha = 0.0;
    PetscReal gamma = 0.5;
    PetscReal iApp = 0.0;
    PetscReal epsilon = 0.1;
    PetscReal diffusion = 0.05;

    PetscReal MaxTime;

    FHNDiffContext(DM dm, PetscReal maxTime) : TSDMContext(dm), MaxTime(maxTime)
    { 
        PetscOptionsGetInt(NULL, NULL, "-monitor_width", &WIDTH, NULL);
        PetscOptionsGetInt(NULL, NULL, "-monitor_height", &HEIGHT, NULL);
        PetscOptionsGetInt(NULL, NULL, "-monitor_frames", &FRAMES, NULL);
        monitor_buffer = new PetscReal[WIDTH];
        monitor_coords = new PetscInt[WIDTH];

        monitor_stream.setf(monitor_stream.scientific, monitor_stream.floatfield);
        
        // TODO: set parameters from options
    }

    ~FHNDiffContext()
    {
        delete[] monitor_buffer;
        delete[] monitor_coords;
    }

    PetscErrorCode InitialState(void* state)
    {
        PetscErrorCode ierr = 0;

        PetscScalar** x = (PetscScalar**)state;

        auto begin = this->localInfo.xs;
        auto end = begin + this->localInfo.xm;

        PetscInt iSpike = 7 * (begin + end) / 8;

        for (PetscInt i = begin; i < end; ++i)
        {
            x[i][0] = exp(-0.1 * SQ(i - iSpike));
            x[i][1] = 0;
        }

        return ierr;
    }

    PetscErrorCode RHS(PetscReal time, const void* state, void* rhs)
    {
        PetscErrorCode ierr = 0;

        const PetscScalar** x = (const PetscScalar**)state;
        PetscScalar** y = (PetscScalar**)rhs;
        
        PetscInt begin = this->localInfo.xs;
        PetscInt end = begin + this->localInfo.xm;
        // NOTE: only with range 1.0
        PetscReal stepsize =  1.0 / this->localInfo.xm;

        for (PetscInt i = begin; i < end; ++i)
        {
            y[i][0] = (x[i][0] * (1 - x[i][0]) * (x[i][0] - alpha) - x[i][1] + iApp + 
                diffusion * ((x[i + 1][0] - x[i - 1][0]) / (2 * stepsize))) / epsilon;
            y[i][1] = x[i][0] - gamma * x[i][1];
        }

        return ierr;
    }

    int WIDTH = 100;
    int HEIGHT = 6;
    int FRAMES = 10;
    int current_frame = 0;

    PetscReal* monitor_buffer;
    PetscInt* monitor_coords;

    std::stringstream monitor_stream;

    PetscErrorCode Monitor(PetscInt step, PetscReal time, const void* state)
    {
        if (time < current_frame * MaxTime / FRAMES)
            return 0;

        ++current_frame;

        PetscErrorCode ierr = 0;

        const PetscScalar** x = (const PetscScalar**)state;

        PetscInt begin = this->localInfo.xs;
        PetscInt end = begin + this->localInfo.xm;
        PetscInt center = (begin + end) / 2;

        /*
        E(PetscPrintf(PETSC_COMM_WORLD, "v(0.5, %9E) = %9E\n", time, x[center][0]))
        */

        PetscInt stepsize = (end - begin) / WIDTH;
        if (stepsize <= 0)
            stepsize = 1;

        PetscReal current, min = PETSC_MAX_REAL, max = PETSC_MIN_REAL;
        for (PetscInt iData = center % stepsize, iBuffer = 0; iData < end; iData += stepsize, ++iBuffer)
        {
            current = PetscRealPart(x[iData][0]);
            if (current < min)
                min = current;
            if (current > max)
                max = current;

#ifdef PETSC_DEBUG
            if (iBuffer >= WIDTH)
                throw "out of range";
#endif

            monitor_buffer[iBuffer] = current;
        }
        
        PetscInt i;
        for (i = 0; i < WIDTH; ++i)
            monitor_coords[i] = remap(min, max, 0, HEIGHT - 1, monitor_buffer[i]);

        monitor_stream << max << '\n';
        for (PetscInt j = HEIGHT - 1; j >= 0; --j)
        {
            for (i = 0; i < WIDTH; ++i)
                if (monitor_coords[i] == j)
                    switch (BalancedTernaryConvert({
                        ThreeWayCompare(i == WIDTH - 1 ? 0 : monitor_coords[i + 1], monitor_coords[i]),
                        ThreeWayCompare(monitor_coords[i], i == 0 ? 0 : monitor_coords[i - 1])
                        }))
                    {
                        case -4:
                        case -1:
                            monitor_stream << '\\';
                            break;
                        case -2:
                            monitor_stream << 'v';
                            break;
                        case -3:
                        case  1:
                            // TODO: change to '_' at one row higher (or subdivide into _- and _ in upper row) need generalized balanced ternary for different sized steps (/\ for large steps and . for small steps) (use half steps) (| for very steep steps, for smaller steps ,')
                            monitor_stream << '-';
                            break;
                        case  0:
                            monitor_stream << '-';
                            break;
                        case  2:
                            monitor_stream << '^';
                            break;
                        case  3:
                        case  4:
                            monitor_stream << '/';
                            break;
                    }
                else
                    monitor_stream << ' ';

            monitor_stream << '\n';
        }
        monitor_stream << min << '\n';

        PetscPrintf(PETSC_COMM_WORLD, monitor_stream.str().c_str());

        monitor_stream.str("");
        return ierr;
    }
};

int main(int argv, char** argc)
{
    PetscErrorCode ierr = 0;
    E(PetscInitialize(&argv, &argc, NULL, NULL))
    {
        DM dm;
        TS ts;

        E(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, 100, 2, 1, NULL, &dm))
        E(DMSetFromOptions(dm))
        E(DMSetUp(dm))
        E(DMDASetUniformCoordinates(dm, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        {
            PetscReal maxTime = 1.0;
            PetscInt steps = 10000;

            E(PetscOptionsGetReal(NULL, NULL, "-time", &maxTime, NULL))
            E(PetscOptionsGetInt(NULL, NULL, "-steps", &steps, NULL))

            FHNDiffContext ctx(dm, maxTime);
            
            Vec state;
            E(DMCreateGlobalVector(dm, &state))
            {
                Vec local;
                void* data;
                E(DMGetLocalVector(dm, &local))
                E(DMGlobalToLocal(dm, state, INSERT_VALUES, local))
                E(DMDAVecGetArrayDOF(dm, local, &data))

                E(ctx.InitialState(data))

                E(DMDAVecRestoreArrayDOF(dm, local, &data))
                E(DMLocalToGlobal(dm, local, INSERT_VALUES, state))
                E(DMRestoreLocalVector(dm, &local))
            }

            E(TSCreate(PETSC_COMM_WORLD, &ts))
            E(TSSetType(ts, TSRK))
            E(TSSetProblemType(ts, TS_NONLINEAR))
            E(TSSetDM(ts, dm))
            E(TSSetSolution(ts, state))
            E(TSSetMaxTime(ts, maxTime))
            E(TSSetTimeStep(ts, maxTime / steps))
            E(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER))
            E(TSSetRHSFunction(ts, state, TSDMRHSFunction, &ctx))
            E(TSMonitorSet(ts, TSDMMonitor, &ctx, NULL))
            E(TSSetFromOptions(ts))

            E(TSSolve(ts, state))

            E(VecDestroy(&state))
        }
        E(TSDestroy(&ts))
        E(DMDestroy(&dm))
    }
    E(PetscFinalize())
    return ierr;
}