#include <vector>
#include <string>
#include <sstream>

#include <petsc.h>
#include <petscts.h>

#include "../common/tp.hpp"

#define E(X) ierr = X; CHKERRQ(ierr);
#define SQ(X) ((X) * (X))

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
        PetscOptionsGetReal(NULL, NULL, "-fhn_alpha", &alpha, NULL);
        PetscOptionsGetReal(NULL, NULL, "-fhn_gamma", &gamma, NULL);
        PetscOptionsGetReal(NULL, NULL, "-fhn_iApp", &iApp, NULL);
        PetscOptionsGetReal(NULL, NULL, "-fhn_epsilon", &epsilon, NULL);
        PetscOptionsGetReal(NULL, NULL, "-fhn_diffusion", &diffusion, NULL);

        PetscOptionsGetInt(NULL, NULL, "-monitor_width", &WIDTH, NULL);
        PetscOptionsGetInt(NULL, NULL, "-monitor_height", &HEIGHT, NULL);
        PetscOptionsGetInt(NULL, NULL, "-monitor_frames", &FRAMES, NULL);
        PetscOptionsGetReal(NULL, NULL, "-monitor_delay", &DELAY, NULL);
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
    int HEIGHT = 10;
    int FRAMES = 100;
    PetscReal DELAY = 0.1;

    int current_frame = 0;

    PetscErrorCode Monitor(PetscInt step, PetscReal time, const void* state)
    {
        if (FRAMES && time < current_frame * MaxTime / (FRAMES - 1))
            return 0;

        ++current_frame;

        PetscErrorCode ierr = 0;

        const PetscScalar** x = (const PetscScalar**)state;

        PetscInt begin = this->localInfo.xs;
        PetscInt end = begin + this->localInfo.xm;

        std::stringstream stream;
        
        stream << tp::prepare(WIDTH, HEIGHT + 1, " ");
        stream << tp::move(0, 1);
        stream << tp::drawCurve(tp::DetailedTheme, WIDTH, HEIGHT, x, begin, end, 0, 1);
        stream << tp::move(-WIDTH, -HEIGHT - 2);
        
        E(PetscPrintf(PETSC_COMM_WORLD, stream.str().c_str()))

        if (DELAY > PETSC_MACHINE_EPSILON)
            E(PetscSleep(DELAY))

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
            PetscReal maxTime = 1.5;
            PetscInt steps = 1 << 17;

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