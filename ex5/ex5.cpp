#include <vector>
#include <string>
#include <sstream>

#include <petsc.h>
#include <petscts.h>

#include <harminv.h>

#include <sciplot/sciplot.hpp>
namespace plt = sciplot;

#include "../common/tp.hpp"

#define E(X) ierr = X; CHKERRQ(ierr);
#define SQ(X) ((X) * (X))

struct TSContext
{
    virtual PetscErrorCode InitialState(void* state) { return 0; };
    virtual PetscErrorCode RHS(PetscReal time, const void* state, void* rhs) = 0;
    virtual PetscErrorCode Monitor(PetscInt step, PetscReal time, const void* state) { return 0; }
};

struct TSDMContext : TSContext
{
    DM dm;
    DMDALocalInfo localInfo;

    Vec localState;
    Vec localRHS;

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

PetscErrorCode TS1RHSFunction(TS ts, PetscReal time, Vec state, Vec rhs, void* ptr)
{
    PetscErrorCode ierr = 0;

    TSContext* ctx = (TSContext*)ptr;

    const PetscScalar* stateData;
    PetscScalar* rhsData;

    PetscFunctionBegin;

    E(VecGetArrayRead(state, &stateData))
    E(VecGetArray(rhs, &rhsData))

    E(ctx->RHS(time, stateData, rhsData))

    E(VecRestoreArray(rhs, &rhsData))
    E(VecRestoreArrayRead(state, &stateData))

    PetscFunctionReturn(ierr);
}

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

PetscErrorCode TSMonitor(TS ts, PetscInt step, PetscReal time, Vec state, void* ptr)
{
    PetscErrorCode ierr = 0;
    
    TSContext* ctx = (TSContext*)ptr;

    const PetscScalar* stateData;

    PetscFunctionBegin;

    E(VecGetArrayRead(state, &stateData))

    E(ctx->Monitor(step, time, stateData))

    E(VecRestoreArrayRead(state, &stateData))

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

        PetscInt iL, iR;
        for (PetscInt i = begin; i < end; ++i)
        {
            iL = i - 1; iR = i + 1;
            if (iL < begin)
                iL += (end - begin);
            if (iR >= end)
                iR -= (end - begin);

            y[i][0] = (x[i][0] * (1 - x[i][0]) * (x[i][0] - alpha) - x[i][1] + iApp + 
                diffusion * ((x[iR][0] - x[iL][0]) / (2 * stepsize))) / epsilon;
            y[i][1] = x[i][0] - gamma * x[i][1];
        }

        return ierr;
    }

    int WIDTH = 100;
    int HEIGHT = 10;
    int FRAMES = 100;
    PetscReal DELAY = 0.1;

    int current_frame = 0;
    // NOTE: state is global not local
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
        
        stream << tp::clear(WIDTH, HEIGHT + 1, " ");
        stream << tp::move(0, 1);
        stream << tp::drawCurve(tp::DetailedTheme, WIDTH, HEIGHT, x, begin, end, 0, 1, -0.5, 1.0);
        stream << tp::move(-WIDTH, -HEIGHT - 2);
        
        E(PetscPrintf(PETSC_COMM_WORLD, stream.str().c_str()))

        if (DELAY > PETSC_MACHINE_EPSILON)
            E(PetscSleep(DELAY))

        return ierr;
    }
};

struct FHNSpectrumContext : TSContext
{
    PetscReal alpha = 0.0;
    PetscReal gamma = 0.5;
    PetscReal iApp = 0.3;
    PetscReal epsilon = 0.01;

    PetscInt steps;
    struct State
    {
        PetscScalar U;
        PetscScalar V;
    } *buffer;

    PetscScalar** indexer;

    FHNSpectrumContext(PetscInt steps) : steps(steps)
    {
        PetscOptionsGetReal(NULL, NULL, "-fhn_alpha", &alpha, NULL);
        PetscOptionsGetReal(NULL, NULL, "-fhn_gamma", &gamma, NULL);
        PetscOptionsGetReal(NULL, NULL, "-fhn_iApp", &iApp, NULL);
        PetscOptionsGetReal(NULL, NULL, "-fhn_epsilon", &epsilon, NULL);

        PetscOptionsGetInt(NULL, NULL, "-monitor_width", &WIDTH, NULL);
        PetscOptionsGetInt(NULL, NULL, "-monitor_height", &HEIGHT, NULL);
        PetscOptionsGetInt(NULL, NULL, "-monitor_stride", &STRIDE, NULL);
        PetscOptionsGetReal(NULL, NULL, "-monitor_delay", &DELAY, NULL);

        buffer = new State[steps];

        PetscInt maxIndex = steps / STRIDE;
        indexer = new PetscScalar*[maxIndex];
        for (int i = 0; i < maxIndex; ++i)
            indexer[i] = &buffer[STRIDE * i].U;
    }

    ~FHNSpectrumContext()
    {
        delete[] buffer;
        delete[] indexer;
    }

    PetscErrorCode InitialState(void* state)
    {
        PetscErrorCode ierr = 0;

        PetscScalar* x = (PetscScalar*)state;

        x[0] = alpha + epsilon;
        x[1] = -epsilon;

        PetscInt count = 2;
        E(PetscOptionsGetScalarArray(NULL, NULL, "-fhn_init", x, &count, NULL))

        return ierr;
    }

    PetscErrorCode RHS(PetscReal time, const void* state, void* rhs)
    {
        PetscErrorCode ierr = 0;

        const PetscScalar* x = (const PetscScalar*)state;
        PetscScalar* y = (PetscScalar*)rhs;

        y[0] = (x[0] * (1 - x[0]) * (x[0] - alpha) - x[1] + iApp) / epsilon;
        y[1] = x[0] - gamma * x[1];

        return ierr;
    }

    int WIDTH = 100;
    int HEIGHT = 10;
    int STRIDE = 4;
    PetscReal DELAY = 0.1;

    PetscErrorCode Monitor(PetscInt step, PetscReal time, const void* state)
    {
        if (step >= steps)
            return 0;

        PetscErrorCode ierr = 0;

        const PetscScalar* x = (const PetscScalar*)state;

        buffer[step].U = x[0];
        buffer[step].V = x[1];

        if (step % STRIDE)
            return 0;

        PetscInt current = step / STRIDE;
        PetscInt begin = current - WIDTH;
        PetscInt end = current + 1;
        PetscInt width = WIDTH;
        if (begin < 0)
        {
            width = current + 1;
            begin = 0;
        }

        std::stringstream stream;

        stream << tp::clear(WIDTH, HEIGHT + 1, " ");
        stream << tp::move(0, 1);
        stream << tp::drawCurve(tp::DetailedTheme, width, HEIGHT, (const PetscScalar**)indexer, begin, end, 0, 1, -0.5, 1.0, 0.0, 0.5);
        stream << tp::move(-WIDTH - WIDTH, -HEIGHT - 2);

        E(PetscPrintf(PETSC_COMM_WORLD, stream.str().c_str()))

        E(PetscSleep(DELAY))     

        return ierr;
    }
};

int main(int argv, char** argc)
{
    PetscErrorCode ierr = 0;
    E(PetscInitialize(&argv, &argc, NULL, NULL))
        
    PetscBool spectrum = PETSC_FALSE;
    E(PetscOptionsGetBool(NULL, NULL, "-spectrum", &spectrum, NULL))

    PetscReal maxTime = 10;
    PetscInt steps = 100000;

    E(PetscOptionsGetReal(NULL, NULL, "-time", &maxTime, NULL))
    E(PetscOptionsGetInt(NULL, NULL, "-steps", &steps, NULL))

    TS ts;
    Vec state;

    E(TSCreate(PETSC_COMM_WORLD, &ts))
    E(TSSetType(ts, TSRK))
    E(TSSetProblemType(ts, TS_NONLINEAR))
    E(TSSetMaxTime(ts, maxTime))
    E(TSSetTimeStep(ts, maxTime / steps))
    E(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER))

    if (spectrum)
    {
        PetscReal minFrequency = 0.001;
        PetscReal maxFrequency = 1.0;

        E(PetscOptionsGetReal(NULL, NULL, "-spectrum_min", &minFrequency, NULL))
        E(PetscOptionsGetReal(NULL, NULL, "-spectrum_max", &maxFrequency, NULL))

        E(VecCreate(PETSC_COMM_WORLD, &state))
        E(VecSetType(state, VECSEQ))
        E(VecSetSizes(state, PETSC_DECIDE, 2))
        E(VecAssemblyBegin(state))
        E(VecAssemblyEnd(state))

        FHNSpectrumContext ctx(steps);
        {
            PetscScalar* data;

            E(VecGetArray(state, &data))

            E(ctx.InitialState(data))

            E(VecRestoreArray(state, &data))
        }

        E(TSSetSolution(ts, state))
        E(TSSetRHSFunction(ts, state, TS1RHSFunction, &ctx))
        E(TSMonitorSet(ts, TSMonitor, &ctx, NULL))
        E(TSSetFromOptions(ts))

        E(TSSolve(ts, state))

        E(PetscPrintf(PETSC_COMM_WORLD, tp::clear(1, ctx.HEIGHT + 1).append(tp::move(0, ctx.HEIGHT + 1)).c_str()))
        {
            PetscReal warmup = 2.0;
            
            E(PetscOptionsGetReal(NULL, NULL, "-spectrum_warmup", &warmup, NULL))
            if (warmup > maxTime * 0.9)
                throw "warmup is too large";

            PetscInt warmupSteps = (PetscInt)(steps * (maxTime - warmup) / maxTime);
            PetscInt restSteps = steps - warmupSteps;

            harminv_complex signal[restSteps];
            PetscBool peak = PETSC_FALSE;
            PetscScalar value;
            PetscReal real;
            PetscInt lastPeak = -1;
            PetscReal period = 0.0;
            PetscInt peaks = 0;

            PetscReal uMin = PETSC_INFINITY, uMax = PETSC_NINFINITY;
            PetscReal vMin = PETSC_INFINITY, vMax = PETSC_NINFINITY;
            for (int i = warmupSteps; i < steps; ++i)
            {
                value = ctx.buffer[i].V;
                real = PetscRealPart(value);
                if (real < vMin)
                    vMin = real;
                if (real > vMax)
                    vMax = real;

                value = ctx.buffer[i].U;
                real = PetscRealPart(value);
                if (real < uMin)
                    uMin = real;
                if (real > uMax)
                    uMax = real;

                if (!peak && real > 0.7)
                {
                    peak = PETSC_TRUE;
                    if (lastPeak >= 0)
                    {
                        period += (i - lastPeak);
                        ++peaks;
                    }
                    lastPeak = i;
                }
                if (peak && real < 0.2)
                    peak = PETSC_FALSE;

                signal[i - warmupSteps] = value;
            }
            // FIXME: buffer returns all zeros?
            E(PetscPrintf(PETSC_COMM_WORLD, "U in [%5E, %5E]; V in [%5E, %5E]\n", uMin, uMax, vMin, vMax))

            if (peaks > 0)
            {
                period /= peaks;
                E(PetscPrintf(PETSC_COMM_WORLD, "frequency guess: %5E\n", 1 / period))
            }

            PetscInt nf = (maxFrequency - minFrequency) * restSteps * 1.1;
            if (nf > 300)
                nf = 300;

            harminv_data data = harminv_data_create(restSteps, signal, minFrequency, maxFrequency, nf);
            harminv_solve(data);

            PetscInt freqs = harminv_get_num_freqs(data);

            E(PetscPrintf(PETSC_COMM_WORLD, "freqeuncy\t\tampiltude\t\tdecay\t\t\terror\n"))

            if (freqs > 0)
            {
                struct harminv_result
                {
                    PetscReal Frequency;
                    PetscReal Decay;
                    PetscReal Error;
                    PetscReal Ampiltude;
                } results[freqs];

                struct harminv_result_compare_amp
                {
                    bool operator() (const harminv_result& a, const harminv_result& b)
                    {
                        return a.Ampiltude > b.Ampiltude;
                    }
                } compare_amp;

                struct harminv_result_compare_freq
                {
                    bool operator() (const harminv_result& a, const harminv_result& b)
                    {
                        return a.Frequency < b.Frequency;
                    }
                } compare_freq;

                harminv_complex amp;
                for (int i = 0; i < freqs; ++i)
                {
                    results[i].Frequency = harminv_get_freq(data, i);
                    results[i].Decay = harminv_get_decay(data, i);
                    results[i].Error = harminv_get_freq_error(data, i);
                    harminv_get_amplitude(&amp, data, i);
                    results[i].Ampiltude = std::abs(amp);

                }

                std::sort(results, results + freqs, compare_amp);

                PetscInt freqCount = 10;
                E(PetscOptionsGetInt(NULL, NULL, "-frequencies", &freqCount, NULL))

                for (int i = 0; i < freqCount; ++i)
                    E(PetscPrintf(PETSC_COMM_WORLD, "%5E\t\t%5E\t\t%5E\t\t%5E\n",
                        results[i].Frequency,
                        results[i].Ampiltude,
                        results[i].Decay,
                        results[i].Error))

                PetscBool plot = PETSC_TRUE;
                E(PetscOptionsGetBool(NULL, NULL, "-spectrum_plot", &plot, NULL))
                if (plot)
                {
                    std::sort(results, results + freqs, compare_freq);

                    plt::Plot plot;
                    plot.size(1300, 650);

                    plot.xlabel("frequncy");
                    //plot.ylabel("amplitude");

                    PetscReal minFreq = 0;
                    PetscReal maxFreq = maxFrequency / 2.0;
                    E(PetscOptionsGetReal(NULL, NULL, "-spectrum_plot_min", &minFreq, NULL))
                    E(PetscOptionsGetReal(NULL, NULL, "-spectrum_plot_max", &maxFreq, NULL))
                    plot.xrange(minFreq, maxFreq);

                    //plot.legend().show(false);

                    plt::Vec xs(freqs);
                    plt::Vec fs(freqs);
                    plt::Vec ds(freqs);
                    for (int i = 0; i < freqs; ++i)
                    {
                        xs[i] = results[i].Frequency;
                        fs[i] = results[i].Ampiltude;
                        ds[i] = std::abs(results[i].Decay);
                    }

                    PetscReal maxF = fs.max();
                    PetscReal maxD = ds.max();
                    fs /= maxF;
                    ds /= maxD;
                    ds *= fs;
                    
                    std::stringstream label;
                    label.setf(label.scientific, label.floatfield);
                    
                    label << "Amplitude (Normalized, max = " << maxF << ')';
                    plot.drawCurveWithPoints(xs, fs).label(label.str());
                    label.str("");

                    label << "Relative Decay (Normalized, max = " << maxD << ')';
                    plot.drawCurveWithPoints(xs, ds).label(label.str());

                    plot.show();
                }
            }
        }
    }
    else
    {
        DM dm;

        PetscInt gridSize = 100;
        E(PetscOptionsGetInt(NULL, NULL, "-gridpoints", &gridSize, NULL))

        E(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, gridSize, 2, 1, NULL, &dm))
        E(DMSetFromOptions(dm))
        E(DMSetUp(dm))
        E(DMDASetUniformCoordinates(dm, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        {
            FHNDiffContext ctx(dm, maxTime);
            
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

            E(TSSetDM(ts, dm))
            E(TSSetSolution(ts, state))
            E(TSSetRHSFunction(ts, state, TSDMRHSFunction, &ctx))
            E(TSMonitorSet(ts, TSDMMonitor, &ctx, NULL))
            E(TSSetFromOptions(ts))

            E(TSSolve(ts, state))

            E(PetscPrintf(PETSC_COMM_WORLD, tp::clear(1, ctx.HEIGHT + 1).append(tp::move(0, ctx.HEIGHT + 1)).c_str()))
        }
        E(DMDestroy(&dm))
    }
    E(VecDestroy(&state))
    E(TSDestroy(&ts))

    E(PetscFinalize())

    return ierr;
}