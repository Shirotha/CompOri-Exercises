#include "../common/la.hpp"

template<int N, int M>
struct LagrangeHermiteContext : la::OptimizerContext
{
    const int n = N;
    const int m = M;
    const PetscReal sixth = 1.0 / 6.0;

    PetscReal mass;

    PetscScalar _xs[N + 1 + 2];
    PetscScalar _ys[N + 1 + 2];

    PetscScalar* xs = _xs + 1;
    PetscScalar* ys = _ys + 1;

    PetscScalar _qxs[N + 1 + 2];
    PetscScalar _qys[N + 1 + 2];

    PetscScalar* qxs = _qxs + 1;
    PetscScalar* qys = _qys + 1;

    PetscScalar h00[M + 1];
    PetscScalar h10[M + 1];
    PetscScalar h01[M + 1];
    PetscScalar h11[M + 1];

    PetscScalar d2[N];

    virtual PetscScalar calcPotential() { return 0.0; }
    virtual void calcPotentialGradient(la::ScalarA g) { }

    LagrangeHermiteContext(PetscReal m, la::Triple<PetscScalar> a, la::Triple<PetscScalar> b) : 
        la::OptimizerContext(TAOLMVM, 2 * (N - 1) + 2 * (N + 1), 0, 0), mass(m)
    {
        xs[0] = a.x;
        ys[0] = a.y;
        xs[N] = b.x;
        ys[N] = b.y;

        xs[-1] = 0;
        ys[-1] = 0;
        xs[N + 1] = 0;
        ys[N + 1] = 0;
        
        qxs[-1] = 0;
        qys[-1] = 0;
        qxs[N + 1] = 0;
        qys[N + 1] = 0;

        double lambda, lambda2, lambda3;
        for (int j = 0; j <= M; ++j)
        {
            lambda = j / (double)M;
            lambda2 = SQ(lambda);
            lambda3 = lambda2 * lambda;

            h00[j] = 2 * lambda3 - 3 * lambda2 + 1;
            h10[j] = lambda3 - 2 * lambda2 + lambda;
            h01[j] = -2 * lambda3 + 3 * lambda2;
            h11[j] = lambda3 - lambda2;
        }

        setup();
    }

    double sampleX(const int i, const int j) const
    {
        return h00[j] * xs[i] + h10[j] * qxs[i] + h01[j] * xs[i + 1] + h11[j] * qxs[i + 1];
    }

    double sampleY(const int i, const int j) const
    {
        return h00[j] * ys[i] + h10[j] * qys[i] + h01[j] * ys[i + 1] + h11[j] * qys[i + 1];
    }

    void updateLength()
    {
        int j;
        double dx, dy;
        for (int i = 0; i < N; ++i)
        {
            d2[i] = 0;
            for (j = 0; j < M; ++j)
            {
                dx = sampleX(i, j + 1) - sampleX(i, j);
                dy = sampleY(i, j + 1) - sampleY(i, j);
                d2[i] += SQ(dx) + SQ(dy);
            }
        }
    }

    void initialize()
    {
        int i;
        auto dx = (xs[N] - xs[0]) / N;
        auto dy = (ys[N] - ys[0]) / N;

        qxs[0] = dx;
        qys[0] = dy;
        qxs[N] = dx;
        qys[N] = dy;

        for (i = 1; i < N; ++i)
        {
            xs[i] = xs[i - 1] + dx;
            ys[i] = ys[i - 1] + dy;
            qxs[i] = dx;
            qys[i] = dy;
        }

        updateLength();

        PetscInt is[N + 1];
        for (i = 0; i < N + 1; ++i)
            is[i] = i;

        E(VecSetValues(state, N - 1, is, xs + 1, INSERT_VALUES));
        for (i = 0; i < N + 1; ++i)
            is[i] += N - 1;

        E(VecSetValues(state, N - 1, is, ys + 1, INSERT_VALUES));
        for (i = 0; i < N + 1; ++i)
            is[i] += N - 1;

        E(VecSetValues(state, N + 1, is, qxs, INSERT_VALUES));
        for (i = 0; i < N + 1; ++ i)
            is[i] += N + 1;

        E(VecSetValues(state, N + 1, is, qys, INSERT_VALUES));

        state.assemble();
    }

    PetscScalar calcValue(la::ScalarAR x)
    {
        PetscInt i0 = 0;
        E(PetscMemcpy(xs + 1, x.get() + i0, (N - 1) * sizeof(PetscScalar)));
        i0 += N - 1;
        E(PetscMemcpy(ys + 1, x.get() + i0, (N - 1) * sizeof(PetscScalar)));
        i0 += N - 1;
        E(PetscMemcpy(qxs, x.get() + i0, (N + 1) * sizeof(PetscScalar)));
        i0 += N + 1;
        E(PetscMemcpy(qys, x.get() + i0, (N + 1) * sizeof(PetscScalar)));

        updateLength();
/* 
        PetscScalar s{};
        for (int i = 0; i < N; ++i)
            s += SQ(d2[i]) * (SQ(qxs[i]) + qxs[i] * qxs[i + 1] + SQ(qxs[i + 1])
               + SQ(qys[i]) + qys[i] * qys[i + 1] + SQ(qys[i + 1]));

        s *= N * sixth * mass;
 */

        PetscScalar s = 0;
        for (int i = 0; i < N; ++i)
            s += d2[i];

        s *= SQ(N) * 0.5 * mass;
        /* 
            s += 2 * SQ(qxs[i]) + 2 * SQ(qxs[i + 1])
               - 3 * (qxs[i] + qxs[i + 1]) * (xs[i + 1] - xs[i])
               + 18 * SQ(xs[i] - xs[i + 1]) - qxs[i] * qxs[i + 1]
               + 2 * SQ(qys[i]) + 2 * SQ(qys[i + 1])
               - 3 * (qys[i] + qys[i + 1]) * (ys[i + 1] - ys[i])
               + 18 * SQ(ys[i] - ys[i + 1]) - qys[i] * qys[i + 1];
        
        s *= SQ(N) * 0.2 * sixth * mass;
         */

        s += calcPotential();

        return s;
    }
    // FIXME: line search failutre -> gradient is probably wrong (but should be correct)
    void calcGradient(la::ScalarAR _, la::ScalarA g)
    {
        /* 
        for (int i = 0; i < N - 1; ++i)
        {
            g[i] = SQ(N) * 0.1 * mass * (qxs[i + 1] - qxs[i - 1] - 12 * (xs[i - 1] - 2 * xs[i] + xs[i + 1]));
            g[i + N - 1] = SQ(N) * 0.1 * mass * (qys[i + 1] - qys[i - 1] - 12 * (ys[i - 1] - 2 * ys[i] + ys[i + 1]));
        }
            
        PetscInt i0 = 2 * (N - 1);
        for (int i = 0; i < N + 1; ++i)
        {
            g[i + i0] = SQ(N) * -0.2 * sixth * mass * (qxs[i - 1] - 8 * qxs[i] + qxs[i + 1] - 3 * (xs[i - 1] - xs[i + 1]));
            g[i + i0 + N + 1] = SQ(N) * -0.2 * sixth * mass * (qys[i - 1] - 8 * qys[i] + qys[i + 1] - 3 * (ys[i - 1] - ys[i + 1]));
        }
        */

        for (int i = 0; i < 2 * (N - 1) + 2 * (N + 1); ++i)
            g[i] = 0;

        calcPotentialGradient(g);
    }
};