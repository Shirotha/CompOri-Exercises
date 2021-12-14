#include "../common/la.hpp"

template<int N>
struct LagrangeLerpContext : la::OptimizerContext
{
    const int n = N;

    PetscReal mass;

    PetscScalar xs[N + 1];
    PetscScalar ys[N + 1];

    virtual PetscScalar calcPotential() { return 0.0; }
    virtual void calcPotentialGradient(la::ScalarA g) { }

    LagrangeLerpContext(PetscReal m, la::Triple<PetscScalar> a, la::Triple<PetscScalar> b) : 
        la::OptimizerContext(TAOLMVM, 2 * (N - 1), 0, 0), mass(m)
    {
        xs[0] = a.x;
        ys[0] = a.y;
        xs[N] = b.x;
        ys[N] = b.y;
        setup();
    }

    void initialize()
    {
        int i;
        auto dx = (xs[N] - xs[0]) / N;
        auto dy = (ys[N] - ys[0]) / N;

        for (i = 1; i < N; ++i)
        {
            xs[i] = xs[i - 1] + dx;
            ys[i] = ys[i - 1] + dy;
        }

        PetscInt is[N - 1];
        for (i = 0; i < N - 1; ++i)
            is[i] = i;

        E(VecSetValues(state, N - 1, is, xs + 1, INSERT_VALUES));
        for (i = 0; i < N - 1; ++i)
            is[i] += N - 1;

        E(VecSetValues(state, N - 1, is, ys + 1, INSERT_VALUES));

        state.assemble();
    }

    PetscScalar calcValue(la::ScalarAR x)
    {
        E(PetscMemcpy(xs + 1, x.get(), (N - 1) * sizeof(PetscScalar)));
        E(PetscMemcpy(ys + 1, x.get() + N - 1, (N - 1) * sizeof(PetscScalar)));

        PetscScalar s{};
        for (int i = 0; i < N; ++i)
            s += SQ(xs[i + 1] - xs[i]) + SQ(ys[i + 1] - ys[i]);

        s *= SQ(N) * 0.5 * mass;

        s += calcPotential();

        return s;
    }

    void calcGradient(la::ScalarAR _, la::ScalarA g)
    {
        for (int i = 0; i < N - 1; ++i)
        {
            g[i] = SQ(N) * mass * (2 * xs[i + 1] - xs[i] - xs[i + 2]);
            g[i + N - 1] = SQ(N) * mass * (2 * ys[i + 1] - ys[i] - ys[i + 2]);
        }

        calcPotentialGradient(g);
    }
};