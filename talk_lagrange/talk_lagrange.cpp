#include <petsc.h>

#include "common.hpp"
#include "splines/lerp.hpp"
#include "potentials/gravity.hpp"

int main(int argc, char** argv)
{
    PetscErrorCode ierr = 0;

    E(PetscInitialize(&argc, &argv, NULL, NULL))
    {
        Lerp spline(10);
        spline.InterpolateBetween({0, 2}, {10, 0});

        SimpleGravity potential;

        TAOContext ctx;
        ctx.spline = &spline;
        ctx.potential = &potential;

        ctx.InitTS(100);

        LagrangeRunTAO(ctx);
    }
    E(PetscFinalize())

    return ierr;
}