#pragma once

#include <memory>
#include <tuple>
#include <functional>

#include <petsc.h>
#include <slepceps.h>


static PetscErrorCode ierr;
#define E(X) do { ierr = (X); if (PetscUnlikely(ierr)) { PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_REPEAT," "); throw ierr; } } while(0)
#define PRINT(format, args...) E(PetscPrintf(PETSC_COMM_WORLD, format "\n", args))

namespace la
{
    struct Vector;

    constexpr std::array<PetscScalar, 3> CFD_D1_A2 { -1.0 / 2.0, 0.0, -1.0 / 2.0 };
    constexpr std::array<PetscScalar, 3> CFD_D2_A2 { 1.0, -2.0, 1.0 };

    template<int D, int A>
    constexpr std::array<PetscScalar, (D + 3) / 2 + A / 2> cfd()
    {
        if constexpr(D == 1)
        {
            if constexpr(A == 2)
                return CFD_D1_A2;
        }
        else if constexpr(D == 2)
        {
            if constexpr(A == 2)
                return CFD_D2_A2;
        }

        throw "unsupported fd order";
    } 

    struct Grid
    {
        ::DM dm;
        PetscInt dimension;
        DMBoundaryType boundary;

        PetscInt size;
        PetscInt dof;
        PetscInt stencil;

        PetscReal min;
        PetscReal max;

        PetscReal step;

        PetscInt begin;
        PetscInt end;

        Grid(const DMBoundaryType boundary, const PetscInt count, const PetscInt dof, const PetscInt stencil, const PetscReal min, const PetscReal max) :
            dimension(1), boundary(boundary), size(count), dof(dof), stencil(stencil), min(min), max(max), step((max - min) / count)
        {
            E(DMDACreate1d(PETSC_COMM_WORLD, boundary, count, dof, stencil, NULL, &dm));
            E(DMSetFromOptions(dm));
            E(DMSetUp(dm));
            E(DMDASetUniformCoordinates(dm, min, max, min, max, min, max));
            DMDALocalInfo info;
            E(DMDAGetLocalInfo(dm, &info));
            begin = info.xs;
            end = info.xs + info.mx;
        }

        ~Grid()
        {
            ierr = DMDestroy(&dm);
        }
    };

    struct Matrix
    {
        ::Mat mat;
        std::shared_ptr<Grid> grid;

        Matrix(std::shared_ptr<la::Grid> g) : grid(g)
        {
            E(DMCreateMatrix(grid->dm, &mat));
        }

        ~Matrix()
        {
            ierr = MatDestroy(&mat);
        }
        // TODO: support 2d, 3d
        template<int D, int A>
        constexpr void apply_cfd(const InsertMode insert, const PetscScalar scale = 1.0)
        {
            constexpr PetscInt n = (D + 3) / 2 + A / 2;
            constexpr PetscInt mi = n / 2;
            
            constexpr auto coeff = la::cfd<D, A>();
            std::array<PetscScalar, n> values;

            MatStencil row[1], col[n];
            PetscInt i, di, c;
            
            for (i = 0; i < n; ++i)
                values[i] = coeff[i] * scale / pow(grid->step, D);
            
            for (i = grid->begin; i < grid->end; ++i)
            {
                row->i = i;
                for (di = 0; di < n; ++di)
                    col[di].i = i + di - mi;

                for (c = 0; c < grid->dof; ++c)
                {
                    row->c = c;
                    for (di = 0; di < n; ++di)
                        col[di].c = c;

                    E(MatSetValuesStencil(mat, 1, row, n, col, values.data(), insert));
                }
            }
        }
        
        void apply_diagonal(const InsertMode insert, const std::shared_ptr<Vector> diag);

        void apply_diagonal(const InsertMode insert, const std::function<PetscScalar(PetscReal)>&& func);

        void assemble(const MatAssemblyType type)
        {
            E(MatAssemblyBegin(mat, type));
            E(MatAssemblyEnd(mat, type));
        }
    };

    struct Vector
    {
        ::Vec vec;
        std::shared_ptr<Grid> grid;

        Vector(std::shared_ptr<Grid> g) : grid(g)
        {
            E(DMCreateGlobalVector(grid->dm, &vec));
        }

        void assemble()
        {
            E(VecAssemblyBegin(vec));
            E(VecAssemblyEnd(vec));
        }

        ~Vector()
        {
            ierr = VecDestroy(&vec);
        }
    };

    struct EigenSolver
    {
        ::EPS eps;
        std::shared_ptr<Matrix> matrix;

        EigenSolver(std::shared_ptr<Matrix> mat, const EPSProblemType type) : matrix(mat)
        {
            E(EPSCreate(PETSC_COMM_WORLD, &eps));
            E(EPSSetOperators(eps, matrix->mat, NULL));
            E(EPSSetProblemType(eps, type));
            E(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL));
            E(EPSSetFromOptions(eps));
        }

        ~EigenSolver()
        {
            ierr = EPSDestroy(&eps);
        }

        void solve()
        {
            E(EPSSolve(eps));
        }

        int getNumIterations() const
        {
            PetscInt iterations;
            E(EPSGetIterationNumber(eps, &iterations));
            return std::move(iterations);
        }

        EPSType getType() const
        {
            EPSType type;
            E(EPSGetType(eps, &type));
            return std::move(type);
        }

        int getDimension() const
        {
            PetscInt dimension;
            E(EPSGetDimensions(eps, &dimension, NULL, NULL));
            return std::move(dimension);
        }

        std::tuple<int, double> getTolerance() const
        {
            PetscInt iterations;
            PetscReal tolerance;
            E(EPSGetTolerances(eps, &tolerance, &iterations));
            return std::move(std::make_tuple(std::move(iterations), std::move(tolerance)));
        }

        int getNumEigenpairs() const
        {
            PetscInt convergedEigenvalues;
            E(EPSGetConverged(eps, &convergedEigenvalues));
            return std::move(convergedEigenvalues);
        }

        void getEigenpair(const int i, PetscScalar& value, Vector& vector, PetscReal& error) const
        {
            E(EPSGetEigenpair(eps, i, &value, NULL, vector.vec, NULL));
            E(EPSComputeError(eps, i, EPS_ERROR_ABSOLUTE, &error));
        }
    };

    void la::Matrix::apply_diagonal(const InsertMode insert, const std::shared_ptr<la::Vector> diag)
    {
        E(MatDiagonalSet(mat, diag->vec, insert));
    }

    void Matrix::apply_diagonal(const InsertMode insert, const std::function<PetscScalar(PetscReal)>&& func)
    {
        Vector vec(grid);
        PetscReal x = grid->min;
        for (int i = 0; i < grid->size; ++i)
        {
            E(VecSetValue(vec.vec, i, func(x), INSERT_VALUES));
            x += grid->step;
        }
        vec.assemble();

        E(MatDiagonalSet(mat, vec.vec, insert));
    }

}