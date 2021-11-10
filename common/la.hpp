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

    template<typename T>
    struct Triple
    {
        T x{};
        T y{};
        T z{};

        Triple() = default;
        Triple(T _x) : x(_x) {}
        Triple(T _x, T _y) : x(_x), y(_y) {}
        Triple(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

        operator T() noexcept
        {
            return x;
        }
    };

    typedef std::function<PetscScalar(PetscReal)> Potential;

    struct SpecializedPotential
    {
        Potential potential;
        std::pair<Triple<PetscReal>, Triple<PetscReal>> range;

        SpecializedPotential(Potential pot, std::pair<Triple<PetscReal>, Triple<PetscReal>> rng) : potential(pot), range(rng) {}
    };

    template<typename... Ts>
    struct PotentialFamily
    {
        virtual Potential get(Ts... args) = 0;
        virtual std::pair<Triple<PetscReal>, Triple<PetscReal>> range(Ts ...args) = 0;
        virtual SpecializedPotential operator()(Ts ...args)
        {
            return { get(args...), range(args...) };
        }
    };

    struct Harmonic : PotentialFamily<PetscReal>
    {
        Potential get(PetscReal omega)
        {
            const double c = 0.5 * omega * omega;
            return [c](PetscReal x) 
            { 
                return c * x * x; 
            };
        }
        std::pair<Triple<PetscReal>, Triple<PetscReal>> range(PetscReal omega)
        {
            constexpr PetscReal threshold = 4.0;
            PetscReal x = threshold / sqrt(omega);
            return { -x, x };
        }
    } harmonic;

    struct Coulomb : PotentialFamily<PetscReal>
    {
        Potential get(PetscReal amp)
        {
            auto r = range(amp);
            PetscReal min = -200.0 / (r.second.x - r.first.x);
            return [amp, min](PetscReal x) 
            { 
                PetscReal y = -amp / abs(x);
                if (y < min)
                    return min;
                return y;
            };
        }
        std::pair<Triple<PetscReal>, Triple<PetscReal>> range(PetscReal amp)
        {
            constexpr PetscReal threshold = 1e-1;
            PetscReal x = sqrt(1/threshold * amp);
            return { -x, x };
        }
    } coulomb;

    struct Morse : PotentialFamily<PetscReal, PetscReal>
    {
        Potential get(PetscReal r0, PetscReal depth)
        {
            PetscReal a = 1.0 / sqrt(2 * depth);
            return [r0, depth, a](PetscReal x)
            {
                PetscReal e = 1 - exp(-a * (x - r0));
                return depth * e * e;
            };
        }
        std::pair<Triple<PetscReal>, Triple<PetscReal>> range(PetscReal r0, PetscReal depth)
        {
            constexpr PetscReal threshold = 0.99;
            return { 0.0, sqrt(2 * depth) * log(1.0 / (1 - sqrt(threshold))) };
        }
    } morse;

    struct DoubleWell : PotentialFamily<PetscReal, PetscReal>
    {
        Potential get(PetscReal h, PetscReal c)
        {
            PetscReal h4_4 = 0.25 * h * h * h * h, c2_2 = 0.5 * c * c;
            return [h4_4, c2_2](PetscReal x)
            {
                PetscReal x2 = x * x;
                return c2_2 * x2 * x2 - h4_4 * x2;
            };
        }
        std::pair<Triple<PetscReal>, Triple<PetscReal>> range(PetscReal h, PetscReal c)
        {
            constexpr PetscReal threshold = 0.6;
            PetscReal x = threshold * sqrt(1 + sqrt(3)) * h * h / c;
            return { -x, x };
        }
    } doubleWell;

    struct Grid
    {
        ::DM dm;
        PetscInt dimension;
        Triple<DMBoundaryType> boundary;

        Triple<PetscInt> size;
        PetscInt dof;
        PetscInt stencil;

        Triple<PetscReal> min;
        Triple<PetscReal> max;

        Triple<PetscReal> step;

        Triple<PetscInt> begin;
        Triple<PetscInt> end;

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

        Grid(const DMBoundaryType boundaryx, const DMBoundaryType boundaryy, const PetscInt countx, const PetscInt county, const PetscInt dof, const PetscInt stencil, const PetscReal minx, const PetscReal maxx, const PetscReal miny, const PetscReal maxy) :
            dimension(2), boundary(boundaryx, boundaryy), size(countx, county), dof(dof), stencil(stencil), min(minx, miny), max(maxx, maxy), step((maxx - minx) / countx, (maxy - miny) / county)
        {
            E(DMDACreate2d(PETSC_COMM_WORLD, boundaryx, boundaryy, DMDA_STENCIL_BOX, countx, county, PETSC_DECIDE, PETSC_DECIDE, dof, stencil, NULL, NULL, &dm));
            E(DMSetFromOptions(dm));
            E(DMSetUp(dm));
            E(DMDASetUniformCoordinates(dm, minx, maxx, miny, maxy, miny, maxy));
            DMDALocalInfo info;
            E(DMDAGetLocalInfo(dm, &info));
            begin = { info.xs, info.ys };
            end = { info.xs + info.mx, info.ys + info.my };
        }

        Grid(const DMBoundaryType boundaryx, const DMBoundaryType boundaryy, const DMBoundaryType boundaryz, const PetscInt countx, const PetscInt county, const PetscInt countz, const PetscInt dof, const PetscInt stencil, const PetscReal minx, const PetscReal maxx, const PetscReal miny, const PetscReal maxy, const PetscReal minz, const PetscReal maxz) :
            dimension(3), boundary(boundaryx, boundaryy, boundaryz), size(countx, county, countz), dof(dof), stencil(stencil), min(minx, miny, minz), max(maxx, maxy, maxz), step((maxx - minx) / countx, (maxy - miny) / county, (maxz - minz) / countz)
        {
            E(DMDACreate3d(PETSC_COMM_WORLD, boundaryx, boundaryy, boundaryz, DMDA_STENCIL_BOX, countx, county, countz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, stencil, NULL, NULL, NULL, &dm));
            E(DMSetFromOptions(dm));
            E(DMSetUp(dm));
            E(DMDASetUniformCoordinates(dm, minx, maxx, miny, maxy, minz, maxz));
            DMDALocalInfo info;
            E(DMDAGetLocalInfo(dm, &info));
            begin = { info.xs, info.ys, info.zs };
            end = { info.xs + info.mx, info.ys + info.my, info.zs + info.mz };
        }

        std::shared_ptr<PetscScalar[]> getPoints() const
        {
            PetscScalar* ptr;
            E(DMDAGetCoordinateArray(dm, &ptr));
            std::shared_ptr<PetscScalar[]> buffer(ptr, [this](PetscScalar* ptr) 
            {  
                E(DMDARestoreCoordinateArray(this->dm, &ptr));
            });
            
            return buffer;
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
        // TODO: support 2d, 3d
        void apply_diagonal(const InsertMode insert, const std::shared_ptr<Vector> diag);
        // TODO: support 2d, 3d
        void apply_diagonal(const InsertMode insert, const Potential&& func);

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

        std::shared_ptr<PetscScalar[]> getArray()
        {
            PetscScalar* ptr;
            E(VecGetArray(vec, &ptr));
            std::shared_ptr<PetscScalar[]> buffer(ptr, [this](PetscScalar* ptr)
            {
                E(VecRestoreArray(this->vec, &ptr));
            });

            return buffer;
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

        EigenSolver(std::shared_ptr<Matrix> mat, const EPSProblemType type, const PetscInt dimension) : matrix(mat)
        {
            E(EPSCreate(PETSC_COMM_WORLD, &eps));
            E(EPSSetOperators(eps, matrix->mat, NULL));
            E(EPSSetProblemType(eps, type));
            E(EPSSetType(eps, EPSARNOLDI));
            /*RG rg;
            E(EPSGetRG(eps, &rg));
            E(RGSetType(rg, RGINTERVAL));
            E(RGIntervalSetEndpoints(rg, 0.0, 5.0, -0.1, 0.1));
            */
            E(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL/*EPS_TARGET_REAL*//* EPS_ALL*/));
            E(EPSSetTarget(eps, 0.5));
            
            E(EPSSetDimensions(eps, dimension, PETSC_DEFAULT, PETSC_DEFAULT));
            //E(EPSKrylovSchurSetDimensions(eps, dimension, PETSC_DEFAULT, PETSC_DEFAULT));
            E(EPSSetTolerances(eps, 1e-10, 10000));
            //E(EPSSetTrueResidual(eps, PETSC_TRUE));
            E(EPSSetConvergenceTest(eps, EPS_CONV_REL));
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

    // TODO: support 2d, 3d
    void la::Matrix::apply_diagonal(const InsertMode insert, const std::shared_ptr<la::Vector> diag)
    {
        E(MatDiagonalSet(mat, diag->vec, insert));
    }
    // TODO: support 2d, 3d
    void Matrix::apply_diagonal(const InsertMode insert, const Potential&& func)
    {
        Vector vec(grid);
        auto points = grid->getPoints();
        for (int i = 0; i < grid->size; ++i)
        {
            E(VecSetValue(vec.vec, i, func(points[i].real()), INSERT_VALUES));
        }
        vec.assemble();

        E(MatDiagonalSet(mat, vec.vec, insert));
    }

}