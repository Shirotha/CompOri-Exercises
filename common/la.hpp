#pragma once

#include <memory>
#include <tuple>
#include <functional>

#include <petsc.h>
#include <slepceps.h>

static PetscErrorCode ierr;
#define E(X) do { ierr = (X); if (PetscUnlikely(ierr)) { PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_REPEAT," "); throw ierr; } } while(0)
#define PRINT(format, ...) E(PetscPrintf(PETSC_COMM_WORLD, format "\n", ##__VA_ARGS__))

enum DIM
{
    X = 0,
    Y = 1,
    Z = 2
};

namespace la
{
    template<typename T>
    T option(const std::string& name, const T deflt);
    template<>
    PetscInt option(const std::string& name, const PetscInt deflt)
    {
        PetscInt result = deflt;
        PetscOptionsGetInt(NULL, NULL, name.c_str(), &result, NULL);
        return result;
    }
    template<>
    PetscBool option(const std::string& name, const PetscBool deflt)
    {
        PetscBool result = deflt;
        PetscOptionsGetBool(NULL, NULL, name.c_str(), &result, NULL);
        return result;
    }
    template<>
    PetscReal option(const std::string& name, const PetscReal deflt)
    {
        PetscReal result = deflt;
        PetscOptionsGetReal(NULL, NULL, name.c_str(), &result, NULL);
        return result;
    }
    template<>
    PetscScalar option(const std::string& name, const PetscScalar deflt)
    {
        PetscScalar result = deflt;
        PetscOptionsGetScalar(NULL, NULL, name.c_str(), &result, NULL);
        return result;
    }

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
        else
            throw "unknown CFD params";
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

    typedef std::function<PetscScalar(PetscReal)> Potential1d;
    typedef std::function<PetscScalar(PetscReal, PetscReal)> Potential2d;
    typedef std::function<PetscScalar(PetscReal, PetscReal, PetscReal)> Potential3d;
    typedef std::pair<Triple<PetscReal>, Triple<PetscReal>> Interval;

    template<typename T>
    struct SpecializedPotential
    {
        T potential;
        Interval range;

        SpecializedPotential(T pot, Interval rng) : potential(pot), range(rng) {}
    };

    template<typename P, typename... Ts>
    struct PotentialFamily
    {
        virtual P get(Ts... args) = 0;
        virtual Interval range(Ts ...args) = 0;
        virtual SpecializedPotential<P> operator()(Ts ...args)
        {
            return { get(args...), range(args...) };
        }
    };

    SpecializedPotential<Potential2d> radial(const SpecializedPotential<Potential1d>& pot)
    {
        PetscReal max = pot.range.second.x;
        PetscReal min = pot.range.first.x;
        if (min >= 0.0)
            min = -max;

        auto p = pot.potential;
        return {
            [p](PetscReal x, PetscReal y) 
            {
                return p(sqrt(x * x + y * y));
            },
            {{ min, min},
             { max, max }}
        };
    }

    SpecializedPotential<Potential3d> spherical(const SpecializedPotential<Potential1d>& pot)
    {
        PetscReal max = pot.range.second.x;
        PetscReal min = pot.range.first.x;
        if (min >= 0.0)
            min = -max;

        auto p = pot.potential;
        return {
            [p](PetscReal x, PetscReal y, PetscReal z) 
            {
                return p(sqrt(x * x + y * y + z * z));
            },
            {{ min, min, min },
             { max, max, max }}
        };
    }

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
            dimension(1), boundary(boundary), size(count, 1, 1), dof(dof), stencil(stencil), min(min), max(max), step((max - min) / count)
        {
            E(DMDACreate1d(PETSC_COMM_WORLD, boundary, count, dof, stencil, NULL, &dm));
            E(DMSetFromOptions(dm));
            E(DMSetUp(dm));
            E(DMDASetUniformCoordinates(dm, min, max, min, max, min, max));
            DMDALocalInfo info;
            E(DMDAGetLocalInfo(dm, &info));
            begin = { info.xs, 0, 0 };
            end = { info.xs + info.mx, 1, 1 };
        }

        Grid(const DMBoundaryType boundaryx, const DMBoundaryType boundaryy, const PetscInt countx, const PetscInt county, const PetscInt dof, const PetscInt stencil, const PetscReal minx, const PetscReal maxx, const PetscReal miny, const PetscReal maxy) :
            dimension(2), boundary(boundaryx, boundaryy), size(countx, county, 1), dof(dof), stencil(stencil), min(minx, miny), max(maxx, maxy), step((maxx - minx) / countx, (maxy - miny) / county)
        {
            E(DMDACreate2d(PETSC_COMM_WORLD, boundaryx, boundaryy, DMDA_STENCIL_BOX, countx, county, PETSC_DECIDE, PETSC_DECIDE, dof, stencil, NULL, NULL, &dm));
            E(DMSetFromOptions(dm));
            E(DMSetUp(dm));
            E(DMDASetUniformCoordinates(dm, minx, maxx, miny, maxy, miny, maxy));
            DMDALocalInfo info;
            E(DMDAGetLocalInfo(dm, &info));
            begin = { info.xs, info.ys, 0 };
            end = { info.xs + info.mx, info.ys + info.my, 1 };
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

        std::shared_ptr<void> getPoints() const
        {
            void* ptr;
            E(DMDAGetCoordinateArray(dm, &ptr));
            std::shared_ptr<void> buffer(ptr, [this](void* ptr) 
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
        
        template<int D, int A, DIM I = X>
        constexpr void apply_cfd(const InsertMode insert, const PetscScalar scale = 1.0)
        {
            constexpr PetscInt n = (D + 3) / 2 + A / 2;
            constexpr PetscInt mi = n / 2;
            
            constexpr auto coeff = la::cfd<D, A>();
            std::array<PetscScalar, n> values;
            
            MatStencil row[1], col[n];
            PetscInt i, j, k, di, c;
            
            for (i = 0; i < n; ++i)
                if constexpr(I == X)
                    values[i] = coeff[i] * scale / pow(grid->step.x, D);
                else if constexpr(I == Y)
                    values[i] = coeff[i] * scale / pow(grid->step.y, D);
                else if constexpr(I == Z)
                    values[i] = coeff[i] * scale / pow(grid->step.z, D);
                else
                    throw "unknown dimension";
            
            for (k = grid->begin.z; k < grid->end.z; ++k)
                for (j = grid->begin.y; j < grid->end.y; ++j)
                    for (i = grid->begin.x; i < grid->end.x; ++i)
                    {
                        row->i = i;
                        row->j = j;
                        row->k = k;
                        for (di = 0; di < n; ++di)
                        {
                            col[di].i = i;
                            col[di].j = j;
                            col[di].k = k;
                        }
                        for (di = 0; di < n; ++di)
                            if constexpr(I == X)
                                col[di].i = i + di - mi;
                            else if constexpr(I == Y)
                                col[di].j = j + di - mi;
                            else if constexpr(I == Z)
                                col[di].k = k + di - mi;
                            else
                                throw "unknown dimension";

                        for (c = 0; c < grid->dof; ++c)
                        {
                            row->c = c;
                            for (di = 0; di < n; ++di)
                                col[di].c = c;

                            E(MatSetValuesStencil(mat, 1, row, n, col, values.data(), insert));
                        }
                    }
        }
        
        void apply_diagonal(const InsertMode insert, const Potential1d&& func);
        void apply_diagonal(const InsertMode insert, const Potential2d&& func);
        void apply_diagonal(const InsertMode insert, const Potential3d&& func);

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
        // NOTE: support multiple solvers (rqcg seems good but slow)
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

    void Matrix::apply_diagonal(const InsertMode insert, const Potential1d&& func)
    {
        Vector vec(grid);
        auto ptr = grid->getPoints();
        PetscScalar* points = (PetscScalar*)ptr.get();;
        for (int i = 0; i < grid->size; ++i)
            E(VecSetValue(vec.vec, i, func(points[i].real()), INSERT_VALUES));
        
        vec.assemble();

        E(MatDiagonalSet(mat, vec.vec, insert));
    }
    void Matrix::apply_diagonal(const InsertMode insert, const Potential2d&& func)
    {
        auto ptr = grid->getPoints();
        PetscScalar** points = (PetscScalar**)ptr.get();

        PetscScalar v;
        MatStencil row, col;
        PetscInt i, j;
        for (j = grid->begin.y; j < grid->end.y; ++j)
            for (i = grid->begin.x; i < grid->end.x; ++i)
            {
                row.i = i;
                row.j = j;
                col.i = i;
                col.j = j;
                v = func(points[j][2 * i].real(), points[j][2 * i + 1].real());
                E(MatSetValuesStencil(mat, 1, &row, 1, &col, &v, insert));
            }
    }
    void Matrix::apply_diagonal(const InsertMode insert, const Potential3d&& func)
    {
        auto ptr = grid->getPoints();
        PetscScalar*** points = (PetscScalar***)ptr.get();

        PetscScalar v;
        MatStencil row, col;
        PetscInt i, j, k;
        for (k = grid->begin.z; k < grid->end.z; ++k)
            for (j = grid->begin.y; j < grid->end.y; ++j)
                for (i = grid->begin.x; i < grid->end.x; ++i)
                {
                    row.i = i;
                    row.j = j;
                    row.k = k;
                    col.i = i;
                    col.j = j;
                    col.k = k;
                    v = func(points[k][j][3 * i].real(), points[k][j][3 * i + 1].real(), points[k][j][3 * i + 2].real());
                    E(MatSetValuesStencil(mat, 1, &row, 1, &col, &v, insert));
                }
    }

}