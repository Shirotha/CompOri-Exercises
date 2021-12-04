#pragma once

#include <memory>
#include <tuple>
#include <functional>

#include <petsc.h>
#include <petsctao.h>
#include <slepceps.h>

static PetscErrorCode ierr;
#define E(X) do { ierr = (X); if (PetscUnlikely(ierr)) { PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_REPEAT," "); throw ierr; } } while(0)
#define PRINT(format, ...) E(PetscPrintf(PETSC_COMM_WORLD, format "\n", ##__VA_ARGS__))
#define PRINTL(format, ...) E(PetscPrintf(PETSC_COMM_SELF, format "\n", ##__VA_ARGS__))

#define SQ(X) ((X) * (X))

enum DIM
{
    X = 0,
    Y = 1,
    Z = 2
};

namespace la
{
    typedef std::shared_ptr<PetscScalar> Scalar;
    typedef std::shared_ptr<PetscScalar[]> ScalarA;
    typedef std::shared_ptr<const PetscScalar[]> ScalarAR;

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
#if PETSC_USE_COMPLEX
    template<>
    PetscScalar option(const std::string& name, const PetscScalar deflt)
    {
        PetscScalar result = deflt;
        PetscOptionsGetScalar(NULL, NULL, name.c_str(), &result, NULL);
        return result;
    }
#endif
    template<>
    DMBoundaryType option(const std::string& name, const DMBoundaryType deflt)
    {
        DMBoundaryType result = deflt;
        E(PetscOptionsGetEnum(NULL, NULL, name.c_str(), DMBoundaryTypes, (PetscEnum*)&result, NULL));
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
        {
            max /= 2.0;
            min = -max;
        }

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
            end = { info.xs + info.xm, 1, 1 };
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
            end = { info.xs + info.xm, info.ys + info.ym, 1 };
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
            end = { info.xs + info.xm, info.ys + info.ym, info.zs + info.zm };
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

        Matrix(const PetscInt n, const PetscInt m, const PetscInt nonzero)
        {
            E(MatCreateSeqAIJ(PETSC_COMM_WORLD, n, m, nonzero, NULL, &mat));
            E(MatSetFromOptions(mat));
            E(MatSetUp(mat));
        }

        Matrix(const PetscInt n, const PetscInt m)
        {
            E(MatCreate(PETSC_COMM_WORLD, &mat));
            E(MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, n, m));
            E(MatSetFromOptions(mat));
            //E(MatSetOption(mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
            E(MatSetUp(mat));
        }

        ~Matrix()
        {
            ierr = MatDestroy(&mat);
        }
        
        void setValue(const InsertMode insert, const PetscInt i, const PetscInt j, const PetscScalar x)
        {
            E(MatSetValue(mat, i, j, x, insert));
        }

        template<int n, int m>
        void setValues(const InsertMode insert, const std::array<PetscScalar, n * m> values, const int i0=0, const int j0=0)
        {
            PetscInt rows[n];
            for (int i = 0; i < n; ++i)
                rows[i] = i0 + i;

            PetscInt cols[m];
            for (int j = 0; j < m; ++j)
                cols[j] = j0 + j;

            E(MatSetValues(mat, n, rows, m, cols, values.data(), insert));
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

        void assemble(const MatAssemblyType type=MAT_FINAL_ASSEMBLY)
        {
            E(MatAssemblyBegin(mat, type));
            E(MatAssemblyEnd(mat, type));
        }

        static void assemble(Mat mat, const MatAssemblyType type=MAT_FINAL_ASSEMBLY)
        {
            E(MatAssemblyBegin(mat, type));
            E(MatAssemblyEnd(mat, type));
        }

        operator Mat() noexcept
        {
            return mat;
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

        Vector(PetscInt n)
        {
            //E(VecCreateSeq(PETSC_COMM_WORLD, n, &vec));
            E(VecCreate(PETSC_COMM_WORLD, &vec));
            E(VecSetSizes(vec, PETSC_DECIDE, n));
            E(VecSetFromOptions(vec));
        }

        void assemble()
        {
            E(VecAssemblyBegin(vec));
            E(VecAssemblyEnd(vec));
        }

        Triple<PetscInt> size() const 
        {
            if (grid == nullptr)
            {
                PetscInt size;
                E(VecGetSize(vec, &size));
                return size;   
            }
            else
                return grid->size;
        }

        void setValue(const InsertMode insert, const PetscInt i, const PetscScalar x)
        {
            E(VecSetValue(vec, i, x, insert));
        }

        void setValues(PetscScalar x)
        {
            E(VecSet(vec, x));
        }

        static std::shared_ptr<PetscScalar[]> getArray(Vec vec)
        {
            PetscScalar* ptr;
            E(VecGetArray(vec, &ptr));
            std::shared_ptr<PetscScalar[]> buffer(ptr, [vec](PetscScalar* ptr)
            {
                E(VecRestoreArray(vec, &ptr));
            });

            return buffer;
        }

        std::shared_ptr<PetscScalar[]> getArray()
        {
            return getArray(vec);
        }

        static std::shared_ptr<const PetscScalar[]> readArray(Vec vec)
        {
            const PetscScalar* ptr;
            E(VecGetArrayRead(vec, &ptr));
            std::shared_ptr<const PetscScalar[]> buffer(ptr, [vec](const PetscScalar* ptr)
            {
                E(VecRestoreArrayRead(vec, &ptr));
            });

            return buffer;
        }

        std::shared_ptr<const PetscScalar[]> readArray()
        {
            return readArray(vec);
        }

        ~Vector()
        {
            ierr = VecDestroy(&vec);
        }

        operator Vec() noexcept
        {
            return vec;
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
        auto ptr = grid->getPoints();
        PetscScalar* points = (PetscScalar*)ptr.get();

        PetscScalar v;
        MatStencil row, col;
        PetscInt i;
        for (i = grid->begin.x; i < grid->end.x; ++i)
        {
            row.i = i;
            col.i = i;
            v = func(PetscRealPart(points[i]));
            E(MatSetValuesStencil(mat, 1, &row, 1, &col, &v, insert));
        }
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
                v = func(PetscRealPart(points[j][2 * i]), PetscRealPart(points[j][2 * i + 1]));
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
                    v = func(PetscRealPart(points[k][j][3 * i]), PetscRealPart(points[k][j][3 * i + 1]), PetscRealPart(points[k][j][3 * i + 2]));
                    E(MatSetValuesStencil(mat, 1, &row, 1, &col, &v, insert));
                }
    }

    template<typename F>
    struct lambda_traits : lambda_traits<decltype(&F::operator())>
    { };

    template<typename F, typename R, typename...Args>
    struct lambda_traits<R(F::*)(Args...)> : lambda_traits<R(F::*)(Args...) const>
    { };

    template<typename F, typename R, typename... Args>
    struct lambda_traits<R(F::*)(Args...) const>
    {
        using pointer = typename std::add_pointer<R(Args...)>::type;

        static pointer cify(F&& f)
        {
            static F fn = std::forward<F>(f);
            return [](Args... args)
            {
                return fn(std::forward<Args>(args)...);
            };
        }
    };

    template<typename F>
    inline typename lambda_traits<F>::pointer cify(F&& f)
    {
        return lambda_traits<F>::cify(std::forward<F>(f));
    }

    struct OptimizerContext
    {
        Tao tao;

        PetscInt degreesOfFreedom;

        Vector state;
        Vector lowerBound;
        Vector upperBound;

        PetscInt eqConstraintCount;
        PetscInt ieqConstraintCount;

    private:
        Vector eqState;
        Vector ieqState;
        Matrix eqJacobian;
        Matrix ieqJacobian;
        Matrix hessian;

        bool check(void* a, void* b)
        {
            return a != b;
        }

    public:
        // FIXME: virtual methods are not correctly called in constructor
        virtual void initialize() { }
        virtual void lateInitialize() { }
        virtual void finalize() { }
        virtual void calcValueGradient(ScalarAR coords, Scalar value, ScalarA gradient) { }
        virtual void calcHessian(ScalarAR coords, ScalarA hessian/*, ScalarA precon*/) { }
        virtual void calcEqs(ScalarAR coords, ScalarA result) { }
        virtual void calcIeqs(ScalarAR coords, ScalarA result) { }
        virtual void calcEqJacobian(ScalarAR coords, ScalarA jacobian/*, ScalarA precon*/) { }
        virtual void calcIeqJacobian(ScalarAR coords, ScalarA jacobian/*, ScalarA precon*/) { }

        OptimizerContext(const TaoType type, const PetscInt dof, const PetscInt eqCount=0, const PetscInt ieqCount=0) :
            degreesOfFreedom(dof), state(dof), lowerBound(dof), upperBound(dof),
            eqConstraintCount(eqCount), ieqConstraintCount(ieqCount), 
            eqState(eqCount), ieqState(ieqCount),
            eqJacobian(eqCount, dof), ieqJacobian(ieqCount, dof),
            hessian(dof, dof)
        {
            E(TaoCreate(PETSC_COMM_WORLD, &tao));
            E(TaoSetType(tao, type));
        }
        
        void setup()
        {
            initialize();
            E(TaoSetInitialVector(tao, state));
            E(TaoSetVariableBounds(tao, lowerBound, upperBound));
            E(TaoSetObjectiveAndGradientRoutine(tao, cify([this](Tao tao, Vec coords, PetscReal* f, Vec grad, void* self)
            {
                auto x = Vector::readArray(coords);
                auto c = Vector::getArray(grad);
                auto fn = std::shared_ptr<PetscScalar>(new PetscScalar{});

                calcValueGradient(x, fn, c);
                *f = PetscRealPart(*fn);

                return ierr;
            }), this));
            //if ((void*)(&TaoContext::calcHessian) != (void*)(this->*(&TaoContext::calcHessian)))
                E(TaoSetHessianRoutine(tao, hessian, hessian, cify([this](Tao tao, Vec coords, Mat hes, Mat precon, void* self)
                {
                    auto x = Vector::readArray(coords);
                    auto vals = std::shared_ptr<PetscScalar[]>(new PetscScalar[degreesOfFreedom * degreesOfFreedom]{});
                    
                    calcHessian(x, vals);
                    
                    PetscInt range[degreesOfFreedom];
                    for (int i = 0; i < degreesOfFreedom; ++i)
                        range[i] = i;
                        
                    E(MatSetValues(hes, degreesOfFreedom, range, degreesOfFreedom, range, vals.get(), INSERT_VALUES));

                    Matrix::assemble(hes);

                    return ierr;
                }), this));
            //if ((void*)(&TaoContext::calcEqs) != (void*)(this->*(&TaoContext::calcEqs)))
                E(TaoSetEqualityConstraintsRoutine(tao, eqState, cify([this](Tao tao, Vec coords, Vec result, void* self)
                {
                    auto x = Vector::readArray(coords);
                    auto c = Vector::getArray(result);

                    calcEqs(x, c);

                    return ierr;
                }), this));
            //if ((void*)(&TaoContext::calcIeqs) != (void*)(this->*(&TaoContext::calcIeqs)))
                E(TaoSetInequalityConstraintsRoutine(tao, ieqState, cify([this](Tao tao, Vec coords, Vec result, void* self)
                {
                    auto x = Vector::readArray(coords);
                    auto c = Vector::getArray(result);

                    calcIeqs(x, c);

                    return ierr;
                }), this));
            //if ((void*)(&TaoContext::calcEqJacobian) != (void*)(this->*(&TaoContext::calcEqJacobian)))
                E(TaoSetJacobianEqualityRoutine(tao, eqJacobian, eqJacobian, cify([this](Tao tao, Vec coords, Mat jacobian, Mat precon, void* self)
                {
                    auto x = Vector::readArray(coords);
                    auto vals = std::shared_ptr<PetscScalar[]>(new PetscScalar[eqConstraintCount * degreesOfFreedom]{});

                    calcEqJacobian(x, vals);

                    int count = std::max(degreesOfFreedom, eqConstraintCount);
                    PetscInt range[count];
                    for (int i = 0; i < count; ++i)
                        range[i] = i;

                    E(MatSetValues(jacobian, eqConstraintCount, range, degreesOfFreedom, range, vals.get(), INSERT_VALUES));

                    Matrix::assemble(jacobian);
                    
                    return ierr;
                }), this));
            //if ((void*)(&TaoContext::calcIeqJacobian) != (void*)(this->*(&TaoContext::calcIeqJacobian)))
                E(TaoSetJacobianInequalityRoutine(tao, ieqJacobian, ieqJacobian, cify([this](Tao tao, Vec coords, Mat jacobian, Mat precon, void* self)
                {
                    auto x = Vector::readArray(coords);
                    auto vals = std::shared_ptr<PetscScalar[]>(new PetscScalar[ieqConstraintCount * degreesOfFreedom]{});

                    calcIeqJacobian(x, vals);

                    int count = std::max(degreesOfFreedom, ieqConstraintCount);
                    PetscInt range[count];
                    for (int i = 0; i < count; ++i)
                        range[i] = i;

                    E(MatSetValues(jacobian, ieqConstraintCount, range, degreesOfFreedom, range, vals.get(), INSERT_VALUES));

                    Matrix::assemble(jacobian);
                    
                    return ierr;
                }), this));
            E(TaoSetFromOptions(tao));
        }

        ~OptimizerContext()
        {
            finalize();
            TaoDestroy(&tao);
        }

        void configureSolvers(KSPType ksp, PCType pc, MatSolverType solver)
        {
            KSP _ksp;
            PC _pc;
            E(TaoGetKSP(tao, &_ksp));
            E(KSPGetPC(_ksp, &_pc));
            E(PCSetType(_pc, pc));
            E(PCFactorSetMatSolverType(_pc, solver));
            E(KSPSetType(_ksp, ksp));
            E(KSPSetFromOptions(_ksp));
        }

        void configureTolerance(PetscReal norm, PetscReal relNorm, PetscReal progNorm, PetscInt iter)
        {
            E(TaoSetTolerances(tao, norm, relNorm, progNorm));
            E(TaoSetMaximumIterations(tao, iter));
            E(TaoSetMaximumFunctionEvaluations(tao, iter));
        }

        TaoConvergedReason solve()
        {
            lateInitialize();
            E(TaoSolve(tao));

            TaoConvergedReason reason;
            E(TaoGetConvergedReason(tao, &reason));
            return reason;
        }
    };
}