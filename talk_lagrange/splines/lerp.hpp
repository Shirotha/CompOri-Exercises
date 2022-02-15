#include "../common.hpp"

// p0 + t * (p1 - p0)
class Lerp : public Spline
{
public:
    Lerp(int segments) : Spline(segments, 2, 1, 1)
    { }

protected:
    constexpr PetscScalar BasisFunction(const PetscInt controlPoint, const PetscReal x) const noexcept
    {
        switch (controlPoint)
        {
            case 0: return 1.0 - x;
            case 1: return x;
            default: return 0;
        }
    }
    constexpr PetscScalar BasisGradient(const PetscInt controlPoint, const PetscReal x) const noexcept
    {
        switch (controlPoint)
        {
            case 0: return -1.0;
            case 1: return 1.0;
            default: return 0;
        }
    }

public:
    constexpr void InterpolateBetween(Point a, Point b)
    {
        for (int i = 0; i < TotalControlPoints(); ++i)
            ControlPoint(i) = a + (i / (PetscReal)Segments()) * (b - a);
    }
};