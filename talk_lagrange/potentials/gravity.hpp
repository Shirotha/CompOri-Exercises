#include "../common.hpp"

class SimpleGravity : public Potential
{
    PetscReal _Gravity = 9.81;
    PetscReal _Mass = 1.0;

public:
    SimpleGravity()
    {
        PetscOptionsGetReal(NULL, NULL, "-gravity", &_Gravity, NULL);
        PetscOptionsGetReal(NULL, NULL, "-mass", &_Mass, NULL);
    }

    constexpr PetscScalar Value(const Point x, const Point v, const PetscReal t) const noexcept
    {
        return _Gravity * _Mass * x.y;
    }

    constexpr Point XDerivative(const Point x, const Point v, const PetscReal t) const noexcept
    {
        return _Gravity * _Mass;
    }

    constexpr Point VDerivative(const Point x, const Point v, const PetscReal t) const noexcept
    {
        return 0;
    }
};