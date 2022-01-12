#include <valarray>
#include <functional>
#include <iterator>
#include <string>
#include <sstream>

#include <petsc.h>

namespace tp
{
    template<typename IN, typename OUT>
    std::valarray<OUT> map(const std::valarray<IN> data, const std::function<OUT(IN)> func)
    {
        std::valarray<OUT> result((OUT)0, data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result[i] = func(data[i]);
        
        return result;
    }

    PetscInt clamp(PetscInt a, PetscInt b, PetscInt x)
    {
        if (x <= a)
            return a;
        if (x >= b)
            return b;
        return x;
    }

    std::valarray<PetscReal> lerp(const PetscReal a, const PetscReal b, const std::valarray<PetscReal> x)
    {
        return a + x * (b - a);
    }

    std::valarray<PetscReal> unlerp(const PetscReal a, const PetscReal b, const std::valarray<PetscReal> x)
    {
        return (x - a) / (b - a);
    }

    std::valarray<PetscReal> remap(const PetscReal a1, const PetscReal b1, const PetscReal a2, const PetscReal b2, const std::valarray<PetscReal> x)
    {
        return lerp(a2, b2, unlerp(a1, b1, x));
    }

    std::valarray<PetscInt> partition(const std::valarray<PetscReal> data, const PetscInt partitions, const PetscReal min, const PetscReal max)
    {
        // TODO: find better boundaries
        PetscReal a = min < -1e10 ? data.min() : min;
        PetscReal b = max > 1e10 ? data.max() : max;

        if ((min < -1e10 || max > 1e10) && b - a < PETSC_MACHINE_EPSILON)
            b = a + 1;

        return map<PetscReal, PetscInt>(
            remap(a, b,
                  0, partitions - 1,
                  data),
            [partitions](PetscReal x) -> PetscInt
            { 
                return PetscFloorReal(x); 
            });
    }
    
    std::string clear(const PetscInt width, const PetscInt height, const std::string clear="")
    {
        std::stringstream stream;
        stream << "\e[s";
        
        for (PetscInt i = 0; i <= height; ++i)
        {
            if (clear.size() > 0)
                for (PetscInt j = 0; j < width; ++j)
                    stream << clear;

            stream << '\n';
        }

        stream << "\e[u";

        return stream.str();
    }

    std::string move(const PetscInt x, PetscInt y)
    {
        std::stringstream stream;
        if (x)
            stream << "\e[" << PetscAbs(x) << (x < 0 ? 'D' : 'C');
        if (y)
            stream << "\e[" << PetscAbs(y) << (y < 0 ? 'A' : 'B');

        return stream.str();
    }

    std::string drawString(const std::string text, const PetscInt yOffset=0, const PetscInt xOffset=0)
    {
        std::stringstream stream;
        if (xOffset || yOffset)
            stream << "\e[s";

        stream << move(xOffset, yOffset);
        stream << text;

        if (xOffset || yOffset)
            stream << "\e[u";

        return stream.str();
    }

    std::valarray<PetscReal> convertData(const PetscInt size, const PetscScalar** data, const PetscInt begin, const PetscInt end, const PetscInt dof=0)
    {
        PetscInt center = (end + begin) / 2;
        PetscInt stepsize = (end - begin) / size;
        if (stepsize == 0)
            stepsize = PetscSign(end - begin);

        std::valarray<PetscReal> buffer(size);
        for (PetscInt iData = begin + (center % stepsize), iBuffer = 0; iData < end && iBuffer < size; iData += stepsize, ++iBuffer)
            buffer[iBuffer] = PetscRealPart(data[iData][dof]);

        return buffer;
    }

    class Theme
    {
    public:
        virtual std::string operator()(const PetscInt a, const PetscInt x, const PetscInt b) const = 0;
        virtual std::string operator[](const PetscInt color) const = 0;
        virtual int subdivisions() const = 0;
        virtual int colors() const = 0; 
    };

    class BasicTheme : public Theme
    {
        PetscInt _base;
        PetscInt _maxValue;

        std::string _data;
        std::valarray<PetscInt> _indices;
    public:
        BasicTheme(const PetscInt base, const std::string* data) : _base(base), _maxValue(base >> 1)
        {
            _indices.resize(base * base, 0);
            for (size_t i = 0; i < _indices.size(); ++i)
                _indices[i] = (i == 0 ? 0 : _indices[i - 1]) + data[i].size();

            _data.resize(_indices[_indices.size() - 1], ' ');

            for (size_t i = 0; i < _indices.size(); ++i)
            {
                auto is = slice(i);
                _data.replace(is.start(), is.size(), data[i]);
            }
        }

        std::string operator()(const PetscInt a, const PetscInt x, const PetscInt b) const
        {
            PetscInt l = clamp(-_maxValue, _maxValue, x - a);
            PetscInt r = clamp(-_maxValue, _maxValue, b - x);
            PetscInt gbt = _base * l + r;
            PetscInt i = gbt + ((_base * _base) >> 1);
            auto is = slice(i);
            std::string result = _data.substr(is.start(), is.size());
            return result;
        }

        std::string operator[](const PetscInt color) const
        {
            return "\e[97m";
        }

    private:
        std::slice slice(PetscInt i) const
        {
            return std::slice(
                i == 0 ? 0           : _indices[i - 1],
                i == 0 ? _indices[0] : _indices[i] - _indices[i - 1],
                sizeof(PetscChar)
            );
        }

    public:
        PetscInt base() const noexcept
        {
            return _base;
        }
        PetscInt subdivisions() const noexcept
        {
            return 1;
        }
        PetscInt colors() const noexcept
        {
            return 1;
        }
    };

    class AdvancedTheme : public Theme
    {
        PetscInt _base;
        PetscInt _maxValue;

        std::string _data;
        std::valarray<PetscInt> _indices;

        std::string _color_data;
        std::valarray<PetscInt> _color_map;

        std::string _overflow;
    public:
        AdvancedTheme(const PetscInt base, const std::string* data, const PetscInt colors, const std::string* color_data, const std::string overflow="") : _base(base), _maxValue(base >> 1), _overflow(overflow)
        {
            _indices.resize(base * base * 2, 0);
            for (size_t i = 0; i < _indices.size(); ++i)
                _indices[i] = (i == 0 ? 0 : _indices[i - 1]) + data[i].size();

            _data.resize(_indices[_indices.size() - 1], ' ');

            for (size_t i = 0; i < _indices.size(); ++i)
            {
                auto is = slice(i);
                _data.replace(is.start(), is.size(), data[i]);
            }

            _color_map.resize(colors, 0);
            for (size_t i = 0; i < _color_map.size(); ++i)
                _color_map[i] = (i == 0 ? 0 : _color_map[i - 1]) + color_data[i].size();

            _color_data.resize(_color_map[_color_map.size() - 1], ' ');

            for (size_t i = 0; i < _color_map.size(); ++i)
            {
                auto is = color_slice(i);
                _color_data.replace(is.start(), is.size(), color_data[i]);
            }
        }

        std::string operator()(const PetscInt a, const PetscInt x, const PetscInt b) const
        {
            PetscInt l = clamp(-_maxValue, _maxValue, x - a);
            PetscInt r = clamp(-_maxValue, _maxValue, b - x);
            PetscInt gbt = _base * l + r;
            PetscInt i = gbt + ((_base * _base) >> 1) + (x % 2) * (_base * _base);
            auto is = slice(i);
            std::string result = _data.substr(is.start(), is.size());
            if (_overflow.size() > 0)
            {
                PetscInt diff = 0;
                if (l < r)
                    diff = b - x;
                if (l > r)
                    diff = x - b;

                if (diff > _maxValue)
                {
                    diff -= _maxValue;
                    std::string step = move(-1, l > r ? 1 : -1);
                    step = step.append(_overflow);
                    while (diff-- > 0)
                        result = result.append(step);
                }
            }
            return result;
        }
        std::string operator[](PetscInt color) const
        {
            if (color < 0)
                color = 0;
            if (color >= colors())
                color = colors() - 1;

            auto is = color_slice(color);
            std::string result = _color_data.substr(is.start(), is.size());
            return result;
        }

    private:
        std::slice slice(PetscInt i) const
        {
            return std::slice(
                i == 0 ? 0           : _indices[i - 1],
                i == 0 ? _indices[0] : _indices[i] - _indices[i - 1],
                sizeof(PetscChar)
            );
        }
        std::slice color_slice(PetscInt i) const
        {
            return std::slice(
                i == 0 ? 0             : _color_map[i - 1],
                i == 0 ? _color_map[0] : _color_map[i] - _color_map[i - 1],
                sizeof(PetscChar)
            );
        }

    public:
        PetscInt base() const noexcept
        {
            return _base;
        }
        PetscInt subdivisions() const noexcept
        {
            return 2;
        }
        PetscInt colors() const noexcept
        {
            return (PetscInt)_color_map.size();
        }
    };

    const std::string SimpleThemeData[]
    {
        "\\", "\\", "\e[92mV\e[0m", "-", 
        "-", 
        "/", "\e[91mA\e[0m", "-", "/"
    };
    const BasicTheme SimpleTheme(3, SimpleThemeData);

    const std::string DetailedThemeData[]
    {
        "\\", "\\", "\\", "v", "V", "_", "_", "_", "v", "v", "_", "_",
        "_",
        ".", "_", ".", ".", "_", "_", "/", ".", ".", "_", "_", "/",
        "\\", "'", "'", "V", "V", "\\", "\\", "-", ".", ".", "-", "-",
        "-",
        "'", "'", "^", "^", "-", "/", "/", "A", "A", "-", "/", "/"
    };
    const std::string DetailedThemeColors[]
    { // mathematica ColorData["Rainbow"]
        "\e[1;38;2;120;27;134m",
        "\e[1;38;2;70;46;186m",
        "\e[1;38;2;63;98;208m",
        "\e[1;38;2;76;144;192m",
        "\e[1;38;2;99;172;154m",
        "\e[1;38;2;131;186;112m",
        "\e[1;38;2;170;190;82m",
        "\e[1;38;2;206;182;65m",
        "\e[1;38;2;228;153;56m",
        "\e[1;38;2;228;99;45m",
        "\e[1;38;2;219;33;33m"
    };
    const AdvancedTheme DetailedTheme(5, DetailedThemeData, 11, DetailedThemeColors/*, "▕"*/);

    std::string drawCurve(const Theme& theme, const PetscInt width, const PetscInt height, const PetscScalar** data, const PetscInt begin, const PetscInt end, const PetscInt dof=0, const PetscInt color_dof=-1, const PetscReal min=PETSC_NINFINITY, const PetscReal max=PETSC_INFINITY, const PetscReal color_min=PETSC_NINFINITY, const PetscReal color_max=PETSC_INFINITY, const PetscInt leftBoundary=-1, const PetscInt rightBoundary=-1)
    {
        auto buffer = convertData(width, data, begin, end, dof);
        
        std::valarray<PetscReal> color_buffer;
        if (color_dof >= 0)
            color_buffer = convertData(width, data, begin, end, color_dof);
        
        PetscInt count = height * theme.subdivisions();
        auto partitions = partition(buffer, count, min, max);
        
        std::valarray<PetscInt> color_partitions;
        if (color_dof >= 0)
            // TODO: find better boundaries
            color_partitions = partition(color_buffer, theme.colors(), color_min, color_max);
        
        std::stringstream stream;
        
        for (PetscInt i = 0; i < width; ++i)
        {
            if (color_dof >= 0)
                stream << theme[color_partitions[i]];

            if (partitions[i] >= 0 && partitions[i] < count)
                stream << drawString(theme(
                    i == 0 ? (leftBoundary < 0 ? partitions[i] : leftBoundary) : partitions[i - 1],
                    partitions[i],
                    i == width - 1 ? (rightBoundary < 0 ? partitions[i] : rightBoundary) : partitions[i + 1]
                ), height - partitions[i] / theme.subdivisions());
            stream << "\e[C";
        }

        if (color_dof >= 0)
            stream << "\e[0m";

        return stream.str();
    }
}