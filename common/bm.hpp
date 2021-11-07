#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <omp.h>

namespace bm
{
    class BM
    {
        double _start;

        std::vector<double> _results{};

    public:
        BM() {}

        void start()
        {
            _start = omp_get_wtime();
        }

        void stop()
        {
            _results.push_back(omp_get_wtime() - _start);
        }

        void lap()
        {
            double tmp = omp_get_wtime();
            _results.push_back(tmp - _start);
            _start = std::move(tmp);
        }

        double get() const
        {
            return _results[_results.size() - 1];
        }

        int size() const
        {
            return _results.size();
        }

        double operator[] (int i) const
        {
            return _results[i];
        }

        double getTotal() const
        {
            double result = 0;
            for (auto i : _results)
                result += i;

            return result;
        }

        double getMean() const
        {
            if (_results.size() == 0)
                return 0;

            double result = 0;
            for (auto i : _results)
                result += i;

            return result / _results.size();
        }

        double getError() const
        {
            return omp_get_wtick();
        }

        double getMeanError() const
        {
            if (_results.size() == 0)
                return 0;

            return omp_get_wtick() / sqrt(_results.size());
        }
    };
}