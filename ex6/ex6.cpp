#include <vector>
#include <numeric>

#include <sciplot/sciplot.hpp>
namespace plt = sciplot;

#include <itensor/all.h>
#include <itensor/util/print_macro.h>
using namespace itensor;

std::vector<ITensor> MPSDecompose(ITensor tensor, Args args = Args::global())
{
    std::vector<ITensor> result;

    auto is = inds(tensor);
    auto i = is.begin(), end = --is.end();

    auto [u, s, v] = svd(tensor, *i, args);
    while (++i < end)
    {
        tensor = s * v;
        result.push_back(u);
        std::tie(u, s, v) = svd(tensor, {*i, commonIndex(u, s)}, args);
    }
    result.push_back(u);
    result.push_back(s * v);

    return result;
}

void __TEST__MPSDecompose()
{
    auto i = Index(2, "i");
    auto j = Index(2, "j");
    auto k = Index(2, "k");
    auto l = Index(2, "l");

    auto t = randomITensor(i, j, k, l);

    Print(t);

    auto mps = MPSDecompose(t);

    for (auto s : mps)
        Print(s);

    t = std::accumulate(mps.begin(), mps.end(), ITensor(1), [](ITensor a, ITensor b) { return a * b; });

    Print(t);
}



std::tuple<MPO, MPS> InitFullLMG(int n, Args args = Args::global())
{
    auto h = args.getReal("Magnetization", 1.0);
    auto gammaX = args.getReal("Interaction", 1.0);
    
    auto sites = SpinOne(n, {
        "ConserveSz=", false,
        "ConserveParity=", true
    });

    auto ampo = AutoMPO(sites);
    for (auto i : range1(n))
    {
        ampo += -h/2.0, "Sz", i;
        for (auto j : range1(n))
            ampo += -gammaX/n/4.0, "Sx", i, "Sx", j;

    }

    auto H = toMPO(ampo);

    auto state = InitState(sites);
    for(auto i : range1(n))
        state.set(i, i % 2 ? "Up" : "Dn");
        
    auto psi0 = MPS(state);

    return std::make_tuple(H, psi0);
}

void __TEST__FullLMG()
{
    int N = 100;
    int M = 50;

    plt::Vec xs = plt::linspace(0.1, 3.0, M);
    plt::Vec ys(xs.size());
    plt::Vec dys(xs.size());
    plt::Vec ddys(xs.size());

    for (auto i : range(xs.size()))
    {
        auto [H, psi0] = InitFullLMG(N, {
            "Magnetization=", 1.0,
            "Interaction=", xs[i]
        });

        auto sweeps = Sweeps(5);
        sweeps.maxdim() = 10, 20, 100, 100, 200;
        sweeps.cutoff() = 1E-10;
        sweeps.niter() = 2;
        sweeps.noise() = 1E-7, 1E-8, 0.0;

        auto [energy, psi] = dmrg(H, psi0, sweeps, { 
            "Silent=", true 
        });

        ys[i] = energy;

        printfln("gammaX = %.3f --> <H>0 = %.5f", xs[i], ys[i]);
    }

    for (size_t i = 0; i < xs.size(); ++i)
        if (i % (xs.size() - 1))
        {
            dys[i] = (ys[i + 1] - ys[i - 1]) / 2;
            ddys[i] = (ys[i - 1] - 2 * ys[i] + ys[i + 1]);
        }

    dys[0] = dys[1];
    dys[xs.size() - 1] = dys[xs.size() - 2];
    ddys[0] = ddys[1];
    ddys[xs.size() - 1] = ddys[xs.size() - 2];

    dys /= xs[1] - xs[0];
    ddys /= xs[1] - xs[0];

    plt::Plot plot;

    plot.size(1300, 650);

    plot.xlabel("gammaX / h");

    plot.drawCurve(xs, ys).label("Ground state");
    plot.drawCurve(xs, dys).label("Ground state derivative");
    plot.drawCurve(xs, ddys).label("Ground state second derivative");

    plot.show();
}



std::tuple<ITensor, ITensor, ITensor> InitCollectiveOperators(int j, Index& m)
{
    auto mP = prime(m);

    auto Jz = ITensor(m, mP);
    auto Jp = ITensor(m, mP);
    auto Jm = ITensor(m, mP);

    int k;
    for (int mVal = -j; mVal <= j; ++mVal)
    {
        k = mVal + j + 1;
        Jz.set(m = k, mP = k, mVal);
        if (mVal < j)
            Jp.set(m = k, mP = k + 1, sqrt(j * (j + 1) - mVal * (mVal + 1)));
        if (mVal > -j)
            Jm.set(m = k, mP = k - 1, sqrt(j * (j + 1) - mVal * (mVal - 1)));
    }

    return std::make_tuple(Jz, Jp, Jm);
}

ITensor InitCollectiveLMG(int n, Args args = Args::global())
{
    auto h = args.getReal("Magnetization", 1.0);
    auto gammaX = args.getReal("Interaction", 1.0);

    auto j = n/2;
    auto m = Index(2 * j + 1, "j=" + str(j) + ",m");

    auto [Jz, Jp, Jm] = InitCollectiveOperators(j, m);
    auto Jx = 0.5 * (Jp + Jm);
    auto Jx2 = mapPrime(Jx * prime(Jx), 2, 1);

    auto H = -h * Jz - gammaX / n * Jx2;

    return H;
}

void __TEST__CollectiveLMG()
{
    int N = 100;
    int M = 50;

    assert(N % 2 == 0);

    plt::Vec xs = plt::linspace(0.1, 3.0, M);
    plt::Vec ys[N + 1];
    for (int i : range(N + 1))
        ys[i] = plt::Vec(xs.size());

    for (int i : range(xs.size()))
    {
        auto H = InitCollectiveLMG(N, {
            "Magnetization=", 1.0,
            "Interaction=", xs[i]
        });

        auto [trafo, diag] = diagHermitian(H);

        auto n = commonIndex(trafo, diag);
        auto nP = prime(n);
        
        for (auto k : range1(N + 1))
            ys[k - 1][i] = elt(diag, n=k, nP=k);

        printfln("gammaX = %.3f", xs[i]);
    }
    
    plt::Plot plot;

    plot.size(1300, 650);

    plot.xlabel("gammaX / h");
    plot.ylabel("E");

    plot.legend().show(false);

    for (int i : range(N + 1))
        plot.drawCurve(xs, ys[N - i])
            .lineColor(i % 2 ? "orange" : "blue")
            .lineType(1 - (i % 2));

    plot.show();



    int D = 50;

    plt::Vec d_xs = plt::linspace(ys[N].min(), ys[0].max(), D);
    auto d_dx = d_xs[1] - d_xs[0];

    plt::Vec d_ys(d_xs.size());
    plt::Vec sum(d_xs.size());

    for (int i : range(xs.size()))
    {
        int i_y = 0;
        for (auto upper_i : range1(d_xs.size() - 1))
        {
            auto lower_y = d_xs[upper_i - 1];
            auto upper_y = d_xs[upper_i];

            assert(ys[N - i_y][i] >= lower_y);
            while (upper_y > ys[N - i_y][i])
            {
                ++i_y;
                if (i_y > N)
                {
                    while (upper_i < d_xs.size())
                        sum[upper_i++] = i_y;

                    goto end_ys;
                }
            }
                
            sum[upper_i] = i_y;
        }

    end_ys:
        for (auto i : range(d_xs.size()))
            if (i % (d_xs.size() - 1))
                d_ys[i] += (sum[i + 1] - sum[i - 1]) / (2 * d_dx);
    }

    d_ys /= xs.size();

    plt::Plot d_plot;

    d_plot.size(1300, 650);

    d_plot.xlabel("E");
    d_plot.ylabel("D(E) average over gammaX");

    d_plot.legend().show(false);

    d_plot.drawCurve(d_xs, d_ys);

    d_plot.show();
}



std::tuple<ITensor, ITensor, ITensor, ITensor, ITensor> InitOpenLMG(int n, Args args = Args::global())
{
    auto kappa = args.getReal("Coupling", 1.0);

    auto H = InitCollectiveLMG(n, args);

    auto j = n/2;
    auto m = noPrime(*inds(H).begin());
    auto mP = prime(m);

    auto [Jz, Jp, Jm] = InitCollectiveOperators(j, m);
    auto Jmp = mapPrime(Jm * prime(Jp), 2, 1);

    // liouville * rho = L0 * rho + rho * R0 + (LJ * rho) * RJ
    auto L0 = -1_i * mapPrime(H, 1, 2) - 0.5_i * kappa / n * mapPrime(Jmp, 1, 2);
    auto R0 = -1_i * mapPrime(dag(H), 0, 3) - 0.5_i * kappa / n * mapPrime(Jmp, 0, 3);
    auto JL = kappa / n * mapPrime(Jp, 1, 2);
    auto JR = kappa / n * mapPrime(Jm, 0, 3);

    auto rho = ITensor(m, mP);
    rho.set(m = n + 1, mP = n + 1, 1);

    return std::make_tuple(L0, R0, JL, JR, rho);
}

void __TEST__OpenLMG()
{
    int N = 20;

    auto [L0, R0, JL, JR, rho] = InitOpenLMG(N, {
        "Magnetization=", 1.0,
        "Interaction=", 1.5,
        "Coupling=", 1.0
    });

    auto j = N/2;
    auto m = noPrime(*inds(rho).begin());
    auto mP = prime(m);

    auto [Jz, Jp, Jm] = InitCollectiveOperators(j, m);

    // FIXNE: rho becomes complex
    auto L = [&,L0,R0,JL,JR](ITensor rho) -> ITensor
    {
        return L0 * mapPrime(rho, 0, 2) + mapPrime(rho, 1, 3) * R0 + JL * prime(rho, 2) * JR;
    };

    auto Z = [&,Jz,j](ITensor rho) -> Complex
    {
        rho = mapPrime(Jz * prime(rho), 2, 1);
        return eltC(rho * delta(inds(rho))) / j;
    };

    Print(rho);
    Print(Z(rho));

    for (int i = 0; i < 10; ++i)
    {
        rho += 0.1 * L(rho);
        Print(rho);
        Print(Z(rho));
    }
}



void run_test(int test)
{
    switch(test)
        {
            case 1:
                __TEST__MPSDecompose();
                break;
            case 2:
                __TEST__FullLMG();
                break;
            case 3:
                __TEST__CollectiveLMG();
                break;
            case 4:
                __TEST__OpenLMG();
                break;
            default:
                printfln("unknown test case: %i", test);
        }
}

int main(int argc, char** argv)
{
    int test = 3;
    if (argc > 1)
        for (auto i : range1(argc - 1))
        {
            test = atoi(argv[i]);
            run_test(test);
        }
    else
        run_test(test);

    return 0;
}
