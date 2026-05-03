#include "scattering.h"
#include <cmath>
#include <omp.h>

using namespace std;

// Yukawa amplitude
double yukawa_amplitude(double g, double alpha, double k, double theta)
{
    double q = 2.0 * k * sin(theta / 2.0);
    return (g*g) / (q*q + alpha*alpha);
}

// Parallel scattering computation
void compute_scattering(
    double g, double alpha,
    const vector<double>& k_vals,
    int Ntheta,
    vector<vector<double>>& dsigma)
{
    int Nk = k_vals.size();
    dsigma.assign(Nk, vector<double>(Ntheta));

    const double PI = 3.141592653589793;

    #pragma omp parallel for
    for (int i = 0; i < Nk; i++) {

        double k = k_vals[i];

        for (int j = 0; j < Ntheta; j++) {

            double theta = PI * j / Ntheta;

            double f = yukawa_amplitude(g, alpha, k, theta);
            dsigma[i][j] = f * f;
        }
    }
}
