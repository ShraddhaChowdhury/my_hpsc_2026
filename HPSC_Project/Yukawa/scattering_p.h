#ifndef SCATTERING_P_H
#define SCATTERING_P_H

#include <vector>

// differential cross section (optional if used separately)
double differential_cross_section(double g, double alpha, double k, double theta);

// main parallel routine
void compute_scattering(
    double g,
    double alpha,
    const std::vector<double>& k_vals,
    int Ntheta,
    std::vector<std::vector<double>>& dsigma
);

#endif
