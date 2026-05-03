#ifndef SCATTERING_H
#define SCATTERING_H

#include <vector>

// amplitude depends on k now
double yukawa_amplitude(double g, double alpha, double k, double theta);

// differential cross section
double differential_cross_section(double g, double alpha, double k, double theta);

#endif
