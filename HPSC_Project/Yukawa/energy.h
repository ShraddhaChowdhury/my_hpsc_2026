#ifndef ENERGY_H
#define ENERGY_H

#include <vector>
#include "particle.h"

//double kinetic_energy(const std::vector<Particle>& p);
double yukawa_potential(const std::vector<Particle>& p, double g, double alpha, double rc);
double coulomb_potential(const std::vector<Particle>& p, double g, double rc);

#endif
