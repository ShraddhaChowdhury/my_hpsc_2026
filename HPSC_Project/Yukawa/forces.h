#ifndef FORCES_H
#define FORCES_H

#include <vector>
#include "particle.h"

void compute_forces(std::vector<Particle>& p, double A, double g, double m, double rc);

#endif
