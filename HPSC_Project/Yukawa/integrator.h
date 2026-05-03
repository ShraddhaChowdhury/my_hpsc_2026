#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <vector>
#include "particle.h"

void integrate(std::vector<Particle>& p, double dt);
void finalize_velocity(std::vector<Particle>& p, double dt);

#endif
