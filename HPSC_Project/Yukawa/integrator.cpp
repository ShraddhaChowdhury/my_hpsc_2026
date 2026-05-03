#include "integrator.h"

using namespace std;

void integrate(vector<Particle>& p, double dt) {
    for (auto &pt : p) {
        for (int d = 0; d < 3; d++) {
            pt.vel[d] += 0.5 * pt.force[d] / pt.mass * dt;
            pt.pos[d] += pt.vel[d] * dt;
        }
    }
}

void finalize_velocity(vector<Particle>& p, double dt) {
    for (auto &pt : p) {
        for (int d = 0; d < 3; d++) {
            pt.vel[d] += 0.5 * pt.force[d] / pt.mass * dt;
        }
    }
}
