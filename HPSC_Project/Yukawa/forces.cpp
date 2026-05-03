#include "forces.h"
#include <cmath>
#include <omp.h>

using namespace std;

void compute_forces(vector<Particle>& p, double A,double alpha, double m, double rc) {
    int N = p.size();

    // Reset forces
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        p[i].force[0] = 0.0;
        p[i].force[1] = 0.0;
        p[i].force[2] = 0.0;
    }

    // Parallel force computation
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {

        double fx = 0.0, fy = 0.0, fz = 0.0;

        for (int j = 0; j < N; j++) {
            if (i == j) continue;

            double dx = p[i].pos[0] - p[j].pos[0];
            double dy = p[i].pos[1] - p[j].pos[1];
            double dz = p[i].pos[2] - p[j].pos[2];

            double r = sqrt(dx*dx + dy*dy + dz*dz + 1e-10);

            if (r > rc) continue;

            double exp_term = exp(-alpha*m * r);
            double factor = A * exp_term * (1.0/(r*r) + (alpha*m)/r);

            fx += factor * dx / r;
            fy += factor * dy / r;
            fz += factor * dz / r;
        }

        p[i].force[0] = fx;
        p[i].force[1] = fy;
        p[i].force[2] = fz;
    }
}
