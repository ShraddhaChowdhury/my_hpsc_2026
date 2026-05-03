#include "energy.h"
#include "particle.h"
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

// ---------------- Yukawa Potential ----------------
double yukawa_potential(const vector<Particle>& p, double g, double alpha, double rc)
{
    int N = p.size();
    double PE = 0.0;
    double rc2 = rc * rc;

    #pragma omp parallel for reduction(+:PE)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {

            double dx = p[i].pos[0] - p[j].pos[0];
            double dy = p[i].pos[1] - p[j].pos[1];
            double dz = p[i].pos[2] - p[j].pos[2];

            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > rc2) continue;

            double r = sqrt(r2 + 1e-12);

            double V = g*g * exp(-alpha * r) / r;
            PE -= V;
        }
    }

    return PE;
}

// ---------------- Coulomb Potential ----------------
double coulomb_potential(const vector<Particle>& p, double g, double rc)
{
    int N = p.size();
    double CE = 0.0;
    double rc2 = rc * rc;

    #pragma omp parallel for reduction(+:CE)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {

            double dx = p[i].pos[0] - p[j].pos[0];
            double dy = p[i].pos[1] - p[j].pos[1];
            double dz = p[i].pos[2] - p[j].pos[2];

            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > rc2) continue;

            double r = sqrt(r2 + 1e-12);

            double V = g*g / r;
            CE -= V;
        }
    }

    return CE;
}
