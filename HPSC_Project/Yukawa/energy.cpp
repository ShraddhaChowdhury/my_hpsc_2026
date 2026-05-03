#include "energy.h"
#include <cmath>
#include "timing.h"

using namespace std;

// ---------------- Potential Energy ----------------
double yukawa_potential(const vector<Particle>& p, double g, double alpha, double rc)
{
	timer.start("yukawa_energy");
    int N = p.size();
    double PE = 0.0;

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {

            double dx = p[i].pos[0] - p[j].pos[0];
            double dy = p[i].pos[1] - p[j].pos[1];
            double dz = p[i].pos[2] - p[j].pos[2];
            double r = sqrt(dx*dx + dy*dy + dz*dz + 1e-12);

            if (r > rc) continue;

            double m_eff = p[i].mass;

            double exp_term = exp(-alpha * m_eff * r);

            double V = g * g * exp_term / r;

            PE -= V;
        }
    }
	timer.stop("yukawa_energy");
    return PE;
}
double coulomb_potential(const vector<Particle>& p, double g, double rc)
{
		timer.start("coulomb_energy");

    int N = p.size();
    double CE = 0.0;

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {

            double dx = p[i].pos[0] - p[j].pos[0];
            double dy = p[i].pos[1] - p[j].pos[1];
            double dz = p[i].pos[2] - p[j].pos[2];
            double r = sqrt(dx*dx + dy*dy + dz*dz + 1e-12);

            if (r > rc) continue;

            double V = (g * g )/ r;

            CE -= V;
        }
    }
	timer.stop("coulomb_energy");

    return CE;
}
