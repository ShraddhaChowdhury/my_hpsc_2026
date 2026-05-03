#include <iostream>
#include <vector>
#include <omp.h>
#include <omp.h>
#include <fstream>
#include "particle.h"
#include "init.h"
#include "energy.h"
#include "scattering_p.h"
#include "timing.h"

using namespace std;

int main()
{
	auto t_start = std::chrono::high_resolution_clock::now();
    
	int N = 1000;
    double g = 1.0;
    double alpha = 1.0;
    double rc = 5.0;
int p = omp_get_max_threads();   // ? define p
    cout << "Threads: " << omp_get_max_threads() << endl;

    // ---------------- PARTICLES ----------------
    vector<Particle> particles(N);
    initialize(particles, N);

    for (auto &p : particles)
        p.mass = 1.0;

    // ---------------- ENERGY TIMING ----------------
    timer.reset();

    timer.start("yukawa_energy");
    double PE = yukawa_potential(particles, g, alpha, rc);
    timer.stop("yukawa_energy");

    timer.start("coulomb_energy");
    double CE = coulomb_potential(particles, g, rc);
    timer.stop("coulomb_energy");

    cout << "PE: " << PE << "  CE: " << CE << endl;

    timer.print();

    // ---------------- SCATTERING ----------------
    vector<double> k_vals = {0.5, 1.0, 2.0, 5.0};
    int Ntheta = 2000;

    vector<vector<double>> dsigma;

    timer.reset();

    timer.start("scattering");
    compute_scattering(g, alpha, k_vals, Ntheta, dsigma);
    timer.stop("scattering");

    timer.print();

// ----- total runtime -----
auto t_end = std::chrono::high_resolution_clock::now();

double total_time =
    std::chrono::duration<double>(t_end - t_start).count();

// ----- number of threads -----
//int p = omp_get_max_threads();

// ----- write to file -----
std::ofstream outfile("timing.dat", std::ios::app);

outfile << p << " " << total_time << std::endl;

outfile.close();

// optional print
std::cout << "\nTOTAL RUNTIME: " << total_time << " s\n";
    return 0;
}
