#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

#include "scattering.h"
#include "particle.h"
#include "init.h"
#include "forces.h"
#include "integrator.h"
#include "energy.h"
//#include "gr.h"
#include "timing.h"

using namespace std;

int main() {
timer.reset();   
    int N = 200;
    int steps = 500;
    double dt = 0.001;
	double g = 1;
    double A = g*g;
    double rc = 5.0;
	double alpha = 1.0;
	vector<int> Ns = {200, 1000, 5000};

for (int N : Ns) {

    cout << "Running simulation for N = " << N << endl;

    vector<Particle> particles(N);
    initialize(particles, N);

    // CEep mass fixed (important!)
    for (auto &p : particles)
        p.mass = 1.0;

    string energy_file = "energy_N_" + to_string(N) + ".dat";
    ofstream file(energy_file);

    compute_forces(particles, g, alpha, 1.0, rc);//DO WE NEED IT?

    for (int step = 0; step < steps; step++) {

        integrate(particles, dt);///commented
        compute_forces(particles, g, alpha, 4.0,rc);///THIS TOO
        finalize_velocity(particles, dt);//AND THIS????????????

        double CE = coulomb_potential(particles, g, rc);
        double PE = yukawa_potential(particles, g, alpha, rc);

        // ?? normalize per particle (VERY IMPORTANT)
        file << step << " "
             << CE/N << " "
             << PE/N << endl;
    }

    file.close();

    // RDF
//    string gr_file = "gr_N_" + to_string(N) + ".dat";
//    compute_gr(particles, 100, 5.0 , gr_file);
}
    // Different m values
    vector<double> m = {0.5, 1.0, 2.0, 4.0,10.0,50.0};

    for (double m : m) {
			double r = 0.0;

        cout << "Running simulation for m = " << m << endl;

        vector<Particle> particles(N);
        initialize(particles, N);

        // Energy output file
        string energy_file = "energy_m_" + to_string(m) + ".dat";
        ofstream file(energy_file);

        // Initial force
        compute_forces(particles, A, alpha, m, rc);

        for (int step = 0; step < steps; step++) {

            integrate(particles, dt);

            compute_forces(particles, A,alpha, m, rc);

            finalize_velocity(particles, dt);
			r += (dt*step);
            double CE = coulomb_potential(particles, A, rc);
			double PE = yukawa_potential(particles, A, alpha, rc);            double E  = CE + PE;

            file << r << " " << CE << " " << PE << endl;

            if (step % 100 == 0)
                cout << "Step: " << step <<" \t r= "<<r<<"\t CE= "<<CE<< endl;
        }

        file.close();

        // ?? Compute radial distribution function AFTER simulation
//        string gr_file = "gr_m_" + to_string(m) + ".dat";
//        compute_gr(particles, 100, 0.5, gr_file);
    }

    cout << "Simulation complete." << endl;

    vector<double> k_values = {0.5, 1.0, 2.0, 5.0};

    const double PI = 3.141592653589793;

    for (double k : k_values)
    {
        cout << "Running scattering for k = " << k << endl;

        string filename = "scattering_k_" + to_string(k) + ".dat";
        ofstream file(filename);

        for (double theta = 0.0; theta <= PI; theta += 0.01)
        {
            double f = yukawa_amplitude(g, alpha, k, theta);
            double ds = differential_cross_section(g, alpha, k, theta);

            file << theta << " " << f << " " << ds << endl;
        }

        file.close();
    }

    cout << "Scattering runs complete." << endl;
    timer.print();

    return 0;
}
