#include "timing.h"
//#include "init.h"
//#include "forces.h"
//#include "integrator.h"
Timer timer;
//#include <chrono>
//
//using namespace std;
//
//TimingResult run_timed_simulation(
//    int N,
//    int steps,
//    double dt,
//    double g,
//    double alpha,
//    double mass,
//    double rc
//) {
//    vector<Particle> particles(N);
//    initialize(particles, N);
//
//    for (auto &p : particles)
//        p.mass = mass;
//
//    double force_time = 0.0;
//    double integrate_time = 0.0;
//    double velocity_time = 0.0;
//
//    auto start_total = chrono::high_resolution_clock::now();
//
//    compute_forces(particles, g, alpha, mass, rc);
//
//    for (int step = 0; step < steps; step++) {
//
//        auto t1 = chrono::high_resolution_clock::now();
//
//        integrate(particles, dt);
//
//        auto t2 = chrono::high_resolution_clock::now();
//
//        compute_forces(particles, g, alpha, mass, rc);
//
//        auto t3 = chrono::high_resolution_clock::now();
//
//        finalize_velocity(particles, dt);
//
//        auto t4 = chrono::high_resolution_clock::now();
//
//        integrate_time += chrono::duration<double>(t2 - t1).count();
//        force_time     += chrono::duration<double>(t3 - t2).count();
//        velocity_time  += chrono::duration<double>(t4 - t3).count();
//    }
//
//    auto end_total = chrono::high_resolution_clock::now();
//
//    double total_time = chrono::duration<double>(end_total - start_total).count();
//
//    TimingResult result;
//    result.total_time = total_time;
//    result.force_time = force_time;
//    result.integrate_time = integrate_time;
//    result.velocity_time = velocity_time;
//
//    return result;
//}
