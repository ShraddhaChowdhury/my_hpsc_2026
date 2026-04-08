#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <omp.h>
using namespace std::chrono; //Code Profiling
using namespace std;

// ---------------- PARAMETERS ----------------
const int N = 200;          // number of particles
const double dt = 1e-4;
const int steps = 5000;

const double Lx = 1.0, Ly = 2.0, Lz = 1.0;

const double kn = 10e5;      // stiffness
const double gamma_n = 10;  // damping
const double g = 9.81;

// ---------------- VECTOR STRUCT ----------------
struct Vec3 {
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_)  : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& b) const {
        return Vec3(x + b.x, y + b.y, z + b.z);
    }

    Vec3 operator-(const Vec3& b) const {
        return Vec3(x - b.x, y - b.y, z - b.z);
    }

    Vec3 operator*(double s) const {
        return Vec3(x * s, y * s, z * s);
    }

    Vec3& operator+=(const Vec3& b) {
        x += b.x; y += b.y; z += b.z;
        return *this;
    }
};

double dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

double norm(const Vec3& a) {
    return sqrt(dot(a,a));
}

// ---------------- PARTICLE STRUCT ----------------
struct Particle {
    Vec3 x, v, f;
    double m, R;
};
double rand01() {
    return rand() / (double)RAND_MAX;
}
// ---------------- INITIALIZATION ----------------
void initialize(vector<Particle>& p) {
    for (int i = 0; i < N; i++) {
       p[i].x = Vec3{rand01()*Lx, rand01()*Ly, rand01()*Lz};
        p[i].v = Vec3(0,0,0);
        p[i].f = Vec3(0,0,0);
        p[i].m = 1.0;
        p[i].R = 0.05;
    }
}
// ---------------- ZERO FORCES ----------------
void zero_forces(vector<Particle>& p) {
    for (auto &pi : p)
        pi.f = Vec3(0,0,0);
}

// ---------------- GRAVITY ----------------
void add_gravity(vector<Particle>& p) {
    #pragma omp parallel for
	for (int i = 0; i < N; i++)
    	p[i].f.z += -p[i].m * g;
}


void compute_contacts(vector<Particle>& p) {

    int nthreads = omp_get_max_threads();

    // Thread-local force arrays
    vector<vector<Vec3>> f_private(nthreads, vector<Vec3>(N, Vec3(0,0,0)));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            for (int j = i+1; j < N; j++) {

                Vec3 rij = p[j].x - p[i].x;
                double dij = norm(rij);

                if (dij < 1e-12) continue;

                double deltaij = p[i].R + p[j].R - dij;

                if (deltaij > 0) {

                    Vec3 nij = rij * (1.0 / dij);

                    Vec3 vij = p[j].v - p[i].v;
                    double vn = dot(vij, nij);

                    double Fn = kn * deltaij - gamma_n * vn;
                    Fn = max(0.0, Fn);

                    Vec3 F = nij * Fn;

                    f_private[tid][i] = f_private[tid][i] - F;
                    f_private[tid][j] += F;
                }
            }
        }
    }

    // Combine forces
    for (int i = 0; i < N; i++) {
		for (int t = 0; t < nthreads; t++) {
        
            p[i].f += f_private[t][i];
        }
    }
}
// ---------------- WALL CONTACT ----------------
void compute_wall_contacts(vector<Particle>& p) {
    #pragma omp parallel for
	for (int i = 0; i < N; i++) {

        // Z lower wall
        double delta = p[i].R - p[i].x.z;
        if (delta > 0) {
            double vn = -p[i].v.z;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            p[i].f.z += Fn;
        }

        // Z upper wall
        delta = p[i].x.z + p[i].R - Lz;
        if (delta > 0) {
            double vn = p[i].v.z;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            p[i].f.z -= Fn;
        }

        // X walls
        delta = p[i].R - p[i].x.x;
        if (delta > 0) {
            double vn = -p[i].v.x;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            p[i].f.x += Fn;
        }

        delta = p[i].x.x + p[i].R - Lx;
        if (delta > 0) {
            double vn = p[i].v.x;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            p[i].f.x -= Fn;
        }

        // Y walls
        delta = p[i].R - p[i].x.y;
        if (delta > 0) {
            double vn = -p[i].v.y;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            p[i].f.y += Fn;
        }

        delta = p[i].x.y + p[i].R - Ly;
        if (delta > 0) {
            double vn = p[i].v.y;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            p[i].f.y -= Fn;
        }
    }
}

// ---------------- INTEGRATION ----------------
void integrate(vector<Particle>& p) {
    #pragma omp parallel for
	for (int i = 0; i < N; i++) {
    	p[i].v += p[i].f * (dt / p[i].m);
    	p[i].x += p[i].v * dt;
}
}

// ---------------- DIAGNOSTICS ----------------
double kinetic_energy(const vector<Particle>& p) {
    double ke = 0;
    for (auto &pi : p)
        ke += 0.5 * pi.m * dot(pi.v, pi.v);
    return ke;
}

// ---------------- OUTPUT ----------------
void write_output(const vector<Particle>& p, int step) {
    ofstream file("output_" + to_string(step) + ".dat");
    for (auto &pi : p)
        file << pi.x.x << " " << pi.x.y << " " << pi.x.z << "\n";
}


double t_total = 0;
double t_contacts = 0;
double t_walls = 0;
double t_integrate = 0;
double t_gravity = 0;

int main() {
	// -------- MAIN SIMULATION --------
    int steps = 5000;
	srand(time(0));
    vector<Particle> p(N);
    initialize(p);
ofstream ke_file("ke_parallel.dat");

auto sim_start = high_resolution_clock::now();

for (int step = 0; step < steps; step++) {

    zero_forces(p);

    auto t1 = high_resolution_clock::now();
    
	add_gravity(p);
    auto t2 = high_resolution_clock::now();
    t_gravity += duration<double>(t2 - t1).count();

    t1 = high_resolution_clock::now();
    
	compute_contacts(p);
    t2 = high_resolution_clock::now();
    t_contacts += duration<double>(t2 - t1).count();

    t1 = high_resolution_clock::now();
    
	compute_wall_contacts(p);
    t2 = high_resolution_clock::now();
    t_walls += duration<double>(t2 - t1).count();

    t1 = high_resolution_clock::now();
    
	integrate(p);
    t2 = high_resolution_clock::now();
    t_integrate += duration<double>(t2 - t1).count();

    if (step % 100 == 0) {
        double ke = kinetic_energy(p);
		write_output(p, step);
    ke_file << step * dt << " " << ke << "\n";
	}
}

auto sim_end = high_resolution_clock::now();
t_total = duration<double>(sim_end - sim_start).count();
  
cout << "Total runtime: " << t_total << " s\n";

int num_threads;

#pragma omp parallel
{
    #pragma omp single
    num_threads = omp_get_num_threads();
}

ofstream perf_file("performance.dat", ios::app);
perf_file << num_threads << " " << t_total << "\n";
perf_file.close();
    return 0;
}
