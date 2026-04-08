#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// ---------------- PARAMETERS ----------------
 int N = 200;
const double dt = 1e-4;
int steps = 5000;

const double Lx = 1.0, Ly = 2.0, Lz = 1.0;

const double kn = 10e5;
const double gamma_n = 10;
const double g = 9.81;

// ---------------- VECTOR STRUCT ----------------
struct Vec3 {
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

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
    for (int i = 0; i<p.size(); i++) {
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
    for (auto &pi : p)
        pi.f.z += pi.m * g;
}

// ---------------- PARTICLE CONTACT ----------------
void compute_contacts(vector<Particle>& p) {
	 int n = p.size();
    for (int i = 0; i<n; i++) {
        for (int j = i+1; j < n; j++) {

            Vec3 rij = p[j].x - p[i].x;
            double dij = norm(rij);

            double deltaij = p[i].R + p[j].R - dij;

            if (deltaij > 0) {
                Vec3 nij = rij * (1.0 / dij);

                Vec3 vij = p[j].v - p[i].v;
                double vn = dot(vij, nij);

                double Fn = kn * deltaij - gamma_n * vn;
                Fn = max(0.0, Fn);

                Vec3 F = nij * Fn;

                p[i].f = p[i].f - F;
                p[j].f += F;
            }
        }
    }
}

// ---------------- WALL CONTACT ----------------
void compute_wall_contacts(vector<Particle>& p) {
    for (auto &pi : p) {

        double delta = pi.R - pi.x.z;
        if (delta > 0) {
            double vn = -pi.v.z;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            pi.f.z += Fn;
        }

        delta = pi.x.z + pi.R - Lz;
        if (delta > 0) {
            double vn = pi.v.z;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            pi.f.z -= Fn;
        }

        delta = pi.R - pi.x.x;
        if (delta > 0) {
            double vn = -pi.v.x;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            pi.f.x += Fn;
        }

        delta = pi.x.x + pi.R - Lx;
        if (delta > 0) {
            double vn = pi.v.x;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            pi.f.x -= Fn;
        }

        delta = pi.R - pi.x.y;
        if (delta > 0) {
            double vn = -pi.v.y;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            pi.f.y += Fn;
        }

        delta = pi.x.y + pi.R - Ly;
        if (delta > 0) {
            double vn = pi.v.y;
            double Fn = kn * delta - gamma_n * vn;
            Fn = max(0.0, Fn);
            pi.f.y -= Fn;
        }
    }
}

// ---------------- INTEGRATION ----------------
void integrate(vector<Particle>& p) {
    for (auto &pi : p) {
        pi.v += pi.f * (dt / pi.m);
        pi.x += pi.v * dt;
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

// ---------------- TESTS ----------------
void test_free_fall() {
    cout << "\nRunning Free Fall Test...\n";

    Particle p;
    p.x = {0.5,0.5,1.0};
    p.v = {0, 0, 0};
    p.m = 1.0;
    p.R = 0.05;

    ofstream file("free_fall.txt");

    double t = 0;

    for (int step = 0; step < 5000; step++) {
        Vec3 F = {0, 0, -g * p.m};

        p.v += F * (dt / p.m);
        p.x += p.v * dt;

        double z_exact = 1.0 - 0.5 * 9.81 * t * t;

        file << t << " " << p.x.z << " " << z_exact << "\n";

        t += dt;
    }

    file.close();
}

void test_constant_velocity() {
    cout << "Running Constant Velocity Test...\n";

    Particle p;
    p.x = {0.2, 0.2, 0.2};
    p.v = {1.0, 0.5, -0.2};
    p.m = 1.0;
    p.R = 0.05;

    ofstream file("constant_velocity.txt");

    double t = 0;

    for (int step = 0; step < 5000; step++) {
        Vec3 F = {0,0,0};

        p.v += F * (dt / p.m);
        p.x += p.v * dt;

        file << t << " " << p.x.x << " " << p.x.y << " " << p.x.z << "\n";

        t += dt;
    }

    file.close();
}

void test_bounce() {
    cout << "Running Bounce Test...\n";

    Particle p;
    p.x = {0.5,0.5,1.0};
    p.v = {0, 0, 0};
    p.m = 1.0;
    p.R = 0.05;

    ofstream file("bounce.txt");

    double t = 0;

    for (int step = 0; step < 5000; step++) {
        Vec3 F = {0, 0, -g * p.m};

        double delta = p.R - p.x.z;

        if (delta > 0) {
            double vn = p.v.z;
            double Fn = max(0.0, kn*delta - gamma_n*vn);
            F.z += Fn;
        }

        p.v += F * (dt / p.m);
        p.x += p.v * dt;

        file << t << " " << p.x.z << "\n";

        t += dt;
    }

    file.close();
}
void run_simulation(int N) {

    vector<Particle> p(N);
    initialize(p);

    ofstream ke_file("ke_" + to_string(N) + ".dat");

    for (int step = 0; step < steps; step++) {

        zero_forces(p);
        add_gravity(p);
        compute_contacts(p);
        compute_wall_contacts(p);
        integrate(p);

        if (step % 100 == 0) {
            double ke = kinetic_energy(p);
            ke_file << step * dt << " " << ke << "\n";
        }
    }

    ke_file.close();
}

double t_total = 0;
double t_contacts = 0;
double t_walls = 0;
double t_integrate = 0;
double t_gravity = 0;

int main() {
vector<Particle> p(N);
    initialize(p);
    srand(time(0));
ofstream ke_file("ke_" + to_string(N) + ".dat");
    vector<int> N_values = {200, 1000, 5000};

    for (int N : N_values) {

        cout << "Running SERIAL for N = " << N << endl;

        auto start = high_resolution_clock::now();

        run_simulation(N);

        auto end = high_resolution_clock::now();

        double t = duration<double>(end - start).count();

        cout << "Time for N=" << N << " : " << t << " s\n";
    }
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
	 ke_file.close();
}

auto sim_end = high_resolution_clock::now();
t_total = duration<double>(sim_end - sim_start).count();
  
   cout << "\nAll tests completed.\n";

cout << "\n===== PROFILING RESULTS =====\n";

cout << "Total runtime: " << t_total << " s\n";

cout << "Gravity time: " << t_gravity << " s ("
     << 100*t_gravity/t_total << "%)\n";

cout << "Contact time: " << t_contacts << " s ("
     << 100*t_contacts/t_total << "%)\n";

cout << "Wall time: " << t_walls << " s ("
     << 100*t_walls/t_total << "%)\n";

cout << "Integration time: " << t_integrate << " s ("
     << 100*t_integrate/t_total << "%)\n";
   
    // Tests (verification)
    test_free_fall();
    test_constant_velocity();
    test_bounce();

    cout << "\nAll simulations completed.\n";

    return 0;
}
