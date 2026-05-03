#include "init.h"
#include <random>

using namespace std;

void initialize(vector<Particle>& p, int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis_pos(0.0, 1.0);
    uniform_real_distribution<> dis_vel(-0.1, 0.1);

    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            p[i].pos[d] = dis_pos(gen);
            p[i].vel[d] = dis_vel(gen);
            p[i].force[d] = 0.0;
        }
        p[i].mass = 1.0;
    }
}
