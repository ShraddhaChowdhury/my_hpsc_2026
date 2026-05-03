#ifndef PARTICLE_H
#define PARTICLE_H

struct Particle {
    double pos[3];
    double vel[3];
    double force[3];
    double mass;
};

#endif
