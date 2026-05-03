#include "io.h"

void write_energy(std::ofstream& file, int step, double KE,double PE) {
    file << step << " " << KE << "\n";
}
