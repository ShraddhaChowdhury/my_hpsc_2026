//#ifndef TIMING_H
//#define TIMING_H
//
//#include <vector>
//#include "particle.h"
//
//struct TimingResult {
//    double total_time;
//    double force_time;
//    double integrate_time;
//    double velocity_time;
//};
//
//TimingResult run_timed_simulation(
//    int N,
//    int steps,
//    double dt,
//    double g,
//    double alpha,
//    double mass,
//    double rc
//);
//
//#endif
#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <map>
#include <string>
#include <iostream>

class Timer {
public:
    void start(const std::string& name) {
        start_times[name] = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string& name) {
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start_times[name]).count();
        timings[name] += elapsed;
    }

    void reset() {
        timings.clear();
    }

    void print() const {
        std::cout << "\n==== Timing Report ====\n";
        for (const auto& t : timings) {
            std::cout << t.first << " : " << t.second << " s\n";
        }
    }

    const std::map<std::string, double>& get() const {
        return timings;
    }

private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::map<std::string, double> timings;
};
extern Timer timer;
#endif
