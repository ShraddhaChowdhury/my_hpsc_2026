//#include "scattering.h"
//#include <cmath>
////double q = 2 * k * sin(theta / 2.0);
//// -------------------------------
//// Yukawa scattering amplitude
//// f(q) = g^2 / (q^2 + alpha^2)
//// -------------------------------
//
//double yukawa_amplitude(double g, double alpha, double k, double theta)
//{
//    double q = 2.0 * k * sin(theta / 2.0);
//    double denom = q*q + alpha*alpha;
//    return (g*g) / denom;
//}
//
//
//// -------------------------------
//// Differential cross-section
//// ds/dO = |f(q)|^2
//// -------------------------------
//// cross section
//double differential_cross_section(double g, double alpha, double k, double theta)
//{
//    double f = yukawa_amplitude(g, alpha, k, theta);
//    return f * f;
//}
#include "scattering.h"
#include "timing.h"
#include <cmath>

// -------------------------------
// Yukawa scattering amplitude
// f(q) = g^2 / (q^2 + alpha^2)
// -------------------------------
double yukawa_amplitude(double g, double alpha, double k, double theta)
{
    timer.start("yukawa_amplitude");

    double q = 2.0 * k * sin(theta / 2.0);
    double denom = q*q + alpha*alpha;
    double result = (g*g) / denom;

    timer.stop("yukawa_amplitude");
    return result;
}


// -------------------------------
// Differential cross-section
// ds/dO = |f(q)|^2
// -------------------------------
double differential_cross_section(double g, double alpha, double k, double theta)
{
    timer.start("differential_cross_section");

    // IMPORTANT:
    // call amplitude BUT don't time it twice artificially
    double f = yukawa_amplitude(g, alpha, k, theta);

    double result = f * f;

    timer.stop("differential_cross_section");
    return result;
}
