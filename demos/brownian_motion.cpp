#include "../basic/MatrixCpp.hpp"
#include "../shared/plot_gpu.hpp"

using namespace mcpu;  // the package lives in mcpu; mgpu is its GPU twin
using namespace std;

// System Constants in SI UNITS
// For water at ~20C
const double part_radius = 1e-7;                                                   // meters
const double part_mass = 1e-17;                                                    // kilograms
const double part_vol = (part_radius * part_radius * part_radius * 4 * M_PI / 3);  // m^3
const double g = 9.81;                                                             // m/s^2
const double kB = 1.380649e-23;                                                    // J/K
const double T = 300;                                                              // Kelvin
const double density_f = 1000;                                                     // kg/m^3
const double density_p = part_mass / part_vol;                                     // kg/m^3
const double viscosity = 0.001;                                                    // Pa*s
const double drag = 6 * M_PI * viscosity * part_radius;                            // kg/s
const double D_0 = kB * T / drag;                                                  // m^2/s

// ── Dimensionless units ─────────────────────────────────────────────────────
//
// The constants above are SI; everything the simulation actually manipulates is
// scaled, so there is exactly one place where the conversion happens and no
// mixed-unit arithmetic anywhere else. The scales are the natural ones for a
// diffusing colloid:
//
//     length   a  = part_radius
//     time     tau = a^2 / D_0        (how long diffusion takes to cover a)
//     energy   kB * T
//
// Everything else follows from those three:
//
//     dt~        = dt * D_0 / a^2                 <- the file's diffusive_time
//     Brownian   dx~ = sqrt(2 dt~) * xi           xi of unit variance
//     any force  dx~ = F~ dt~   with  F~ = F a / (kB T)
//
// The last line is the one that was wrong: a force becomes a displacement by
// multiplying by dt~, and it is made dimensionless by a/(kB T), NOT by
// D_0/(kB T). Those differ by D_0/a^2 = 220, which is how much too fast the
// particles were sinking.

// One N(0,1) deviate drawn from a PERSISTENT ran2 stream.
//
// ran2 keeps its entire state in the variable you hand it and advances it on
// every call, so the seed must be threaded through by pointer and never
// re-initialised. That is exactly what went wrong before: both
// set_Ran_values(lo, hi, seed) and randn(..., seed) copy the seed into a local
// and start a fresh sequence, so calling either in a loop returns the SAME
// number every time -- every particle moved in lockstep.
//
// Box-Muller produces two deviates per pair of uniforms; the spare is kept
// rather than thrown away, which also keeps a given seed reproducible.
double gauss(long* idum) {
    static bool have_spare = false;
    static double spare = 0.0;
    if (have_spare) {
        have_spare = false;
        return spare;
    }
    double u1 = double(ran2(idum));
    const double u2 = double(ran2(idum));
    if (u1 < 1e-300)
        u1 = 1e-300;  // log(0) guard
    const double r = sqrt(-2.0 * log(u1));
    const double theta = 2.0 * M_PI * u2;
    spare = r * sin(theta);
    have_spare = true;
    return r * cos(theta);
}

// Buoyant weight of one particle, made dimensionless: F_g * a / (kB T).
// About 1.4e-3, so gravity is weak next to diffusion at this size -- which is
// the whole reason a 100 nm particle stays suspended.
const double grav_number = (density_p - density_f) * part_vol * g * part_radius / (kB * T);
// Lennard-Jones FORCE, dimensionless: F * a / (kB T). Positions come in scaled
// by the particle radius, so sigma is 1 and every length here is a pure number.
// The caller turns this into a displacement by multiplying by dt~.
Matrix<double> LJ(const Matrix<double>& particle, const Matrix<double>& container) {
    double epsilon = 1.0;  // depth of the potential well, in units of kB*T
    double cutoff = 2.5;   // in units of sigma, which is 1 here
    double wall_cutoff = 10.0;
    Matrix<double> force(particle.rows(), particle.cols());
    for (auto i = 0; i < particle.rows(); i++) {
        // calculates the force between particle i and all other particles j
        for (auto j = i + 1; j < particle.rows(); j++) {
            double r = sqrt(sum((particle(i, all) - particle(j, all)).pow(2), ROW));
            if (r < cutoff) {
                // F = -dU/dr = (24 eps / sigma) [2 (sigma/r)^13 - (sigma/r)^7].
                // With the 13/7 exponents the prefactor is 24 eps / SIGMA; the
                // 24 eps / r form goes with the 12/6 exponents. Mixing the two
                // left an extra factor of 1/r.
                double F = 24 * epsilon * (2 * pow(1.0 / r, 13) - pow(1.0 / r, 7));
                force(i, all) += F * (particle(i, all) - particle(j, all)) / r;
                force(j, all) -= F * (particle(i, all) - particle(j, all)) / r;
            }
        }
        // calculates the force between particle i and the container walls
        //  dim x, y, z
        for (int k : {0, 1, 2}) {
            if (particle(i, k) < wall_cutoff) {
                double x_k = particle(i, k);
                force(i, k) += 24 * epsilon * (2 * pow(1.0 / x_k, 13) - pow(1.0 / x_k, 7));
            } else if (particle(i, k) > container[k] - wall_cutoff) {
                double x_k = container[k] - particle(i, k);
                // MINUS: the far wall pushes back toward the origin. Both
                // branches added before, so the upper wall attracted.
                force(i, k) -= 24 * epsilon * (2 * pow(1.0 / x_k, 13) - pow(1.0 / x_k, 7));
            }
        }
    }
    return force;
}
int main() {
    // inital parameters
    const unsigned int N = 100;      // number of particles;
    Matrix<double> particles(N, 3);  // coordinates of particles, cols are x,y,z
    // for a cartesian container, assumes the corner is at (0,0,0) and grows in the postive
    // direction
    // dimensions of the container (x,y,z), in PARTICLE RADII -- the same units
    // the positions are kept in, so no conversion happens after this point
    Matrix<double> container_dims_cart(3, 1);
    container_dims_cart[0] = 100.0;   // x dimension
    container_dims_cart[1] = 100.0;   // y dimension
    container_dims_cart[2] = 1000.0;  // z dimension
    long seed;
    setRan(seed);  // seeds the stream ONCE; ran2 advances it from here on
    // initial coordinates, drawn from that one stream.
    //
    // Three separate set_Ran_values(lo, hi) calls will NOT do: each derives a
    // fresh seed from the clock to whole-second resolution, so all three get
    // the same sequence and every particle lands on the diagonal
    // x = y = z/10 -- a dead straight line, plainly visible in the first frame.
    for (auto i = 0; i < N; i++)
        for (int k : {0, 1, 2})
            particles(i, k) = double(ran2(&seed)) * container_dims_cart[k];
    for (auto i = 0; i < N; i++) {
        for (auto j = i + 1; j < N; j++) {
            // incase the particles are too close to each other
            double script_r = sqrt(sum((particles(i, all) - particles(j, all)).pow(2), ROW));
            while (script_r < 0.5) {
                particles(j, 0) = set_Ran_values(0.0, container_dims_cart[0]);
                particles(j, 1) = set_Ran_values(0.0, container_dims_cart[1]);
                particles(j, 2) = set_Ran_values(0.0, container_dims_cart[2]);
                script_r = sqrt(sum((particles(i, all) - particles(j, all)).pow(2), ROW));
            }
        }
    }
    double dt = 1e-5;  // time step
    double total_time = 10;
    double diffusive_time = D_0 * dt / (part_radius * part_radius);  // dimensionless time step
    int total_time_steps = int(total_time / dt);
    Matrix<double> displacement(N, 3);  // displacement of particles, cols are x,y,z
    const int frame = total_time_steps / 20000;
    for (int t = 0; t < total_time_steps; t++) {
        // plotting
        if (t % frame == 0) {
            // positions are already in particle radii, so nothing is scaled
            // here; the casts just materialise the slices for scatter3()
            plt::scatter3(Matrix<double>(particles(all, 0)),
                          Matrix<double>(particles(all, 1)),
                          Matrix<double>(particles(all, 2)));
            // fixed axes AND a fixed camera, or Plots rescales and re-aims each
            // frame and the container appears to move instead of the particles
            plt::xlim(0.0, container_dims_cart[0]);
            plt::ylim(0.0, container_dims_cart[1]);
            plt::zlim(0.0, container_dims_cart[2]);
            plt::view(45.0, 20.0);
            plt::xlabel("x  (particle radii)");
            plt::ylabel("y  (particle radii)");
            plt::zlabel("z  (particle radii)");
            char ttl[64];
            std::snprintf(ttl, sizeof(ttl), "t = %.4f s", t * dt);
            plt::title(ttl);
            plt::frame();
        }

        // update the particles -- every term below is a displacement in
        // particle radii, so they can simply be added together
        //
        // a force becomes a displacement by multiplying by the timestep
        displacement = LJ(particles, container_dims_cart) * diffusive_time;
        for (auto i = 0; i < N; i++) {
            // Brownian motion: dx~ = sqrt(2 dt~) * xi, xi ~ N(0,1).
            // The coefficient assumes UNIT VARIANCE, which is why this is a
            // Gaussian and not the old uniform on [-1,1] -- that has variance
            // 1/3 and so diffused at a third of the intended rate.
            Matrix<double> dW(1, 3);
            for (int k : {0, 1, 2})
                dW(0, k) = gauss(&seed);
            displacement(i, all) += sqrt(2 * diffusive_time) * dW;
        }
        // buoyant weight, already dimensionless, times the timestep
        displacement(all, 2) -= grav_number * diffusive_time;
        particles += displacement;
    }
    plt::gif("demos/out/brownian.gif", 20);
}