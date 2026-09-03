#include "basic/MatrixCpp.hpp"
using namespace std;

int main() {
    // inital parameters
    const unsigned int N = 100;      // number of particles;
    Matrix<double> particles(N, 3);  // coordinates of particles, cols are x,y,z
    // for a cartesian container, assumes the corner is at (0,0,0) and grows in the postive
    // direction
    Matrix<double> container_dims_cart(3, 1);  // dimensions of the container (x,y,z)
    container_dims_cart[0] = 10.0;             // x dimension
    container_dims_cart[1] = 10.0;             // y dimension
    container_dims_cart[2] = 100.0;            // z dimension
    // initial coordinates
    particles(all, 0).set_Ran_values(0, container_dims_cart[0]);
    particles(all, 1).set_Ran_values(0, container_dims_cart[1]);
    particles(all, 2).set_Ran_values(0, container_dims_cart[2]);
    for (auto i = 0; i < N; i++) {
        for (auto j = i + 1; j < N; j++) {
            // incase the particles are too close to each other
            double r = sqrt(sum((particles(i, all) - particles(j, all)).pow(2), ROW));
            while (r < 0.5) {
                particles(j, 0) = set_Ran_values(0.0, container_dims_cart[0]);
                particles(j, 1) = set_Ran_values(0.0, container_dims_cart[1]);
                particles(j, 2) = set_Ran_values(0.0, container_dims_cart[2]);
                r = sqrt(sum((particles(i, all) - particles(j, all)).pow(2), ROW));
            }
        }
    }
}