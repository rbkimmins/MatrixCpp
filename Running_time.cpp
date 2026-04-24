#include "Matrix1.0.hpp"
#include <fstream>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main(){
    // Max square matrix size to test (inclusive).
    // Warn: matrix multiply is O(n^3) — n=500 already takes a few seconds.
    long n = 300;

    ofstream outFile("running_time.txt");
    if (!outFile) {
        cerr << "Error: could not open running_time.txt for writing\n";
        return 1;
    }

    outFile << "size,time_seconds\n";

    for (long i = 2; i <= n; i++) {
        // Fresh random matrices for each size — avoid timing allocation by
        // constructing and populating before the clock starts.
        Matrix<double> A(i, i);
        Matrix<double> B(i, i);
        A.set_Ran_values(-1000.0, 1000.0);
        B.set_Ran_values(-1000.0, 1000.0);

        auto start = high_resolution_clock::now();
        Matrix<double> C = A * B;
        auto stop  = high_resolution_clock::now();

        double seconds = duration<double>(stop - start).count();
        outFile << i << "," << seconds << "\n";

        cout << "i=" << i << "  t=" << seconds << "s\n";
    }

    outFile.close();
    cout << "Results written to running_time.txt\n";
    return 0;
}
