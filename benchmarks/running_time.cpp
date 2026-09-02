// ══════════════════════════════════════════════════════════════════════════
//  MatrixCpp benchmark harness — real and complex
//
//  Times every operation the library supports, for BOTH Matrix<double> and
//  Matrix<complex<double>>, and DRAWS THE RESULTS ITSELF using the plotting
//  package. No Julia, no matplotlib, no intermediate file to hand to another
//  language.
//
//  It also writes one CSV per (dtype, operation) into bench/. Those exist for
//  ONE reason: numpy_timings.py reads them to learn which operations and which
//  sizes to time, and writes its own alongside. A cross-language comparison has
//  to persist data somewhere; a single-language plot does not, and no longer
//  does.
//
//      bench/cpp_<dtype>_<op>.csv      dtype in {real, complex}
//      columns: size,time_seconds
//
//  Build and run from the REPO ROOT:
//      g++ -std=c++17 -O3 -march=native -fopenmp -I. \
//          benchmarks/running_time.cpp -o running_time
//      ./running_time
//
//  Writes benchmarks/plots/speed_<dtype>_<group>.png.
//
//  Note on -ffast-math: it is deliberately NOT in the line above. It implies
//  -fcx-limited-range, which changes how complex multiply and divide are
//  evaluated, so complex timings taken with it are not comparable to NumPy's.
// ══════════════════════════════════════════════════════════════════════════
#include "plotting/MatrixPlot.hpp"

#include <chrono>
#include <set>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "Matrix1.0.hpp"
using namespace std;
using namespace std::chrono;

// ════════════════════════════════════════════════════════════════════════════
//  BENCHMARK SELECTION — comment out any line to skip that operation
// ════════════════════════════════════════════════════════════════════════════
#define BENCH_ELEMENTWISE  // add, subtract, hadamard, elem_div, scalar_mul/div
#define BENCH_MAPS         // elem_exp, elem_ln, elem_pow
#define BENCH_FFT          // fft                      O(n log n)
#define BENCH_SHAPE        // transpose, conj_transpose, reshape, triu, concat
#define BENCH_REDUCE       // sum_all, sum_dim, trace, norm_fro, isdiagonal
#define BENCH_MULTIPLY     // A * B                    O(n³)  Strassen-Winograd
#define BENCH_KRON         // A.kron(B)              O(n⁴)  Kronecker
#define BENCH_DECOMP       // det, inverse, solve, LU, QR, svd, cholesky   (real
                           // only)
#define BENCH_EIG  // eig                      O(n³)  iterative     (real only)
#define BENCH_MATFN  // mat_pow_int, mat_pow_real, mat_log, mat_exp   (real
                     // only)

// ════════════════════════════════════════════════════════════════════════════
//  SIZE LIMITS per complexity class. Sizes are log-spaced, not every integer:
//  a curve is defined by its shape, and 20 well-measured points show that far
//  better than 1500 noisy ones — and finish in a fraction of the time.
// ════════════════════════════════════════════════════════════════════════════
static const long N_LINEAR = 8000;  // O(n):  trace
static const long N_SQUARE =
    2000;                         // O(n²): element-wise, transpose, reductions
static const long N_CUBIC = 512;  // O(n³): multiply, LU, QR, det, inverse, svd
static const long N_ITER = 256;   // O(n³) iterative: eig, matrix pow/log/exp
static const long N_KRON = 48;    // O(n⁴): tensor — the result is n²×n²
static const int POINTS = 18;     // measurement points per complexity class

// Each point is measured repeatedly and the BEST time kept. Minimum-of-k is the
// standard choice for microbenchmarks: the true cost is a floor, and everything
// the OS scheduler, another process or a cache miss adds only ever pushes a
// sample above it. An average would report the noise as if it were the work.
static const double MIN_SECONDS =
    0.08;  // keep repeating until this much time has passed
static const int MAX_REPS = 200;
// Three warm-ups, not one. Several setups do a long SERIAL pass over the matrix
// before handing back the callable to time (elem_ln and elem_pow shift every
// element to keep the argument positive, for instance). By the time that
// finishes, the OpenMP thread pool has been idle long enough to go to sleep,
// and the first parallel region afterwards pays to wake 32 threads. With a
// single warm-up that cost lands inside the measured window often enough to
// publish a number ~8x too high — real elem_ln at n=2000 alternated between
// 0.63 ms and 5.7 ms across runs. Extra warm-ups cost microseconds and make the
// floor real.
static const int WARMUP = 3;

// ════════════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════════════
template <typename T>
struct dtype_traits;
template <>
struct dtype_traits<double> {
  static constexpr const char* name = "real";
};
template <>
struct dtype_traits<std::complex<double>> {
  static constexpr const char* name = "complex";
};

// Log-spaced sizes from 2 to nMax, de-duplicated and ascending.
static vector<long> logspace(long nMax, int points) {
  vector<long> out;
  for (int k = 0; k < points; k++) {
    double f = double(k) / double(points - 1);
    long v = (long)llround(2.0 * std::pow(double(nMax) / 2.0, f));
    if (v < 2) v = 2;
    if (out.empty() || v > out.back()) out.push_back(v);
  }
  return out;
}

// Deterministic fill. set_Ran_values() writes datatype(val), which for a
// complex matrix leaves every imaginary part at zero — that is not a complex
// matrix in any meaningful sense and would let a benchmark hit real-valued
// fast paths. So complex operands get both parts filled explicitly.
template <typename T>
static void fillRandom(Matrix<T>& A, long seed) {
  if constexpr (is_complex<T>::value) {
    Matrix<double> re(A.rows(), A.cols()), im(A.rows(), A.cols());
    re.set_Ran_values(-1.0, 1.0, seed);
    im.set_Ran_values(-1.0, 1.0, seed - 7919);
    for (long i = 0; i < A.rows() * A.cols(); i++)
      A[int(i)] = std::complex<double>(re[int(i)], im[int(i)]);
  } else {
    A.set_Ran_values(-1.0, 1.0, seed);
  }
}

// Diagonally dominant — non-singular, so det/inverse/solve/LU cannot blow up.
static Matrix<double> makeWellConditioned(long n, long seed) {
  Matrix<double> A(n, n);
  A.set_Ran_values(-1.0, 1.0, seed);
  for (long i = 0; i < n; i++) A(int(i), int(i)) += double(n);
  return A;
}

// Symmetric — guarantees real eigenvalues, which eig() now requires.
static Matrix<double> makeSymmetric(long n, long seed) {
  Matrix<double> A(n, n);
  A.set_Ran_values(-1.0, 1.0, seed);
  return (A + A.T()) * 0.5;
}

// Symmetric positive definite — needed by cholesky, and by pow(A,0.5)/log(A),
// which are only real-valued when every eigenvalue is positive.
static Matrix<double> makeSPD(long n, long seed) {
  Matrix<double> A = makeSymmetric(n, seed);
  for (long i = 0; i < n; i++) A(int(i), int(i)) += double(n) + 1.0;
  return A;
}

// Adaptive best-of-k timing, run as TWO independent bursts.
//
// Min-of-k inside a single burst removes ordinary jitter, but not a disturbance
// that lasts longer than the whole measurement window — and on a desktop with a
// browser and an editor running, a stall of a few tens of milliseconds is
// routine. When the op itself costs ~5 ms, MIN_SECONDS of sampling is only ~15
// reps, so one such stall can contaminate every sample and the published figure
// comes out 5-20x too high. (Observed: matrix exp at n=145 reported 87 ms
// against a true 4.5 ms.) Two separate bursts, taking the better, makes that
// require two stalls in a row — rare enough that the numbers stopped moving.
template <typename F>
static double timeOneBurst(F&& f) {
  double best = 1e300, total = 0.0;
  int reps = 0;
  while (reps < MAX_REPS && total < MIN_SECONDS) {
    auto t0 = high_resolution_clock::now();
    f();
    auto t1 = high_resolution_clock::now();
    double dt = duration<double>(t1 - t0).count();
    if (dt < best) best = dt;
    total += dt;
    reps++;
  }
  return best;
}

template <typename F>
static double timeBest(F&& f) {
  for (int w = 0; w < WARMUP; w++) f();
  const double a = timeOneBurst(f);
  const double b = timeOneBurst(f);
  return std::min(a, b);
}

static int written = 0;

// Every timed series is kept so the run can plot itself at the end. The CSV is
// still written, but only because numpy_timings.py needs it to match sizes.
struct Series {
  string dtype, op, group;
  vector<double> n, t;
};
static vector<Series> results;

// The group an operation belongs to, from its name. Deriving it here rather
// than passing it at each of the fifty-odd call sites keeps the benchmark list
// readable and means a new operation lands in the right figure automatically.
static string groupOf(const string& op) {
  auto has = [&](const char* s) { return op.find(s) != string::npos; };
  if (has("lu") || has("qr") || has("svd") || has("eig") || has("det") ||
      has("inverse") || has("cholesky") || has("schur") || has("solve") ||
      has("rank") || has("pinv"))
    return "factorisations";
  if (has("mat_pow") || has("expm") || has("logm") || has("sqrtm") || has("funm"))
    return "matrix_functions";
  if (has("sum") || has("trace") || has("norm") || has("mean") || has("min") ||
      has("max") || has("prod"))
    return "reductions";
  if (has("transpose") || has("concat") || has("isdiagonal") || has("conj") ||
      has("diag") || has("reshape"))
    return "structure";
  return "elementwise";
}

// Runs one operation across its size range and writes
// bench/cpp_<dtype>_<op>.csv. `body` receives the size and returns the callable
// to be timed.
static void bench(const string& dtype, const string& op, long nMax,
                  const function<function<void()>(long)>& setup) {
  string path = "bench/cpp_" + dtype + "_" + op + ".csv";
  ofstream out(path);
  out << "size,time_seconds\n";
  out.precision(12);
  Series s;
  s.dtype = dtype;
  s.op = op;
  s.group = groupOf(op);
  for (long n : logspace(nMax, POINTS)) {
    auto fn = setup(n);
    const double secs = timeBest(fn);
    out << n << "," << secs << "\n";
    s.n.push_back(double(n));
    s.t.push_back(secs * 1e3);   // milliseconds read better on a plot
  }
  out.flush();
  results.push_back(std::move(s));
  cout << "  " << path << "\n";
  written++;
}

// ════════════════════════════════════════════════════════════════════════════
//  Operations available for BOTH real and complex
// ════════════════════════════════════════════════════════════════════════════
template <typename T>
static void benchShared() {
  const string dt = dtype_traits<T>::name;
  cout << "\n── " << dt << " ──\n";

  // volatile sinks stop the optimiser deleting work whose result is unused.
  static volatile double sink = 0.0;
  auto keep = [](const auto& M) { sink += std::abs(M[0]); };

#ifdef BENCH_ELEMENTWISE
  bench(dt, "add", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    auto B = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    fillRandom(*B, -22);
    return [A, B, keep] {
      auto C = *A + *B;
      keep(C);
    };
  });
  bench(dt, "subtract", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    auto B = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    fillRandom(*B, -22);
    return [A, B, keep] {
      auto C = *A - *B;
      keep(C);
    };
  });
  bench(dt, "hadamard", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    auto B = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    fillRandom(*B, -22);
    return [A, B, keep] {
      auto C = *A % *B;
      keep(C);
    };
  });
  bench(dt, "elem_div", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    auto B = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    fillRandom(*B, -22);
    for (long i = 0; i < n * n; i++)
      (*B)[int(i)] += T(2);  // keep the divisor away from 0
    return [A, B, keep] {
      auto C = A->div(*B);
      keep(C);
    };
  });
  bench(dt, "scalar_mul", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A, keep] {
      auto C = *A * 1.0000001;
      keep(C);
    };
  });
  bench(dt, "scalar_div", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A, keep] {
      auto C = *A / 1.0000001;
      keep(C);
    };
  });
#endif

#ifdef BENCH_MAPS
#ifdef BENCH_FFT
  // One long 1-D transform. Power-of-two sizes ONLY: a curve that silently mixed
  // radix-2 and Bluestein points would have a step in it that says nothing about
  // either path. The Python side rounds the same way.
  bench(dt, "fft", 1L << 20, [&](long n) {
    long p = 1;
    while (p * 2 <= n) p *= 2;  // round down to a power of two
    auto A = make_shared<Matrix<T>>(1, p);
    fillRandom(*A, -11);
    return [A, keep] {
      auto C = fft(*A);
      keep(C);
    };
  });
#endif

  bench(dt, "elem_exp", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A, keep] {
      auto C = A->exp();
      keep(C);
    };
  });
  bench(dt, "elem_ln", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    for (long i = 0; i < n * n; i++)
      (*A)[int(i)] += T(2);  // keep the argument positive
    return [A, keep] {
      auto C = A->ln();
      keep(C);
    };
  });
  bench(dt, "elem_pow", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    for (long i = 0; i < n * n; i++) (*A)[int(i)] += T(2);
    return [A, keep] {
      auto C = A->pow(2.5);
      keep(C);
    };
  });
#endif

#ifdef BENCH_SHAPE
  bench(dt, "transpose", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A, keep] {
      auto C = A->T();
      keep(C);
    };
  });
  bench(dt, "triu", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A, keep] {
      auto C = A->triu();
      keep(C);
    };
  });
  bench(dt, "reshape", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A, n, keep] {
      auto C = A->reshape(1, n * n);
      keep(C);
    };
  });
  bench(dt, "concat", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    auto B = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    fillRandom(*B, -22);
    return [A, B, keep] {
      auto C = A->concat(*B, 1);
      keep(C);
    };
  });
  if constexpr (is_complex<T>::value) {
    bench(dt, "conj_transpose", N_SQUARE, [&](long n) {
      auto A = make_shared<Matrix<T>>(n, n);
      fillRandom(*A, -11);
      return [A, keep] {
        auto C = A->H();
        keep(C);
      };
    });
    bench(dt, "conjugate", N_SQUARE, [&](long n) {
      auto A = make_shared<Matrix<T>>(n, n);
      fillRandom(*A, -11);
      return [A, keep] {
        auto C = A->conj();
        keep(C);
      };
    });
  }
#endif

#ifdef BENCH_REDUCE
  bench(dt, "sum_all", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A] { sink += std::abs(A->sum()); };
  });
  bench(dt, "sum_dim", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A, keep] {
      auto C = A->sum(true);
      keep(C);
    };
  });
  bench(dt, "norm_fro", N_SQUARE, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A] { sink += A->norm(NormType::Fro); };
  });
  bench(dt, "isdiagonal", N_SQUARE, [&](long n) {
    // A diagonal matrix is the worst case: IsDiagonal() exits early on the
    // first non-zero off-diagonal, so a random matrix would measure almost
    // nothing at all and the curve would be meaningless.
    auto A = make_shared<Matrix<T>>(n, n);
    for (long i = 0; i < n; i++) (*A)(int(i), int(i)) = T(1);
    return [A] { sink += A->IsDiagonal() ? 1.0 : 0.0; };
  });
  bench(dt, "trace", N_LINEAR, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    return [A] { sink += std::abs(A->tr()); };
  });
#endif

#ifdef BENCH_MULTIPLY
  bench(dt, "multiply", N_CUBIC, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    auto B = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    fillRandom(*B, -22);
    return [A, B, keep] {
      auto C = *A * *B;
      keep(C);
    };
  });
#endif

#ifdef BENCH_KRON
  bench(dt, "kron", N_KRON, [&](long n) {
    auto A = make_shared<Matrix<T>>(n, n);
    auto B = make_shared<Matrix<T>>(n, n);
    fillRandom(*A, -11);
    fillRandom(*B, -22);
    return [A, B, keep] {
      auto C = A->kron(*B);
      keep(C);
    };
  });
#endif
}

// ════════════════════════════════════════════════════════════════════════════
//  Real-only operations
//
//  Everything below needs det/inverse/LU/QR/eig/pow/log, which still take the
//  double(grid[k]) path and so do not compile for a complex datatype. See
//  roadmap item 1 (work_t) at the top of Matrix1.0.hpp.
// ════════════════════════════════════════════════════════════════════════════
static void benchRealOnly() {
  const string dt = "real";
  static volatile double sink = 0.0;
  auto keep = [](const auto& M) { sink += std::abs(M[0]); };

#ifdef BENCH_DECOMP
  bench(dt, "det", N_CUBIC, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeWellConditioned(n, -31));
    return [A] { sink += A->det(); };
  });
  bench(dt, "inverse", N_CUBIC, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeWellConditioned(n, -31));
    return [A, keep] {
      auto C = A->inverse();
      keep(C);
    };
  });
  bench(dt, "solve", N_CUBIC, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeWellConditioned(n, -31));
    auto b = make_shared<Matrix<double>>(n, 1);
    b->set_Ran_values(-1.0, 1.0, -41);
    return [A, b, keep] {
      auto C = A->solve(*b);
      keep(C);
    };
  });
  bench(dt, "lu", N_CUBIC, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeWellConditioned(n, -31));
    return [A] {
      auto [L, U, P] = A->LU();
      sink += L(0, 0) + U(0, 0) + P(0, 0);
    };
  });
  bench(dt, "qr", N_CUBIC, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeWellConditioned(n, -31));
    return [A] {
      auto [Q, R, P] = A->QR();
      sink += Q(0, 0) + R(0, 0) + P(0, 0);
    };
  });
  bench(dt, "svd", N_ITER, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeWellConditioned(n, -31));
    return [A] {
      auto [U, S, V] = A->svd();
      sink += U(0, 0) + S(0, 0) + V(0, 0);
    };
  });
  bench(dt, "cholesky", N_CUBIC, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeSPD(n, -51));
    return [A, keep] {
      auto C = A->cholesky();
      keep(C);
    };
  });
  bench(dt, "rank", N_ITER, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeWellConditioned(n, -31));
    return [A] { sink += double(A->rank()); };
  });
#endif

#ifdef BENCH_EIG
  // Symmetric: eig() throws on a complex-conjugate pair, which a random
  // non-symmetric matrix produces almost every time.
  bench(dt, "eig", N_ITER, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeSymmetric(n, -61));
    return [A] {
      auto [e, Q] = A->eig();
      sink += e(0, 0) + Q(0, 0);
    };
  });
#endif

#ifdef BENCH_MATFN
  bench(dt, "mat_pow_int", N_ITER, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeWellConditioned(n, -31));
    return [A, keep] {
      auto C = pow(*A, 3);
      keep(C);
    };
  });
  // SPD: pow(A,0.5) and log(A) are real-valued only for positive eigenvalues.
  bench(dt, "mat_pow_real", N_ITER, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeSPD(n, -51));
    return [A, keep] {
      auto C = pow(*A, 0.5);
      keep(C);
    };
  });
  bench(dt, "mat_log", N_ITER, [&](long n) {
    auto A = make_shared<Matrix<double>>(makeSPD(n, -51));
    return [A, keep] {
      auto C = log(*A, M_E);
      keep(C);
    };
  });
  bench(dt, "mat_exp", N_ITER, [&](long n) {
    auto A = make_shared<Matrix<double>>(n, n);
    A->set_Ran_values(-0.5, 0.5, -71);
    return [A, keep] {
      auto C = exp(*A);
      keep(C);
    };
  });
#endif
}

int main() {
  if (system("mkdir -p bench") != 0) {
    cerr << "could not create bench/\n";
    return 1;
  }
  cout << "MatrixCpp benchmarks -> bench/cpp_<dtype>_<op>.csv\n";
#ifdef _OPENMP
  cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
  cout << "OpenMP: disabled\n";
#endif

  auto t0 = high_resolution_clock::now();
  benchShared<double>();
  benchRealOnly();
  benchShared<std::complex<double>>();
  auto t1 = high_resolution_clock::now();

  cout << "\n"
       << written << " series timed in " << duration<double>(t1 - t0).count() << " s\n";

  // ── Draw the results ──────────────────────────────────────────────────
  // One figure per (dtype, group), every operation in that group overlaid on
  // log-log axes. Fifty-odd separate figures would be a minute of Julia
  // startup and nothing anyone would look at; grouped, the shape of each
  // family is visible at a glance and a curve that bends the wrong way stands
  // out.
  if (system("mkdir -p benchmarks/plots") != 0) {
    cerr << "could not create benchmarks/plots/\n";
    return 1;
  }
  cout << "\nplotting -> benchmarks/plots/\n";
  std::set<string> seen;
  for (const Series& s : results) seen.insert(s.dtype + "|" + s.group);
  for (const string& key : seen) {
    const size_t bar = key.find('|');
    const string dt = key.substr(0, bar), grp = key.substr(bar + 1);
    plt::figure();
    int drawn = 0;
    for (const Series& s : results) {
      if (s.dtype != dt || s.group != grp || s.n.empty()) continue;
      plt::plot(s.n, s.t, s.op);
      drawn++;
    }
    if (drawn == 0) continue;
    plt::set("xscale", ":log10");
    plt::set("yscale", ":log10");
    plt::title(dt + " — " + grp + " (" + std::to_string(drawn) + " operations)");
    plt::xlabel("n");
    plt::ylabel("time (ms)");
    plt::margin(6.0);
    plt::legend(":outerright");
    plt::size(1000, 600);
    const string out = "benchmarks/plots/speed_" + dt + "_" + grp + ".png";
    plt::save(out);
    cout << "  " << out << "  (" << drawn << " ops)\n";
  }

  cout << "\nFor the NumPy comparison:\n"
       << "  python3 benchmarks/numpy_timings.py    # times NumPy into bench/\n"
       << "  ./plot_comparison                      # draws C++ vs NumPy\n";
  return 0;
}
