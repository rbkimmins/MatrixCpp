#pragma once

// ==========================================================================
//  Computational Learning — backend selection
// ==========================================================================
//
// One switch decides whether the whole library runs on the CPU or the GPU:
//
//     g++ -std=c++17 -O3 -march=native -fopenmp yours.cpp -o yours
//         ... builds against mcpu, the default, and needs no CUDA at all
//
//     g++ -std=c++17 -O3 -march=native -fopenmp -DMATRIXCPP_GPU yours.cpp
//         -Lgpu -lmatrixcpp_gpu -L/usr/local/cuda/lib64
//         -lcudart -lcublas -lcusolver -lcurand -lcufft -o yours
//         ... builds against mgpu
//
// Everything below is written against `ml::Matrix<T>`, which is whichever of
// mcpu::Matrix and mgpu::Matrix the switch picked. The two have deliberately
// been kept one-to-one -- same spellings, same shapes, same conventions -- so
// an algorithm written once compiles for either.
//
// WHAT IT DOES NOT DO. Selecting the GPU does not make a small problem fast.
// A kernel launch costs ~5 us and a scalar element read crosses PCIe at ~5 us
// whatever its size, so anything under roughly 10^4 elements loses to the CPU
// no matter how it is written. Batch the work, keep it resident, and check
// gpu/README.md before assuming a layer belongs on the device.

// ── Choosing a backend ──────────────────────────────────────────────────────
//
// MATRIXCPP_GPU or MATRIXCPP_CPU, defined on the command line. Bare GPU and
// CPU are honoured too, since they are the obvious thing to reach for, but
// they are NOT the recommended spelling. Single-word macros collide easily in
// a large build, and a stray -DGPU from an unrelated library would silently
// switch this one over. Prefer the prefixed names.

#if (defined(MATRIXCPP_GPU) || defined(GPU)) && (defined(MATRIXCPP_CPU) || defined(CPU))
    #error "Machine_Learning.hpp: define exactly one of MATRIXCPP_GPU and MATRIXCPP_CPU, not both."
#endif

#if defined(MATRIXCPP_GPU) || defined(GPU)
    #define MATRIXCPP_ML_ON_GPU 1
#else
    #define MATRIXCPP_ML_ON_GPU 0
#endif

// The CPU package comes in either way: the GPU one is built on top of it (a
// device matrix converts to a host matrix through .cpu()), and anything that
// has to reach the host -- reading a loss, writing a checkpoint, plotting --
// needs it regardless of where the arithmetic ran.
#include <string>

#include "../basic/MatrixCpp.hpp"

#if MATRIXCPP_ML_ON_GPU
    #include "../gpu/MatrixGpu.hpp"
#endif

namespace ml {

    // ── The backend, as a namespace ─────────────────────────────────────────────
    //
    // A namespace ALIAS rather than `using namespace`, which a header must never
    // do: it would dump every name into every translation unit that includes this
    // one, and mcpu and mgpu both export `Matrix`, so a file that ended up with
    // both would find the unqualified name ambiguous. `backend::` keeps the choice
    // explicit and reversible.

#if MATRIXCPP_ML_ON_GPU
    namespace backend = mgpu;
#else
    namespace backend = mcpu;
#endif

    // The one name most code needs. Write ml::Matrix<double> and the switch above
    // decides where it lives.
    template <typename T>
    using Matrix = backend::Matrix<T>;

    // Always a HOST matrix, whichever backend is active. Use it for anything that
    // has to be read element by element or handed to another library.
    template <typename T>
    using HostMatrix = mcpu::Matrix<T>;

    // ── The shared vocabulary ───────────────────────────────────────────────────
    //
    // mgpu re-exports these from mcpu rather than defining its own, so both
    // backends name the SAME objects and this block is unambiguous either way.

    using backend::all;
    using backend::all_t;
    using backend::COL;
    using backend::NormType;
    using backend::ROW;

    // ── Which backend is this? ──────────────────────────────────────────────────
    //
    // Compile-time, so a layer that genuinely has to differ can branch with
    // `if constexpr (ml::on_gpu)` and pay nothing for the branch not taken.
    // Reach for it sparingly: an algorithm that needs it twice is usually one
    // that should have been written against the common interface instead.

    inline constexpr bool on_gpu = (MATRIXCPP_ML_ON_GPU != 0);
    inline constexpr const char* backend_name() {
        return on_gpu ? "gpu" : "cpu";
    }

    // ── Crossing between host and backend ───────────────────────────────────────
    //
    // These are identity functions on the CPU build and real transfers on the GPU
    // one, so loading data and reading results can be written once. On the GPU
    // each call moves the whole buffer across PCIe at 9-22 GB/s -- fine at the
    // edges of a training loop, ruinous inside one.

    template <typename T>
    Matrix<T> to_backend(const mcpu::Matrix<T>& host) {
#if MATRIXCPP_ML_ON_GPU
        return mgpu::Matrix<T>(host);
#else
        return host;
#endif
    }

    template <typename T>
    HostMatrix<T> to_host(const Matrix<T>& x) {
#if MATRIXCPP_ML_ON_GPU
        return x.cpu();
#else
        return x;
#endif
    }

    // Blocks until queued work has finished. A no-op on the CPU, a device
    // synchronise on the GPU -- needed before timing anything, since a launch
    // returns as soon as the work is queued rather than done.
    inline void sync() {
#if MATRIXCPP_ML_ON_GPU
        mgpu::sync();
#endif
    }

    // True when the selected backend can actually run. Always true on the CPU;
    // on the GPU it is false when there is no driver or no card, which is a state
    // worth branching on rather than crashing over.
    inline bool available() {
#if MATRIXCPP_ML_ON_GPU
        return mgpu::available();
#else
        return true;
#endif
    }

    // Names the backend and, on the GPU, the device and library versions -- which
    // is two lines there and one here. For the top of a training run, so a log
    // says what it actually ran on rather than what it was meant to.
    inline std::string describe() {
#if MATRIXCPP_ML_ON_GPU
        return std::string("gpu — ") + (mgpu::available() ? mgpu::describe() : "no CUDA device");
#else
        return "cpu — mcpu::Matrix";
#endif
    }

    // ── Activations ─────────────────────────────────────────────────────────
    //
    // The step that turns a weighted sum into a bool. Anything callable as
    // `bool(T)` will do, so a caller is never limited to what is here:
    //
    //     struct my_rule { bool operator()(double s) const { return s >= 0.5; } };
    //     bool_perceptron_train(w, X, y, 0.1, 100, -1, my_rule{});
    //
    // A lambda works too, which is usually the shortest route:
    //
    //     bool_perceptron_train(w, X, y, 0.1, 100, -1,
    //                           [](double s) { return s > 0.0; });
    namespace activation {

        // The classic perceptron rule -- Heaviside / sign, thresholded at zero.
        // This is what a "boolean perceptron" means unless something else is
        // said, so it is the default everywhere below.
        struct sign {
            template <typename T>
            bool operator()(T s) const {
                return s > T(0);
            }
        };

        // The same with the threshold somewhere other than zero. Only useful
        // WITHOUT a bias input -- with one, the perceptron learns its own
        // threshold and this is redundant.
        struct threshold {
            double level;
            explicit threshold(double lvl = 0.0) : level(lvl) {}
            template <typename T>
            bool operator()(T s) const {
                return double(s) > level;
            }
        };

    }  // namespace activation

    // boolian perceptron
    // note All functions assume that the vectors
    // are column vectors unless otherwise stated.
    template <typename T, typename Activation = activation::sign>
    inline bool bool_perceptron_sign(const Matrix<T>& weight,
                                     const Matrix<T>& x,
                                     Activation act = Activation{}) {
        return act(sum(weight % x));
    }

    // Trains `weight` in place by the perceptron rule.
    //
    //   x:             training data, one SAMPLE PER COLUMN, so x is
    //                  (features x samples)
    //   weight:        column vector, one entry per feature (per ROW of x)
    //   y:             the answers, one per sample -- a column vector with
    //                  x.cols() entries. Always a HOST matrix: labels are read
    //                  one at a time, which is host work, and mgpu::Matrix has
    //                  no bool instantiation anyway.
    //   learning_rate: how far each correction moves the weights
    //   max_epochs:    the cap. The loop below cannot be trusted to end on its
    //                  own -- see the note on it.
    //   seed:          negative, as ran2 requires; each epoch derives its own
    //                  shuffle from it, so a run is reproducible.
    //
    // Returns the number of epochs used, or -1 if it hit the cap without
    // separating the data.
    //
    // WHY THE CAP IS NOT OPTIONAL. A perceptron converges only if the classes
    // are linearly separable. XOR is not, and neither is most real data, so
    // `while (!trained)` is an infinite loop on any input that cannot be
    // solved -- which is the common case, not the rare one.
    template <typename T, typename Activation = activation::sign>
    long bool_perceptron_train(Matrix<T>& weight,
                               const Matrix<T>& x,
                               const HostMatrix<bool>& y,
                               double learning_rate,
                               long max_epochs = 100,
                               long seed = -1,
                               Activation act = Activation{}) {
        const long samples = x.cols();  // COLUMNS: one sample each
        if (weight.numel() != x.rows())
            throw std::invalid_argument("bool_perceptron_train: weight has " +
                                        std::to_string(weight.numel()) + " entries but x has " +
                                        std::to_string(x.rows()) + " features (rows)");
        if (y.numel() != samples)
            throw std::invalid_argument("bool_perceptron_train: y has " +
                                        std::to_string(y.numel()) + " labels but x has " +
                                        std::to_string(samples) + " samples (columns)");
        if (seed >= 0)
            throw std::invalid_argument("bool_perceptron_train: seed must be negative");

        for (long epoch = 0; epoch < max_epochs; epoch++) {
            bool trained = true;

            // A PERMUTATION, not a random draw. Sampling with replacement
            // visits some rows twice and misses others entirely, and an
            // unseeded fill repeats the same sequence every epoch because its
            // seed comes from the clock at whole-second resolution. randperm
            // gives each sample exactly once, and the per-epoch offset makes
            // the order genuinely differ between passes.
            // mcpu:: explicitly. randperm takes only longs, so unlike
            // sum(weight % x) above there is no Matrix argument for ADL to
            // follow -- and the shuffle is host-side bookkeeping regardless of
            // where the arithmetic runs.
            const HostMatrix<long> order = mcpu::randperm(samples, seed - epoch);

            for (long k = 0; k < samples; k++) {
                const long i = order[k];
                const Matrix<T> xi = x(all, i);  // const slice materialises
                const bool pred = bool_perceptron_sign(weight, xi, act);
                if (y[i] == pred)
                    continue;

                // The perceptron update moves along the sample in the
                // direction of the TARGET, so the sign has to come from the
                // label mapped to +-1. Multiplying by the bool itself makes
                // every negative example a no-op and only ever pushes the
                // weights one way.
                const T step = T(learning_rate) * (y[i] ? T(1) : T(-1));
                weight += xi * step;
                trained = false;
            }

            if (trained)
                return epoch + 1;
        }
        return -1;  // hit the cap; the data may not be linearly separable
    }

    // ── Boolean functions to train against ──────────────────────────────────
    //
    // The textbook targets. AND and OR are linearly separable and a perceptron
    // finds them in a handful of epochs; XOR is NOT, which is the whole point
    // of it being here -- it is the standard demonstration that a single layer
    // has a ceiling, and what max_epochs exists to survive.
    //
    // Any `bool(bool, bool)` works, so a caller can pass their own.
    namespace logic {

        inline bool AND(bool a, bool b) {
            return a && b;
        }
        inline bool OR(bool a, bool b) {
            return a || b;
        }
        inline bool NAND(bool a, bool b) {
            return !(a && b);
        }
        inline bool NOR(bool a, bool b) {
            return !(a || b);
        }
        inline bool XOR(bool a, bool b) {
            return a != b;
        }

        // The four-row truth table of `f`, laid out the way the trainer wants
        // it: x is (features x 4) with one sample per column, y is 4 x 1.
        //
        // The third feature row is a constant 1, the BIAS. Without it the
        // decision boundary is forced through the origin and even AND becomes
        // unlearnable, which looks like a broken trainer rather than a missing
        // input. Pass bias = false only to see that happen on purpose.
        template <typename T, typename F>
        std::pair<HostMatrix<T>, HostMatrix<bool>> truth_table(F f, bool bias = true) {
            const long features = bias ? 3 : 2;
            HostMatrix<T> x(features, 4);
            HostMatrix<bool> y(4, 1);
            for (long c = 0; c < 4; c++) {
                const bool a = (c & 2) != 0, b = (c & 1) != 0;
                x(0, c) = T(a);
                x(1, c) = T(b);
                if (bias)
                    x(2, c) = T(1);
                y[c] = f(a, b);
            }
            return {x, y};
        }

    }  // namespace logic

    // ── ROADMAP -- what is not here yet ─────────────────────────────────────
    //
    // What is above is ONE layer, ONE output, and a hard yes/no. The steps
    // below are in dependency order: each needs the one before it, and each
    // names the reading that covers it.
    //
    // Sources, since page numbers are edition-specific:
    //   [E]  Ekman, "Learning Deep Learning", Addison-Wesley 2021
    //   [G]  Goodfellow/Bengio/Courville, "Deep Learning", MIT Press 2016
    //   [M]  Mohri/Rostamizadeh/Talwalkar, "Foundations of Machine
    //        Learning", 2nd ed, MIT Press 2018
    // Pages are PRINTED book pages, not PDF viewer pages.
    //
    //
    // 0. RENAME bool_perceptron_sign -> bool_perceptron_predict.
    //
    //    It applies whatever activation it is handed, so `_sign` names the
    //    default rather than the function -- and the name reads as though it
    //    were interchangeable with activation::sign, which is the thing
    //    actually passed as `act`. It is not: the functor is `bool(T)` over
    //    one scalar, this takes two matrices. Cheap fix, do it first.
    //
    //
    // 1. MULTI-CLASS, still a hard decision. NO ACTIVATION CHANGE NEEDED.
    //
    //    weight becomes (classes x features), the score is the whole vector
    //    W * xi, and the prediction is argmax over it. The ARGMAX REPLACES
    //    THE STEP, which is why nothing differentiable is required yet. y
    //    becomes HostMatrix<long> of class indices. The update stays
    //    mistake-driven -- predicted p, true class t, p != t:
    //
    //        W.row(t) += learning_rate * xi;
    //        W.row(p) -= learning_rate * xi;
    //
    //    Novikoff's guarantee survives this, so `trained` and the -1-on-cap
    //    contract keep meaning exactly what they mean above. This is the
    //    cheapest real capability jump and it reuses the whole epoch loop.
    //
    //    READ  [M] 9.1 p213 for the score-vector formulation; 9.4.1-9.4.2
    //          p229 for one-vs-all against one-vs-one, and why a single
    //          argmax beats K independent binary perceptrons (the ambiguous
    //          -region problem). [E] ch4 p101 and p103 for the code view.
    //
    //
    // 2. GRADED OUTPUT -- and the TRAINING RULE HAS TO CHANGE WITH IT.
    //
    //    This is the step that cannot be done by swapping the activation
    //    alone. The perceptron rule works BECAUSE the output is a step: it
    //    is mistake-driven, so no error means no update. With a sigmoid the
    //    output is never exactly the label, `y[i] == pred` never holds, and
    //    the `if (trained) return epoch + 1` early exit becomes dead code.
    //    Replace it with a loss tolerance or a no-improvement test.
    //
    //    The update becomes gradient descent on a loss. Logistic output with
    //    cross-entropy is the pairing worth writing, because the activation
    //    derivative cancels:
    //
    //        yhat = sigma(dot(w, xi));
    //        w += learning_rate * (y[i] - yhat) * xi;   // sigma' cancels
    //
    //    Pair the same sigmoid with SQUARED error instead and the update
    //    carries a sigma' = sigma(1 - sigma) factor that goes to zero
    //    exactly when the unit is confidently wrong -- the saturation stall.
    //    Cross-entropy exists to kill that term. Do not discover this the
    //    hard way.
    //
    //    API consequence: the Activation concept above is `bool(T)`. Graded
    //    output makes it `T(T)`, and gradient descent also needs the
    //    derivative, so those structs grow a second member:
    //
    //        struct logistic {
    //            T operator()(T s) const;    // sigma(s)
    //            T derivative(T s) const;    // sigma(s) * (1 - sigma(s))
    //        };
    //
    //    That is a genuinely different contract from activation::sign, and
    //    it constrains T to floating point, which sign does not. Give it its
    //    own namespace rather than mixing it in with the boolean ones.
    //
    //    READ  [E] ch2 p49, "Analytic Explanation of the Perceptron Learning
    //          Algorithm" -- derives the rule above AS gradient descent,
    //          which is what makes this step follow rather than be asserted.
    //          [G] 6.2.1 p178 on cost functions: you do not pick a loss, you
    //          derive it from the distribution you claim to model. Then
    //          [G] 6.2.2.2 p182 for the saturation argument in full.
    //          [E] ch5 p124 and p130 for the same ground concretely, and
    //          p135 for the numerical traps -- naive log(sigmoid(x)) bites.
    //
    //
    // 3. THE OTHER TWO OUTPUT UNITS, once 2 is in place.
    //
    //    Softmax over K scores with cross-entropy generalises 2 to the
    //    multi-class case of 1; a bare linear unit with squared error gives
    //    regression (ADALINE / the delta rule). Same loop, different output
    //    unit and loss.
    //
    //    READ  [G] 6.2.2.3 p184 (softmax) and 6.2.2.1 p181 (linear).
    //          [E] ch6 "Output Units" p154 lays all three out as a menu
    //          keyed to problem type -- the most useful single section for
    //          deciding what this API should actually offer.
    //          [M] 13.7 p325 for logistic regression stated properly.
    //
    //
    // 4. A HIDDEN LAYER, which is the only thing that lifts the ceiling.
    //
    //    Worth being blunt about, because it is the common misreading of
    //    steps 2-3: swapping the step for a sigmoid does NOT make XOR
    //    learnable. A single unit with any monotone activation still cuts
    //    the input space with one hyperplane; it just reports distance from
    //    it smoothly instead of which side. The note on logic::XOR above
    //    stays true verbatim.
    //
    //    The real reason a differentiable activation is needed is that it is
    //    the PREREQUISITE FOR STACKING: backprop has to push error through
    //    the activation, so it needs the derivative. A step function has a
    //    zero derivative everywhere it is defined, which is why the
    //    perceptron never became a multi-layer method on its own.
    //
    //    READ  [G] 6.1 p171, the XOR example worked end to end, showing
    //          geometrically what the hidden layer buys. Then [G] 6.5 p204
    //          for backprop proper. [E] ch3 p60 and p82 for the code.
    //
    //
    // SUGGESTED ORDER, if reading rather than working step by step:
    //
    //    [E] ch1-2      -- fast, mostly what is already implemented above,
    //                      but ch1 p20 "Implementing Perceptrons with Linear
    //                      Algebra" walks dot product -> matrix-vector ->
    //                      matrix-matrix as the deliberate path from one
    //                      perceptron to a layer, which maps straight onto
    //                      the primitives this file is built on
    //    [M] 8.3.1 p190 -- the perceptron as stochastic gradient descent on
    //                      a convex but NON-DIFFERENTIABLE objective, plus
    //                      Theorem 8.8, Novikoff's mistake bound: updates
    //                      bounded by (r/rho)^2, independent of dimension.
    //                      This is the citation for the max_epochs note
    //                      above -- it states plainly that the algorithm
    //                      simply does not terminate on non-separable data
    //    [E] ch3        -- sigmoid neurons and backpropagation
    //    [G] 6.2.1-6.2.2 -- output units and the losses that go with them
    //    [M] 9.4 p229   -- read before writing the multi-class version
    //
    //    Optional, for the general theory behind step 2: [M] 4.7 p73,
    //    "Convex surrogate losses" -- why the 0-1 loss gets replaced by
    //    something differentiable at all.

}  // namespace ml
