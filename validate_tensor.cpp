// Assertion-based validation for Tensor.hpp.
//
// Same shape as validate.cpp: every case either matches a hand-computed value or
// an identity that must hold, and the process exits non-zero if any fail.
//
// Build: g++ -std=c++17 -O2 -fopenmp -o validate_tensor validate_tensor.cpp
#include "Tensor.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <complex>
#include <functional>

static int checks = 0, failures = 0;
static std::string suite;

static void section(const std::string& s) {
    suite = s;
    std::cout << "\n\033[1m── " << s << " ──\033[0m\n";
}
static void ok(bool cond, const std::string& what) {
    checks++;
    if (cond) { std::cout << "  \033[32mPASS\033[0m  " << what << "\n"; return; }
    failures++;
    std::cout << "  \033[31mFAIL\033[0m  " << what << "   [" << suite << "]\n";
}
static bool near(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol + tol * std::abs(b);
}
static bool threw(const std::function<void()>& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
// 0,1,2,... in row-major order.
static Tensor<double> iota(std::vector<long> shape) {
    Tensor<double> t(shape);
    auto flat = t.reshape(-1);
    for (long i = 0; i < t.size(); i++) flat(i) = double(i);
    return t;
}

int main() {
    std::cout << std::setprecision(12);
    // Quieten the deliberate error paths — every throw below is expected.
    std::cerr.setstate(std::ios_base::failbit);

    // ═══════════════════════════════════════════════════════════════════
    section("Shape and construction");
    {
        Tensor<double> A(2, 3, 4);
        ok(A.rank() == 3,                    "variadic constructor sets the rank");
        ok(A.size() == 24,                   "size is the product of the shape");
        ok(A.shape(0) == 2 && A.shape(1) == 3 && A.shape(2) == 4, "shape() per axis");
        ok(A.isContiguous(),                 "a fresh tensor is contiguous");
        ok(A.sum() == 0.0,                   "a fresh tensor is zero-filled");
        ok(A.stride(0) == 12 && A.stride(1) == 4 && A.stride(2) == 1,
                                             "row-major strides");
        Tensor<double> B(std::vector<long>{2, 3, 4});
        ok(B.rank() == 3 && B.size() == 24,  "vector<long> constructor agrees");
        Tensor<double> E;
        ok(E.empty() && E.rank() == 0,       "default constructor is the empty tensor");
        ok(threw([]{ Tensor<double> X(2, -3); }), "a negative dimension throws");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Indexing");
    {
        Tensor<double> A = iota({2, 3, 4});
        ok(A(0, 0, 0) == 0.0,                "first element");
        ok(A(1, 2, 3) == 23.0,               "last element, row-major order");
        ok(A(0, 1, 2) == 6.0,                "interior element matches i*12+j*4+k");
        A(1, 1, 1) = -5.0;
        ok(A(1, 1, 1) == -5.0,               "operator() assigns");
        ok(A.at({1, 1, 1}) == -5.0,          "at(vector) agrees with operator()");
        ok(threw([&]{ A(0, 0); }),           "too few indices throws");
        ok(threw([&]{ A(0, 0, 0, 0); }),     "too many indices throws");
        ok(threw([&]{ A(0, 3, 0); }),        "an out-of-range index throws");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Views: reshape, permute, slice");
    {
        Tensor<double> A = iota({2, 3, 4});

        Tensor<double> R = A.reshape(6, 4);
        ok(R.rank() == 2 && R.shape(0) == 6, "reshape changes the shape");
        ok(R(5, 3) == 23.0,                  "reshape preserves memory order");
        ok(R.reshape(-1).size() == 24,       "reshape(-1) infers the free dimension");
        ok(R.reshape(2, -1, 4).shape(1) == 3, "-1 works in an interior position");
        ok(threw([&]{ A.reshape(5, 5); }),   "a reshape that changes the element count throws");
        ok(threw([&]{ A.reshape(-1, -1); }), "two inferred dimensions throw");

        Tensor<double> P = A.permute({2, 0, 1});
        ok(P.shape(0) == 4 && P.shape(1) == 2 && P.shape(2) == 3, "permute reorders the shape");
        ok(!P.isContiguous(),                "a permuted view is not contiguous");
        ok(P(3, 1, 2) == A(1, 2, 3),         "permute preserves the elements");
        ok(threw([&]{ A.permute({0, 0, 1}); }), "a repeated axis throws");
        ok(threw([&]{ A.permute({0, 1}); }),    "the wrong number of axes throws");

        ok(A.swapAxes(0, 2).shape(0) == 4,   "swapAxes swaps two axes");
        ok(A.T().shape(0) == 4 && A.T().shape(2) == 2, "T() reverses every axis");

        Tensor<double> S = A.slice(1, 2);
        ok(S.rank() == 2 && S.shape(0) == 2 && S.shape(1) == 4, "slice drops its axis");
        ok(S(1, 3) == A(1, 2, 3),            "slice picks the right elements");
        ok(threw([&]{ A.slice(1, 9); }),     "an out-of-range slice index throws");

        // Rank-2 T() must agree with Matrix::T(), which is the consistency claim.
        Tensor<double> M2 = iota({3, 5});
        Matrix<double> Mm = M2.toMatrix();
        ok(M2.T().toMatrix() == Mm.T(),      "rank-2 Tensor::T() agrees with Matrix::T()");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Copy semantics: views alias, copies do not");
    {
        Tensor<double> A = iota({2, 3});

        Tensor<double> V = A.permute({1, 0});     // a view
        V(0, 1) = 99.0;
        ok(A(1, 0) == 99.0,                  "writing through a view writes through to the source");

        Tensor<double> C = A;                     // an ordinary copy
        C(0, 0) = -1.0;
        ok(A(0, 0) == 0.0,                   "copy-construction is a deep copy");
        ok(C.isContiguous(),                 "a copy is contiguous");

        Tensor<double> D = V.clone();             // deep copy of a strided view
        ok(D.isContiguous(),                 "clone() of a view materialises it");
        ok(D(0, 1) == V(0, 1),               "clone() preserves the logical elements");
        D(0, 1) = 7.0;
        ok(V(0, 1) == 99.0,                  "clone() is independent of the view");

        Tensor<double> Cd;
        Cd = V;                                   // copy assignment from a view
        ok(Cd.isContiguous() && Cd(0, 1) == 99.0, "copy assignment materialises a view");

        // A view keeps its storage alive after the original is destroyed.
        Tensor<double> keep;
        {
            Tensor<double> tmp = iota({2, 2});
            keep = std::move(tmp.permute({1, 0}));
        }
        ok(keep(0, 1) == 2.0,                "a view outlives the tensor it came from");

        // contiguous() on an already-contiguous tensor shares rather than copies.
        Tensor<double> F = iota({2, 2});
        Tensor<double> G = F.contiguous();
        G(0, 0) = 42.0;
        ok(F(0, 0) == 42.0,                  "contiguous() is a no-op share when already contiguous");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Element-wise arithmetic");
    {
        Tensor<double> A = iota({2, 3});
        Tensor<double> B = iota({2, 3});
        B = B + B;                                   // 0,2,4,...

        ok((A + B)(1, 2) == 5.0 + 10.0,      "operator+");
        ok((B - A)(1, 2) == 10.0 - 5.0,      "operator-");
        ok((A % B)(1, 2) == 5.0 * 10.0,      "operator% is the element-wise product");
        ok(A.mul(B) == A % B,               "A.mul(B) is a named spelling of %");
        ok((A *dot* B) == A % B,             "A *dot* B is A .* B");
        ok((-A)(1, 2) == -5.0,               "unary minus");
        ok((A * 3.0)(1, 2) == 15.0,          "scalar multiply");
        ok((3.0 * A)(1, 2) == 15.0,          "scalar multiply from the left");
        ok((B / 2.0)(1, 2) == 5.0,           "scalar divide");

        Tensor<double> Pos = iota({2, 3}) + Tensor<double>(2, 3).fill(1.0);   // 1..6
        ok(near(Pos.div(Pos)(1, 2), 1.0),   "A.div(B) is element-wise division");
        ok(Pos.div(Pos) == (Pos /dot/ Pos), "A /dot/ B is A ./ B");

        Tensor<double> Acc = iota({2, 3});
        Acc += B;
        ok(Acc(1, 2) == 5.0 + 10.0,          "operator+= accumulates");

        ok(threw([&]{ Tensor<double> W(3, 2); Tensor<double> Z = A + W; (void)Z; }),
                                             "mismatched shapes throw");

        // Element-wise ops on a strided view must give the logical answer.
        Tensor<double> Pv = A.permute({1, 0});
        ok((Pv + Pv)(2, 1) == 2.0 * A(1, 2), "element-wise works through a permuted view");
        ok((Pv + Pv).isContiguous(),         "the result of an op on a view is contiguous");

        // Chained expressions must not corrupt anything: the rvalue overloads
        // write into the temporary rather than allocating each time.
        Tensor<double> Chain = A + B + A + B;
        ok(Chain(1, 2) == 2.0 * (5.0 + 10.0), "a four-term chain is correct");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Reductions");
    {
        Tensor<double> A = iota({2, 3, 4});   // 0..23
        ok(A.sum() == 276.0,                 "sum() over every element");
        ok(near(A.mean(), 276.0 / 24.0),     "mean()");
        ok(A.min() == 0.0 && A.max() == 23.0, "min() and max()");

        Tensor<double> S2 = A.sum(2);
        ok(S2.rank() == 2 && S2.shape(0) == 2 && S2.shape(1) == 3, "sum(axis) drops that axis");
        ok(S2(0, 0) == 0 + 1 + 2 + 3,        "sum over the last axis");
        Tensor<double> S0 = A.sum(0);
        ok(S0.shape(0) == 3 && S0.shape(1) == 4, "sum(0) shape");
        ok(S0(0, 0) == A(0, 0, 0) + A(1, 0, 0), "sum over the first axis");
        // Every axis-sum must total the same as the full sum.
        ok(near(S0.sum(), 276.0) && near(S2.sum(), 276.0) && near(A.sum(1).sum(), 276.0),
                                             "all three axis-sums total 276");
        ok(threw([&]{ A.sum(3); }),          "an out-of-range axis throws");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Contraction");
    {
        // Rank-2 contraction MUST equal the matrix product — the consistency claim.
        Matrix<double> Am(4, 5), Bm(5, 3);
        Am.set_Ran_values(-1, 1, -11);
        Bm.set_Ran_values(-1, 1, -22);
        Tensor<double> At = Tensor<double>::fromMatrix(Am);
        Tensor<double> Bt = Tensor<double>::fromMatrix(Bm);

        Tensor<double> Ct = At * Bt;
        ok(Ct.rank() == 2 && Ct.shape(0) == 4 && Ct.shape(1) == 3, "rank-2 product shape");
        ok(Ct.toMatrix().allclose(Am * Bm, 1e-12, 1e-12),
                                             "Tensor A*B equals Matrix A*B exactly");
        ok(contract(At, Bt, 1) == Ct,        "contract(A,B,1) is operator*");

        // Higher-rank tensordot against a hand-rolled reference.
        Tensor<double> X = iota({2, 3, 4});
        Tensor<double> Y = iota({4, 5});
        Tensor<double> Z = contract(X, Y, 1);
        ok(Z.rank() == 3 && Z.shape(0) == 2 && Z.shape(1) == 3 && Z.shape(2) == 5,
                                             "tensordot shape is A.shape[:-k] + B.shape[k:]");
        bool tdOK = true;
        for (long i = 0; i < 2 && tdOK; i++)
            for (long j = 0; j < 3 && tdOK; j++)
                for (long l = 0; l < 5 && tdOK; l++) {
                    double acc = 0.0;
                    for (long m = 0; m < 4; m++) acc += X(i, j, m) * Y(m, l);
                    if (!near(Z(i, j, l), acc, 1e-10)) tdOK = false;
                }
        ok(tdOK,                             "tensordot matches an explicit triple loop");

        // Contracting two axes at once.
        Tensor<double> U = iota({2, 3, 4});
        Tensor<double> V = iota({3, 4, 5});
        Tensor<double> W = contract(U, V, 2);
        ok(W.rank() == 2 && W.shape(0) == 2 && W.shape(1) == 5, "two-axis contraction shape");
        bool twoOK = true;
        for (long i = 0; i < 2 && twoOK; i++)
            for (long l = 0; l < 5 && twoOK; l++) {
                double acc = 0.0;
                for (long j = 0; j < 3; j++) for (long m = 0; m < 4; m++)
                    acc += U(i, j, m) * V(j, m, l);
                if (!near(W(i, l), acc, 1e-9)) twoOK = false;
            }
        ok(twoOK,                            "two-axis contraction matches an explicit loop");

        ok(threw([&]{ contract(X, X, 1); }), "mismatched contracted axes throw");

        // Named-axis contraction, checked against a permutation of tensordot.
        Tensor<double> Ax = iota({2, 3, 4});
        Tensor<double> Bx = iota({5, 3});
        Tensor<double> N1 = contract(Ax, Bx, {1}, {1});     // over the length-3 axes
        ok(N1.rank() == 3 && N1.shape(0) == 2 && N1.shape(1) == 4 && N1.shape(2) == 5,
                                             "named-axis contraction shape");
        bool nOK = true;
        for (long i = 0; i < 2 && nOK; i++)
            for (long m = 0; m < 4 && nOK; m++)
                for (long p = 0; p < 5 && nOK; p++) {
                    double acc = 0.0;
                    for (long j = 0; j < 3; j++) acc += Ax(i, j, m) * Bx(p, j);
                    if (!near(N1(i, m, p), acc, 1e-9)) nOK = false;
                }
        ok(nOK,                              "named-axis contraction matches an explicit loop");

        // Out-parameter forms — the std::linalg lesson.
        Tensor<double> Out(std::vector<long>{4, 3});
        contractInto(At, Bt, 1, Out);
        ok(Out == Ct,                        "contractInto writes the same answer");
        contractAccInto(At, Bt, 1, Out);
        ok(Out.allclose(Ct + Ct, 1e-12, 1e-12), "contractAccInto accumulates");
        ok(threw([&]{ Tensor<double> Bad(3, 3); contractInto(At, Bt, 1, Bad); }),
                                             "a wrongly shaped output throws");

        // Contraction through views must not need an explicit .contiguous().
        Tensor<double> Pv = At.T();                      // (5 x 4) strided view
        Tensor<double> Pr = contract(Pv, Tensor<double>::fromMatrix(Matrix<double>(4, 2)), 1);
        ok(Pr.shape(0) == 5 && Pr.shape(1) == 2, "contraction accepts a strided operand");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Logical masks");
    {
        Tensor<double> A = iota({2, 3, 4});    // 0 .. 23
        Tensor<bool> big = A > 10.0;
        ok(big.rank() == 3 && big.shape(0) == 2, "a mask has the operand's shape");
        ok(big.nnz() == 13,                  "A > 10 selects 11..23");
        ok((A <= 10.0).nnz() == 11,          "A <= 10 is the complement");
        ok((!big).nnz() == 11,               "operator! negates a mask");
        ok(A.gt(10.0) == big,                "gt() is the named form of >");
        ok(big.any() && !big.all(),          "any() and all()");
        ok((A > -1.0).all(),                 "all() when everything matches");
        ok(!(A > 100.0).any(),               "any() when nothing matches");

        // Combinators, C spellings and named, exactly as Matrix has them.
        Tensor<bool> mid = (A > 5.0) && (A < 15.0);
        ok(mid.nnz() == 9,                   "operator&& intersects");
        ok(mid == (A > 5.0).land(A < 15.0),  "&& is land()");
        ok(((A < 2.0) || (A > 21.0)).nnz() == 4, "operator|| unions");
        ok(((A > 5.0) ^ (A > 15.0)).nnz() == 10, "operator^ is the symmetric difference");
        ok(((A > 5.0) ^ (A > 15.0)) == (A > 5.0).lxor(A > 15.0), "^ is lxor()");
        ok((big || !big).all(),              "m || !m covers everything");
        ok(!(big && !big).any(),             "m && !m selects nothing");

        // == stays whole-tensor; .eq() is the element-wise one.
        Tensor<double> Acopy = A;
        ok(A == Acopy,                       "operator== is whole-tensor equality");
        ok(A.eq(Acopy).all(),                "eq() on an identical tensor is all true");
        ok(A.ne(Acopy).nnz() == 0,           "ne() on an identical tensor is empty");

        // find() is flat here, not per-axis pairs — unravel() converts.
        auto f = (A > 21.0).find();
        ok(f.size() == 2 && f[0] == 22 && f[1] == 23, "find() gives flat row-major indices");
        std::vector<long> u = A.unravel(23);
        ok(u.size() == 3 && u[0] == 1 && u[1] == 2 && u[2] == 3, "unravel() splits a flat index");
        ok(A.at(A.unravel(23)) == 23.0,      "unravel round-trips through at()");

        // Logical indexing.
        Tensor<double> picked = A(A > 20.0);
        ok(picked.rank() == 1 && picked.size() == 3, "A(mask) is a rank-1 tensor");
        ok(picked(0) == 21.0 && picked(2) == 23.0,   "A(mask) selects in row-major order");

        Tensor<double> W = A;
        W(W < 12.0) = -1.0;
        ok(W(0, 0, 0) == -1.0 && W(1, 2, 3) == 23.0, "A(mask) = scalar writes through");
        ok((W < 0.0).nnz() == 12,            "exactly the selected elements changed");

        Tensor<double> W2 = A;
        Tensor<double> repl(3);
        repl(0) = 100.0; repl(1) = 200.0; repl(2) = 300.0;
        W2(W2 > 20.0) = repl;
        ok(W2(1, 2, 1) == 100.0 && W2(1, 2, 3) == 300.0, "A(mask) = tensor writes one each");

        // Masks work through a strided view.
        Tensor<double> P = A.permute({2, 1, 0});
        ok((P > 10.0).nnz() == 13,           "comparison works through a permuted view");
        ok(!threw([&]{ Tensor<double> q = P(P > 10.0); (void)q; }),
                                             "logical indexing works through a view");
        ok(threw([&]{ Tensor<bool> bad(2, 2); Tensor<double> q = A(bad); (void)q; }),
                                             "a mask of the wrong shape throws");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Matrix interop");
    {
        Matrix<double> M(3, 4);
        M.set_Ran_values(-1, 1, -7);
        Tensor<double> T = Tensor<double>::fromMatrix(M);
        ok(T.rank() == 2 && T.shape(0) == 3 && T.shape(1) == 4, "fromMatrix shape");
        ok(T.toMatrix() == M,                "round trip Matrix -> Tensor -> Matrix");
        ok(T(2, 3) == M(2, 3),               "elements agree");

        Tensor<double> V = iota({5});
        ok(V.toMatrix().rows() == 5 && V.toMatrix().cols() == 1,
                                             "a rank-1 tensor becomes a column vector");
        ok(threw([&]{ iota({2, 2, 2}).toMatrix(); }),
                                             "rank 3 cannot become a Matrix");
        // toMatrix on a strided view must give the logical elements.
        ok(T.T().toMatrix() == M.T(),        "toMatrix materialises a view correctly");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Complex tensors");
    {
        using namespace matrix_literals;
        Tensor<std::complex<double>> Z(2, 2);
        Z(0, 0) = 1.0 + 2.0i;  Z(0, 1) = 3.0;
        Z(1, 0) = 0.0;         Z(1, 1) = 1.0i;
        ok(Z(0, 0) == std::complex<double>(1, 2), "complex literals build a tensor");
        ok((Z + Z)(0, 0) == std::complex<double>(2, 4), "complex addition");
        ok(Z.sum() == std::complex<double>(4, 3),  "complex sum");

        // Contraction against Matrix's complex product.
        Matrix<std::complex<double>> Cm(2, 2);
        Cm(0, 0) = 1.0 + 2.0i;  Cm(0, 1) = 3.0;
        Cm(1, 0) = 0.0;         Cm(1, 1) = 1.0i;
        ok((Z * Z).toMatrix().allclose(Cm * Cm, 1e-12, 1e-12),
                                             "complex contraction equals the matrix product");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Quantum gate application — the workload this exists for");
    {
        // A 5-qubit state is a (2,2,2,2,2) tensor. Applying a single-qubit gate
        // to qubit q is: permute q to the front, reshape to (2, rest), multiply
        // by the 2x2 gate, reshape back, permute back.
        const long nq = 5;
        std::vector<long> qshape((std::size_t)nq, 2);
        Tensor<std::complex<double>> psi(qshape);
        psi.at(std::vector<long>((std::size_t)nq, 0)) = 1.0;   // |00000>

        // X (NOT) gate.
        Tensor<std::complex<double>> X(2, 2);
        X(0, 1) = 1.0; X(1, 0) = 1.0;

        auto applyGate = [&](Tensor<std::complex<double>> s,
                             const Tensor<std::complex<double>>& g, long q) {
            std::vector<long> fwd; fwd.push_back(q);
            for (long k = 0; k < nq; k++) if (k != q) fwd.push_back(k);
            auto flat = s.permute(fwd).reshape(2, -1);      // (2, 2^(n-1))
            auto out  = contract(g, flat, 1);               // gate applied
            auto back = out.reshape(qshape);
            std::vector<long> inv((std::size_t)nq);
            for (long k = 0; k < nq; k++) inv[(std::size_t)fwd[(std::size_t)k]] = k;
            return back.permute(inv).clone();
        };

        psi = applyGate(std::move(psi), X, 2);   // flip qubit 2
        std::vector<long> want{0, 0, 1, 0, 0};
        ok(std::abs(psi.at(want) - std::complex<double>(1, 0)) < 1e-12,
                                             "X on qubit 2 moves |00000> to |00100>");
        double norm2 = 0.0;
        for (long i = 0; i < psi.size(); i++) norm2 += std::norm(psi.reshape(-1)(i));
        ok(near(norm2, 1.0, 1e-12),          "the state stays normalised");

        // Hadamard on qubit 0 then again is the identity.
        Tensor<std::complex<double>> H(2, 2);
        const double s2 = 1.0 / mconst::sqrt2;
        H(0, 0) = s2; H(0, 1) = s2; H(1, 0) = s2; H(1, 1) = -s2;
        Tensor<std::complex<double>> before = psi;
        psi = applyGate(std::move(psi), H, 0);
        psi = applyGate(std::move(psi), H, 0);
        ok(psi.allclose(before, 1e-12, 1e-12), "H applied twice to qubit 0 is the identity");
    }

    // ═══════════════════════════════════════════════════════════════════
    std::cerr.clear();
    std::cout << "\n════════════════════════════════════════\n";
    std::cout << "  " << (checks - failures) << " / " << checks << " checks passed\n";
    if (failures) std::cout << "  \033[31m" << failures << " FAILED\033[0m\n";
    else          std::cout << "  \033[32mall green\033[0m\n";
    std::cout << "════════════════════════════════════════\n";
    return failures ? 1 : 0;
}
