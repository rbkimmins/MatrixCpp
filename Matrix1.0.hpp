// #include <initializer_list>
// #include <utility>
#include "random.hpp"
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>   // std::min
#include <limits>      // std::numeric_limits
#include <tuple>       // std::tuple, std::make_tuple
#ifdef _OPENMP
#include <omp.h>
#endif

// Slice sentinel — used in place of all for row/column extraction.
// A dedicated type prevents ambiguity with operator()(int, int).
// Usage: A(i, all)  or  A(all, j)
struct all_t {};
inline constexpr all_t all;
//for std=C++11
//constexpr all_t all;
template <typename datatype>
class Matrix{
    public:
        // --- Constructors ---

        // Default: creates an empty 0x0 matrix
        Matrix(){
            rowSize = 0;
            colSize = 0;
            grid = nullptr;
        }
        // Creates an i x j matrix, zero-initialised
        Matrix(long i, long j){
            if (i < 0 || j < 0)
                throw std::invalid_argument(
                    "Matrix: dimensions must be non-negative, got " +
                    std::to_string(i) + "x" + std::to_string(j));
            // Indexing operators use signed int, so both dimensions and their product
            // must fit within INT_MAX to guarantee every element is reachable.
            static constexpr long MAX_IDX = std::numeric_limits<int>::max();
            if (i > MAX_IDX || j > MAX_IDX || i * j > MAX_IDX)
                throw std::invalid_argument(
                    "Matrix: dimensions " + std::to_string(i) + "x" + std::to_string(j) +
                    " exceed the maximum indexable size (" + std::to_string(MAX_IDX) + ")");
            rowSize = i;
            colSize = j;
            grid = new datatype[rowSize*colSize]();
        }
        // Copy constructor: deep copies M
        Matrix(const Matrix &M){
            rowSize = M.rowSize;
            colSize = M.colSize;
            grid = new datatype[rowSize*colSize]();
            for (long index = 0; index < rowSize*colSize; index++)
                grid[index] = M.grid[index];
        }

        // --- Assignment operators ---

        // Assigns from another Matrix (deep copy)
        Matrix& operator=(const Matrix &M){
            if (this == &M) return *this;  // self-assignment guard

            delete[] grid; // empty the Matrix
            rowSize = M.rowSize;
            colSize = M.colSize;
            grid = new datatype[rowSize*colSize];
            for (long index = 0; index < rowSize*colSize; index++)
                grid[index] = M.grid[index];
            return *this;
        }
        // Assigns from a 2D initializer list, e.g. A = {{1,2},{3,4}}
        Matrix& operator=(const std::initializer_list<std::initializer_list<datatype>> &M){
            try{
                if (M.size() == 0)
                    throw std::invalid_argument("Cannot assign empty initializer list to Matrix");

                delete[] grid;

                rowSize = M.size();
                auto itr = M.begin();
                colSize = itr->size();

                grid = new datatype[rowSize*colSize];         // deep copy

                int i = 0, j = 0, index = 0;
                for (auto row : M){
                    for (auto element : row){
                        index = i * colSize + (j++ % colSize);
                        grid[index] = element;
                    }
                    i++;
                }
            }catch(const std::exception& e){
                std::cerr << "Matrix assignment error: " << e.what() << std::endl;
                throw;
            }
            return *this;
        }

        // --- Arithmetic operators ---

        // Element-wise addition. Requires identical dimensions. Returns a new Matrix.
        Matrix operator+(const Matrix &M){
            Matrix ans = *this;
            try{
                if (this->colSize != M.colSize)
                    throw std::invalid_argument(
                        "Column size mismatch in operator+: " + std::to_string(colSize) + " != " + std::to_string(M.colSize));
                if (this->rowSize != M.rowSize)
                    throw std::invalid_argument(
                        "Row size mismatch in operator+: " + std::to_string(rowSize) + " != " + std::to_string(M.rowSize));

                for (long index = 0; index < colSize*rowSize; index++)
                    ans.grid[index] += M.grid[index];

            }catch(const std::exception& e){
                std::cerr << "Matrix addition error: " << e.what() << std::endl;
                throw;
            }
            return ans;
        }

        // Scalar multiplication: multiplies every element by num. Returns a new Matrix.
        template <typename scalar>
        Matrix operator*(const scalar& num) const {
            Matrix ans = *this;
            for (long index = 0; index < rowSize*colSize; index++)
                ans.grid[index] *= num;
            return ans;
        }
        // In-place scalar multiplication
        template <typename scalar>
        Matrix& operator*= (const scalar& num) {
            (*this) = (*this) * num;
            return (*this);
        }

        // Matrix multiplication (dot product). Requires this->cols == M.rows.
        // Uses Strassen-Winograd for square matrices >= STRASSEN_THRESHOLD,
        // falling back to naive O(n³) for smaller or rectangular matrices.
        Matrix operator*(const Matrix &M) const{
            try{
                if (this->colSize != M.rowSize)
                    throw std::invalid_argument(
                        "Inner dimensions must match for operator*: (" +
                        std::to_string(rowSize) + "x" + std::to_string(colSize) + ") * (" +
                        std::to_string(M.rowSize) + "x" + std::to_string(M.colSize) + ")");

                // Strassen-Winograd path: both operands must be square and same size.
                // Only pad to the next power of 2 when the overhead is modest (< 2×).
                // Beyond that, the extra recursion level costs more than naive saves.
                if (rowSize == colSize && M.rowSize == M.colSize && rowSize == M.rowSize
                    && rowSize >= STRASSEN_THRESHOLD) {
                    long sz = nextPow2(rowSize);
                    if (sz == rowSize)
                        return strassenWinograd(*this, M);
                    // Only use Strassen if padding stays within 41% overhead (i.e. the
                    // padded size is at most sqrt(2)*n ≈ 1.41n, keeping work < 2×).
                    // Otherwise fall through to naive — the plateau jump is not worth it.
                    if (sz <= rowSize * 3 / 2) {
                        Matrix Ap(sz, sz), Bp(sz, sz);
                        long total = rowSize * colSize;
                        for (long k = 0; k < total; k++) {
                            Ap.grid[k / colSize * sz + k % colSize] = grid[k];
                            Bp.grid[k / M.colSize * sz + k % M.colSize] = M.grid[k];
                        }
                        Matrix Cp = strassenWinograd(Ap, Bp);
                        Matrix ans(rowSize, rowSize);
                        long ansTotal = rowSize * rowSize;
                        for (long k = 0; k < ansTotal; k++)
                            ans.grid[k] = Cp.grid[k / rowSize * sz + k % rowSize];
                        return ans;
                    }
                }

                return naiveMul(*this, M);

            }catch(const std::exception& e){
                std::cerr << "Matrix multiplication error: " << e.what() << std::endl;
                throw;
            }
        }

        // In-place matrix multiplication
        Matrix& operator*= (const Matrix &M){
            (*this) = (*this) * M;
            return (*this);
        }

        // Hadamard (element-wise) product. Requires identical dimensions. Returns a new Matrix.
        Matrix operator%(const Matrix &M) const{
            try{
                if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                    throw std::invalid_argument(
                        "Dimension mismatch in Hadamard product: (" +
                        std::to_string(rowSize) + "x" + std::to_string(colSize) + ") vs (" +
                        std::to_string(M.rowSize) + "x" + std::to_string(M.colSize) + ")");
                Matrix <datatype> ans(M.rowSize, M.colSize);
                for (long i = 0; i < M.rowSize * M.colSize; i++){
                    ans.grid[i] = this->grid[i] * M.grid[i];
                }
                return ans;
            }catch(const std::exception& e){
                std::cerr << "Hadamard product error: " << e.what() << std::endl;
                throw;
            }
        }
        // Integer modulo: applies modulo to every element. Returns a new Matrix.
        // e.g. A % 3 gives a matrix where each element is a_ij % 3.
        // Note: n % A has no defined meaning and is not supported.
        Matrix operator% (const int& modulo) const{
            Matrix ans;
            ans = *this;
            for (long i = 0; i < rowSize*colSize; i++) ans.grid[i] %= modulo;
            return ans;
        }

        // In-place integer modulo
        Matrix& operator%= (const int& modulo){
            *this = (*this) % modulo;
            return *this;
        }

        // In-place Hadamard product (Element-wise Matrix multipication)
        Matrix& operator%= (const Matrix& M){
            *this = (*this) % M;
            return *this;
        }

        //Element-wise Matrix division
        Matrix operator/ (const Matrix& M) const{
            try{
                if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                    throw std::invalid_argument(
                        "Dimension mismatch in Element-wise division: (" +
                        std::to_string(rowSize) + "x" + std::to_string(colSize) + ") vs (" +
                        std::to_string(M.rowSize) + "x" + std::to_string(M.colSize) + ")");
                Matrix ans(rowSize, colSize);
                for (long i = 0; i < rowSize * colSize; i++)
                    ans.grid[i] = grid[i] / M.grid[i];
                return ans;
            }catch(const std::exception& e){
                std::cerr << "Element-wise division error: " << e.what() << std::endl;
                throw;
            }
        }

        Matrix& operator/= (const Matrix& M){
            *this = (*this) / M;
            return *this;
        }
        // Scalar division: divides every element by n. Preserves datatype.
        // Note: n / A has no defined meaning and is not supported.
        template <typename scalar>
        Matrix operator/ (const scalar& n) const{
            Matrix ans;
            ans = *this;
            for(long i = 0; i < rowSize*colSize; i++) ans.grid[i] /= n;
            return ans;
        }

        // In-place scalar division
        template <typename scalar>
        Matrix& operator/= (const scalar& n){
            *this = *this / n;
            return *this;
        }

        // Element-wise subtraction. Requires identical dimensions. Returns a new Matrix.
        Matrix operator-(const Matrix &M){
            return *this + M*-1;
        }

        Matrix& operator-= (const Matrix &M){
            *this = (*this) - M;
            return *this;
        }

        //Agrumented Matrix operator, allows for similar math notation.
        //Note, to preserve predence use with (), EX (A|B)
        Matrix operator| (const Matrix &M){
            return (*this).concat(M, 1);
        }
        // --- Proxy classes for slice assignment ---
        // Returned by non-const slice operators. Holds a reference back to the
        // parent Matrix so that A(i, all) = B writes through to A.
        // Implicit Matrix<datatype> conversion lets them be used in read contexts too.

        class RowProxy {
            Matrix& mat;
            int row;
        public:
            RowProxy(Matrix& m, int r) : mat(m), row(r) {}
            // Write: A(i, all) = src  — src must be a (1 x cols) row vector
            RowProxy& operator=(const Matrix<datatype>& src) {
                if (src.rows() != 1 || src.cols() != mat.cols())
                    throw std::invalid_argument(
                        "RowProxy: source must be (1 x " + std::to_string(mat.cols()) +
                        "), got (" + std::to_string(src.rows()) + " x " + std::to_string(src.cols()) + ")");
                for (long j = 0; j < mat.cols(); j++)
                    mat(row, j) = src(0, j);
                return *this;
            }
            // Read: implicit conversion to Matrix for use in expressions
            operator Matrix<datatype>() const {
                Matrix<datatype> ans(1, mat.cols());
                for (long j = 0; j < mat.cols(); j++)
                    ans[j] = mat(row, j);
                return ans;
            }
            friend std::ostream& operator<<(std::ostream& os, const RowProxy& p) {
                return os << static_cast<Matrix<datatype>>(p).toString();
            }
        };

        class ColProxy {
            Matrix& mat;
            int col;
        public:
            ColProxy(Matrix& m, int c) : mat(m), col(c) {}
            // Write: A(all, j) = src  — src must be a (rows x 1) column vector
            ColProxy& operator=(const Matrix<datatype>& src) {
                if (src.cols() != 1 || src.rows() != mat.rows())
                    throw std::invalid_argument(
                        "ColProxy: source must be (" + std::to_string(mat.rows()) + " x 1), got (" +
                        std::to_string(src.rows()) + " x " + std::to_string(src.cols()) + ")");
                for (long i = 0; i < mat.rows(); i++)
                    mat(i, col) = src(i, 0);
                return *this;
            }
            // Read: implicit conversion to Matrix for use in expressions
            operator Matrix<datatype>() const {
                Matrix<datatype> ans(mat.rows(), 1);
                for (long i = 0; i < mat.rows(); i++)
                    ans[i] = mat(i, col);
                return ans;
            }
            friend std::ostream& operator<<(std::ostream& os, const ColProxy& p) {
                return os << static_cast<Matrix<datatype>>(p).toString();
            }
        };

        class SubProxy {
            Matrix& mat;
            int r1, c1, rStep, cStep, numRows, numCols;
        public:
            SubProxy(Matrix& m, int r1, int c1, int rStep, int cStep, int numRows, int numCols)
                : mat(m), r1(r1), c1(c1), rStep(rStep), cStep(cStep), numRows(numRows), numCols(numCols) {}
            // Write: A({r1,r2},{c1,c2}) = src  — src must match the slice dimensions
            SubProxy& operator=(const Matrix<datatype>& src) {
                if (src.rows() != numRows || src.cols() != numCols)
                    throw std::invalid_argument(
                        "SubProxy: source is (" + std::to_string(src.rows()) + "x" + std::to_string(src.cols()) +
                        ") but slice is (" + std::to_string(numRows) + "x" + std::to_string(numCols) + ")");
                for (int i = 0; i < numRows; i++)
                    for (int j = 0; j < numCols; j++)
                        mat(r1 + i * rStep, c1 + j * cStep) = src(i, j);
                return *this;
            }
            // Read: implicit conversion to Matrix for use in expressions
            operator Matrix<datatype>() const {
                Matrix<datatype> ans(numRows, numCols);
                for (int idx = 0; idx < numRows * numCols; idx++) {
                    int i = idx / numCols, j = idx % numCols;
                    ans[idx] = mat(r1 + i * rStep, c1 + j * cStep);
                }
                return ans;
            }
            friend std::ostream& operator<<(std::ostream& os, const SubProxy& p) {
                return os << static_cast<Matrix<datatype>>(p).toString();
            }
        };

        // --- Indexing operators ---
        // Note: negative indices wrap backwards (e.g. -1 gives last element).
        // Matrix is indexed as A(i, j) where i = row, j = column (0-based).

        // Returns a reference to element (i, j) — supports A(i,j) = x
        datatype& operator()(const int& i, const int& j){
            return grid[(i % rowSize) * colSize + (j % colSize)];
        }
        // Const element access
        const datatype& operator()(const int& i, const int& j) const {
            return grid[(i % rowSize) * colSize + (j % colSize)];
        }
        // Flat index access into the underlying row-major array — supports A[i] = x
        datatype& operator[](const int& i){
            return grid[i % (rowSize * colSize)];
        }

        // Row extraction — non-const returns RowProxy: supports A(i, all) = B
        RowProxy operator()(const int& i, all_t) {
            int r = ((i % (int)rowSize) + (int)rowSize) % (int)rowSize;
            return RowProxy(*this, r);
        }
        // Row extraction — const returns Matrix by value for reading
        Matrix operator()(const int& i, all_t) const {
            Matrix<datatype> ans(1, this->colSize);
            for (long j = 0; j < this->colSize; j++)
                ans[j] = this->grid[(i % rowSize) * colSize + (j % colSize)];
            return ans;
        }

        // Column extraction — non-const returns ColProxy: supports A(all, j) = B
        ColProxy operator()(all_t, const int& i) {
            int c = ((i % (int)colSize) + (int)colSize) % (int)colSize;
            return ColProxy(*this, c);
        }
        // Column extraction — const returns Matrix by value for reading
        Matrix operator()(all_t, const int& i) const {
            Matrix<datatype> ans(rowSize, 1);
            for (long j = 0; j < rowSize; j++)
                ans[j] = grid[(j % rowSize) * colSize + (i % colSize)];
            return ans;
        }

        // Submatrix — non-const returns SubProxy: supports A({r1,r2},{c1,c2}) = B
        // Negative indices wrap; reversed ranges (e.g. {9,7}) return elements in reverse order.
        SubProxy operator()(std::pair<int,int> rowRange, std::pair<int,int> colRange) {
            int rN = (int)rowSize, cN = (int)colSize;
            int r1 = ((rowRange.first  % rN) + rN) % rN;
            int r2 = ((rowRange.second % rN) + rN) % rN;
            int c1 = ((colRange.first  % cN) + cN) % cN;
            int c2 = ((colRange.second % cN) + cN) % cN;
            int numRows = std::abs(r2 - r1) + 1, numCols = std::abs(c2 - c1) + 1;
            return SubProxy(*this, r1, c1, (r2>=r1)?1:-1, (c2>=c1)?1:-1, numRows, numCols);
        }
        // Submatrix — const returns Matrix by value for reading
        Matrix operator()(std::pair<int,int> rowRange, std::pair<int,int> colRange) const {
            int rN = (int)rowSize, cN = (int)colSize;
            int r1 = ((rowRange.first  % rN) + rN) % rN;
            int r2 = ((rowRange.second % rN) + rN) % rN;
            int c1 = ((colRange.first  % cN) + cN) % cN;
            int c2 = ((colRange.second % cN) + cN) % cN;
            int numRows = std::abs(r2 - r1) + 1, numCols = std::abs(c2 - c1) + 1;
            int rStep = (r2>=r1)?1:-1, cStep = (c2>=c1)?1:-1;
            Matrix<datatype> ans(numRows, numCols);
            for (int idx = 0; idx < numRows * numCols; idx++) {
                int i = idx / numCols, j = idx % numCols;
                ans[idx] = (*this)(r1 + i * rStep, c1 + j * cStep);
            }
            return ans;
        }
        // --- Inspection ---

        // Returns true if the matrix has no elements (0x0 or any zero dimension)
        inline bool empty() const { return !(rowSize * colSize); }
        // Returns the number of rows
        long rows() const { return rowSize; }
        // Returns the number of columns
        long cols() const { return colSize; }

        // --- Printing and string conversion ---

        // Returns a vector of formatted row strings, one per row.
        // Used internally by toString() and printSideBySide().
        // precision: decimal places for floating-point types (ignored for integral types).
        std::vector<std::string> toLines(int precision = 6) const {
            // Pre-pass: format every element to find the widest string
            std::vector<std::string> cells(rowSize * colSize);
            size_t colWidth = 0;
            for (long i = 0; i < rowSize * colSize; i++) {
                std::ostringstream oss;
                if (std::is_floating_point<datatype>::value)
                    oss << std::fixed << std::setprecision(precision);
                oss << grid[i];
                cells[i] = oss.str();
                if (cells[i].size() > colWidth) colWidth = cells[i].size();
            }

            std::vector<std::string> lines(rowSize);
            for (long i = 0; i < rowSize; i++) {
                std::ostringstream row;
                row << "[ ";
                for (long j = 0; j < colSize; j++) {
                    row << std::setw((int)colWidth) << cells[i * colSize + j];
                    if (j < colSize - 1) row << "  ";
                }
                row << " ]";
                lines[i] = row.str();
            }
            return lines;
        }

        // Returns the full matrix as a formatted multi-line string.
        // precision: decimal places (default 6, matching NumPy's default).
        std::string toString(int precision = 6) const {
            auto lines = toLines(precision);
            std::string result;
            for (size_t i = 0; i < lines.size(); i++) {
                result += lines[i];
                if (i + 1 < lines.size()) result += '\n';
            }
            return result;
        }

        // Explicit cast to std::string using default precision (6dp).
        // Use toString(n) directly when a specific precision is needed.
        explicit operator std::string() const { return toString(); }

        // Prints the matrix to stdout. precision: decimal places (default 6).
        void print(int precision = 6) const { std::cout << toString(precision) << '\n'; }

        // --- Linear algebra ---

        // Returns the transpose of this matrix as a new (cols x rows) Matrix.
        Matrix T() const{
            Matrix <datatype> ans(colSize, rowSize);
            for (long i = 0; i < colSize*rowSize; i++){
                //By definition of transpose.
                ans.grid[(i % colSize) * rowSize + (i / colSize)] = grid[i];
            }
            return ans;
        }

        //element-wise power
        template <typename scalar>
        Matrix pow(scalar num) const{
            Matrix ans = *this;
            for (long i = 0; i < (*this).colSize*(*this).rowSize; i++)
                ans[i] = std::pow(ans[i],num);
            return ans;
        }  

        //element-wise exp
        Matrix exp() const{
            Matrix ans = *this;
            for (long i = 0; i < (*this).colSize*(*this).rowSize; i++)
                ans[i] = std::exp(ans[i]);
            return ans;
        }

        //element-wise log base 10
        Matrix log10() const{
            Matrix ans = *this;
            for (long i = 0; i < (*this).colSize*(*this).rowSize; i++)
                ans[i] = std::log10(ans[i]);
            return ans;
        }

        //element-wise log base 2, using computer science notation.
        Matrix lg() const{
            Matrix ans = *this;
            for (long i = 0; i < (*this).colSize*(*this).rowSize; i++)
                ans[i] = std::log2(ans[i]);
            return ans;
        }

        //element-wise natural log
        Matrix ln() const{
            Matrix ans = *this;
            for (long i = 0; i < (*this).colSize*(*this).rowSize; i++)
                ans[i] = std::log(ans[i]);
            return ans;
        }

        //element-wise log with arbitrary base
        template <typename scalar>
        Matrix log(scalar base) const{
            Matrix ans = *this;
            for (long i = 0; i < (*this).colSize*(*this).rowSize; i++)
                ans[i] = std::log2(ans[i])/std::log2(base);
            return ans;
        }

        // Returns whether a matrix is diagonal: all off-diagonal elements are zero.
        // Works for non-square matrices. Single flat loop — no allocations, early exit.
        bool IsDiagonal() const{
            for (long k = 0; k < rowSize * colSize; k++)
                if (k / colSize != k % colSize && grid[k] != datatype(0))
                    return false;
            return true;
        }

        // Returns the sum of the main diagonal elements. Requires a square matrix.
        datatype tr() const{
            try
            {
                if (!(rowSize == colSize && rowSize > 0))
                    throw std::invalid_argument(
                        "tr() requires a square non-empty matrix, got " +
                        std::to_string(rowSize) + "x" + std::to_string(colSize));
                datatype sum = datatype(0);
                for (long i = 0; i < rowSize; i++) sum += (*this)(int(i), int(i));

                return sum;
            }
            catch(const std::exception& e)
            {
                std::cerr << "Matrix trace error: " << e.what() << '\n';
                throw;
            }
        }
        // Returns the sum of all elements in the matrix.
        datatype sum() const{
            datatype total = datatype(0);
            for (long i = 0; i < rowSize * colSize; i++) total += grid[i];
            return total;
        }
        // Dimensional sum. addcol=0: returns a (1 x cols) row matrix of column sums.
        //                  addcol=1: returns a (rows x 1) column matrix of row sums.
        Matrix sum(const bool& addcol) const{
            try{
                if (!addcol) {
                    Matrix<datatype> total_vec(1, colSize);
                    for (long i = 0; i < colSize; i++)
                        total_vec[i] = (*this)(all, int(i)).sum();
                    return total_vec;
                } else {
                    Matrix<datatype> total_vec(rowSize, 1);
                    for (long i = 0; i < rowSize; i++)
                        total_vec[i] = (*this)(int(i), all).sum();
                    return total_vec;
                }
            }catch(const std::exception& e){
                std::cerr << "Matrix summation error: " << e.what() << std::endl;
                throw;
            }
        }
        // Concatenates M to this matrix. concatCol=0: vertical (stack rows, cols must match).
        //                               concatCol=1: horizontal (stack cols, rows must match).
        // If this matrix is empty, returns M directly.
        Matrix concat(const Matrix& M, const bool& concatCol) const{
            try{
                if (this->empty()) return M;
                if (!concatCol) {
                    // vertical concat: stack rows, columns must match
                    if (this->colSize != M.colSize) throw std::invalid_argument(
                        "concat: column size mismatch: " + std::to_string(colSize) +
                        " != " + std::to_string(M.colSize));
                    Matrix<datatype> argumentMatrix(this->rowSize + M.rowSize, colSize);
                    for (long i = 0; i < this->rowSize; i++)
                        for (long j = 0; j < colSize; j++)
                            argumentMatrix(int(i), int(j)) = (*this)(int(i), int(j));
                    for (long i = 0; i < M.rowSize; i++)
                        for (long j = 0; j < M.colSize; j++)
                            argumentMatrix(int(i + this->rowSize), int(j)) = M(int(i), int(j));
                    return argumentMatrix;
                } else {
                    // horizontal concat: stack columns, rows must match
                    if (this->rowSize != M.rowSize) throw std::invalid_argument(
                        "concat: row size mismatch: " + std::to_string(rowSize) +
                        " != " + std::to_string(M.rowSize));
                    Matrix<datatype> argumentMatrix(rowSize, this->colSize + M.colSize);
                    for (long i = 0; i < this->rowSize; i++)
                        for (long j = 0; j < this->colSize; j++)
                            argumentMatrix(int(i), int(j)) = (*this)(int(i), int(j));
                    for (long i = 0; i < M.rowSize; i++)
                        for (long j = 0; j < M.colSize; j++)
                            argumentMatrix(int(i), int(j + this->colSize)) = M(int(i), int(j));
                    return argumentMatrix;
                }
            }catch(const std::exception& e){
                std::cerr << "Matrix concat error: " << e.what() << std::endl;
                throw;
            }
        }

        // Kronecker (tensor) product: replaces every element A_ij with the block A_ij * M.
        // Returns a (rows*M.rows x cols*M.cols) Matrix.
        Matrix tensor(const Matrix& M) const{
            long rM = M.rowSize, cM = M.colSize;
            long cOut = colSize * cM;
            Matrix<datatype> ans(rowSize * rM, cOut);
            for (long idx = 0; idx < ans.rowSize * ans.colSize; idx++){
                long r = idx / cOut, c = idx % cOut;
                ans[idx] = (*this)(int(r / rM), int(c / cM)) * M(int(r % rM), int(c % cM));
            }
            return ans;
        }

        // --- Random initialisation ---

        // Fills every element with a random value in [lowBound, highBound).
        // Seed is derived from the current time (unique within a 24-hour window).
        // Integral types are rounded to the nearest integer.
        // Returns *this to allow chaining.
        Matrix& set_Ran_values(double lowBound, double highBound){
            try
            {
                if(lowBound >= highBound) throw std::invalid_argument(
                    "Improper boundaries given, low: " + std::to_string(lowBound) + " > high: " + std::to_string(highBound));
                double range = highBound - lowBound;
                long seed;
                setRan(seed);
                for (long i = 0; i < rowSize*colSize; i++){
                    double val = range * ran2(&seed) + lowBound;
                    grid[i] = std::is_integral<datatype>::value ? datatype(std::round(val)) : datatype(val);
                }
            }catch(const std::exception& e){
                std::cerr << "Bounds error: " << e.what() << std::endl;
                throw;
            }
            return *this;
        }
        // Fills every element with a random value in [lowBound, highBound) using a custom seed.
        // customSeed MUST be negative (required by ran2 to trigger initialisation).
        // Integral types are rounded to the nearest integer.
        // Returns *this to allow chaining.
        Matrix& set_Ran_values(double lowBound, double highBound, long customSeed){
            try
            {
                if( customSeed >= 0) throw std::invalid_argument(
                    "Seed, " + std::to_string(customSeed) + " is >= 0");
                if(lowBound >= highBound) throw std::invalid_argument(
                    "Improper boundaries given, low: " + std::to_string(lowBound) + " > high: " + std::to_string(highBound));
                double range = highBound - lowBound;
                long seed = customSeed;
                for (long i = 0; i < rowSize*colSize; i++){
                    double val = range * ran2(&seed) + lowBound;
                    grid[i] = std::is_integral<datatype>::value ? datatype(std::round(val)) : datatype(val);
                }
            }catch(const std::exception& e){
                std::cerr << "Error: " << e.what() << std::endl;
                throw;
            }
            return *this;
        }
        //


        // QR factorisation with column pivoting using Householder reflections.
        // Mirrors LAPACK's DGEQP3: at each step the column with the largest remaining
        // norm is pivoted to the front, then a Householder reflector eliminates the
        // sub-diagonal entries of that column. Column norms are maintained via the
        // Bischof-Pan rank-1 downdate, avoiding a full norm recomputation each step.
        //
        // For m×n matrix A, computes  A * P = Q * R  where:
        //   Q — m×m orthogonal (product of Householder reflectors)
        //   R — m×n upper triangular
        //   P — n×n permutation matrix (column pivoting for stability)
        //
        // Works for any m×n, including m < n.
        // Usage: auto [Q, R, P] = A.QR();
        std::tuple<Matrix<double>, Matrix<double>, Matrix<datatype>> QR() const {
            try {
                if (rowSize == 0 || colSize == 0)
                    throw std::invalid_argument("QR: matrix must be non-empty");

                int m = (int)rowSize, n = (int)colSize;
                int r = std::min(m, n);

                // Working copy in double (row-major flat array)
                std::vector<double> work(m * n);
                for (int k = 0; k < m * n; k++) work[k] = double(grid[k]);
                auto W = [&](int i, int j) -> double& { return work[i * n + j]; };

                // Column pivot tracking — pivots[k] = original column index at position k
                std::vector<int> pivots(n);
                std::iota(pivots.begin(), pivots.end(), 0);

                // Squared column norms for Bischof-Pan pivot selection
                std::vector<double> sqNorms(n, 0.0);
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < m; i++) sqNorms[j] += W(i, j) * W(i, j);

                // Householder taus + vectors stored for Q accumulation
                std::vector<double>              taus(r, 0.0);
                std::vector<std::vector<double>> hvecs(r);

                for (int k = 0; k < r; k++) {
                    // ── Pivot: bring the largest-norm remaining column to position k ──
                    int jmax = k;
                    for (int j = k + 1; j < n; j++)
                        if (sqNorms[j] > sqNorms[jmax]) jmax = j;
                    if (jmax != k) {
                        for (int i = 0; i < m; i++) std::swap(W(i, k), W(i, jmax));
                        std::swap(pivots[k],  pivots[jmax]);
                        std::swap(sqNorms[k], sqNorms[jmax]);
                    }

                    // ── Householder reflector for column k, rows k:m-1 ──
                    // Choose alpha opposite in sign to x[0] to avoid cancellation.
                    double xnorm = 0.0;
                    for (int i = k; i < m; i++) xnorm += W(i, k) * W(i, k);
                    xnorm = std::sqrt(xnorm);

                    if (xnorm == 0.0) { hvecs[k].assign(m - k, 0.0); continue; }

                    double alpha = (W(k, k) >= 0.0 ? -1.0 : 1.0) * xnorm;
                    std::vector<double> v(m - k);
                    for (int i = 0; i < m - k; i++) v[i] = W(k + i, k);
                    v[0] -= alpha;  // v = x - alpha*e_1

                    double vTv = 0.0;
                    for (double vi : v) vTv += vi * vi;
                    double tau = 2.0 / vTv;
                    taus[k]  = tau;
                    hvecs[k] = v;

                    // Apply H_k = I - tau*v*v^T to trailing block W(k:m-1, k:n-1)
                    for (int j = k; j < n; j++) {
                        double vTw = 0.0;
                        for (int i = 0; i < m - k; i++) vTw += v[i] * W(k + i, j);
                        for (int i = 0; i < m - k; i++) W(k + i, j) -= tau * v[i] * vTw;
                    }

                    // Bischof-Pan downdate: H_k is orthogonal so column norms are
                    // preserved; the squared norm below row k shrinks by W(k,j)^2.
                    for (int j = k + 1; j < n; j++) {
                        sqNorms[j] -= W(k, j) * W(k, j);
                        if (sqNorms[j] < 0.0) sqNorms[j] = 0.0;
                    }
                }

                // ── Materialise R (upper triangle of the worked array) ──
                Matrix<double> R(m, n);
                for (int i = 0; i < m; i++)
                    for (int j = i; j < n; j++)
                        R(i, j) = W(i, j);

                // ── Accumulate Q = H_0 * H_1 * … * H_{r-1} ──
                // Apply reflectors in reverse order to the m×m identity.
                // At descending step k, columns 0:k-1 of Q are zero in rows k:m-1,
                // so only columns k:m-1 need updating.
                Matrix<double> Q(m, m);
                for (int i = 0; i < m; i++) Q(i, i) = 1.0;
                for (int k = r - 1; k >= 0; k--) {
                    if (taus[k] == 0.0) continue;
                    const auto& v = hvecs[k];
                    int sz = (int)v.size();
                    for (int j = k; j < m; j++) {
                        double vTq = 0.0;
                        for (int i = 0; i < sz; i++) vTq += v[i] * Q(k + i, j);
                        for (int i = 0; i < sz; i++) Q(k + i, j) -= taus[k] * v[i] * vTq;
                    }
                }

                // ── Materialise P: A*P = Q*R, so P[pivots[k], k] = 1 ──
                Matrix<datatype> P(n, n);
                for (int k = 0; k < n; k++) P(pivots[k], k) = datatype(1);

                return std::make_tuple(Q, R, P);

            } catch (const std::exception& e) {
                std::cerr << "QR factorization error: " << e.what() << '\n';
                throw;
            }
        }

        // Performs LU factorization with partial pivoting (Doolittle's method).
        // Requires a square matrix of at least 2x2. Throws if singular.
        // Returns std::tuple<L, U, P> where PA = LU.
        // Usage: auto [L, U, P] = A.LU();
        std::tuple<Matrix<double>, Matrix<double>, Matrix<datatype>> LU() const {
            try {
                auto [packed, pivotVec] = luPacked();
                int n = (int)rowSize;
                auto pat = [&](int i, int j) { return packed[i * n + j]; };

                Matrix<double> L(n, n), U(n, n);
                for (int i = 0; i < n; i++) {
                    L(i, i) = 1.0;
                    for (int j = 0; j < i;  j++) L(i, j) = pat(i, j);
                    for (int j = i; j < n;  j++) U(i, j) = pat(i, j);
                }

                Matrix<datatype> P(n, n);
                for (int i = 0; i < n; i++) P(i, i) = datatype(1);
                for (int k = 0; k < n; k++) {
                    if (pivotVec[k] != k)
                        for (int j = 0; j < n; j++)
                            std::swap(P(k, j), P(pivotVec[k], j));
                }

                return std::make_tuple(L, U, P);
            } catch (const std::exception& e) {
                std::cerr << "LU factorization error: " << e.what() << '\n';
                throw;
            }
        }

        // Returns the determinant via LU factorisation.
        // Integer types are rounded to avoid floating-point drift (e.g. 2.9999 → 3).
        datatype det() const {
            try {
                if (rowSize != colSize)
                    throw std::invalid_argument(
                        "det() requires a square matrix, got " +
                        std::to_string(rowSize) + "x" + std::to_string(colSize));
                auto [packed, pivotVec] = luPacked();
                int n = (int)rowSize;
                int sign = 1;
                for (int k = 0; k < n; k++)
                    if (pivotVec[k] != k) sign = -sign;
                double d = double(sign);
                for (int i = 0; i < n; i++) d *= packed[i * n + i];
                return std::is_integral<datatype>::value
                    ? datatype(std::round(d))
                    : datatype(d);
            } catch (const std::exception& e) {
                std::cerr << "det() error: " << e.what() << '\n';
                throw;
            }
        }

        // Eigendecomposition via the implicit-shift QR algorithm.
        // Usage: auto [eigenvalues, Q] = A.eig();
        //   eigenvalues — n×1 column vector (diagonal of Schur form)
        //   Q           — n×n orthogonal matrix (eigenvectors for symmetric A,
        //                 Schur vectors for general A)
        std::pair<Matrix<double>, Matrix<double>> eig() const {
            try {
                if (rowSize != colSize)
                    throw std::invalid_argument(
                        "eig: matrix must be square, got " +
                        std::to_string(rowSize) + "x" + std::to_string(colSize));
                if (rowSize == 0)
                    throw std::invalid_argument("eig: matrix must be non-empty");
                int n = (int)rowSize;
                auto [H, Qv] = schurDecomp();
                Matrix<double> eigenvals(n, 1), eigenvecs(n, n);
                for (int i = 0; i < n; i++) eigenvals(i, 0) = H[i * n + i];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        eigenvecs(i, j) = Qv[i * n + j];
                return {eigenvals, eigenvecs};
            } catch (const std::exception& e) {
                std::cerr << "eig() error: " << e.what() << '\n';
                throw;
            }
        }

        // Computes A^(-1) via LU factorisation and back-substitution.
        // Solves A * X = I column by column. Requires square, non-singular matrix.
        Matrix<double> inverse() const {
            try {
                if (rowSize != colSize)
                    throw std::invalid_argument(
                        "inverse: matrix must be square, got " +
                        std::to_string(rowSize) + "x" + std::to_string(colSize));
                auto [packed, pivots] = luPacked();
                int n = (int)rowSize;
                Matrix<double> inv(n, n);
                for (int col = 0; col < n; col++) {
                    std::vector<double> b(n, 0.0);
                    b[col] = 1.0;
                    // Apply row permutations from LU pivoting
                    for (int i = 0; i < n; i++)
                        if (pivots[i] != i) std::swap(b[i], b[pivots[i]]);
                    // Forward substitution: L y = b  (L has unit diagonal)
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < i; j++)
                            b[i] -= packed[i * n + j] * b[j];
                    // Backward substitution: U x = y
                    for (int i = n - 1; i >= 0; i--) {
                        for (int j = i + 1; j < n; j++)
                            b[i] -= packed[i * n + j] * b[j];
                        b[i] /= packed[i * n + i];
                    }
                    for (int i = 0; i < n; i++) inv(i, col) = b[i];
                }
                return inv;
            } catch (const std::exception& e) {
                std::cerr << "inverse() error: " << e.what() << '\n';
                throw;
            }
        }

        //deconstructor
        ~Matrix(){
            if (grid) delete[] grid;
        }
    private:
        long rowSize;
        long colSize;
        datatype *grid;

        // Computes the real Schur decomposition of this matrix.
        // Returns {T_flat, Q_flat} where A = Q * T * Q^T,
        // T is upper (quasi-)triangular and Q is orthogonal (both n×n, row-major double).
        // Public so that the free pow() function can access it; also useful on its own.
        public:
        std::pair<std::vector<double>, std::vector<double>> schurDecomp() const {
            int n = (int)rowSize;
            std::vector<double> H(n * n), Q(n * n, 0.0);
            for (int k = 0; k < n * n; k++) H[k] = double(grid[k]);
            for (int i = 0; i < n; i++) Q[i * n + i] = 1.0;
            auto h  = [&](int i, int j) -> double& { return H[i * n + j]; };
            auto qv = [&](int i, int j) -> double& { return Q[i * n + j]; };

            // ── Hessenberg reduction ──────────────────────────────────────────
            for (int k = 0; k < n - 2; k++) {
                double xn = 0.0;
                for (int i = k + 1; i < n; i++) xn += h(i, k) * h(i, k);
                xn = std::sqrt(xn);
                if (xn < 1e-14) continue;
                int sz = n - k - 1;
                double alpha = (h(k + 1, k) >= 0.0 ? -1.0 : 1.0) * xn;
                std::vector<double> v(sz);
                for (int i = 0; i < sz; i++) v[i] = h(k + 1 + i, k);
                v[0] -= alpha;
                double vv = 0.0;
                for (double vi : v) vv += vi * vi;
                if (vv < 1e-28) continue;
                double tau = 2.0 / vv;
                for (int j = k; j < n; j++) {
                    double s = 0.0;
                    for (int i = 0; i < sz; i++) s += v[i] * h(k + 1 + i, j);
                    for (int i = 0; i < sz; i++) h(k + 1 + i, j) -= tau * v[i] * s;
                }
                for (int i = 0; i < n; i++) {
                    double s = 0.0;
                    for (int j = 0; j < sz; j++) s += h(i, k + 1 + j) * v[j];
                    for (int j = 0; j < sz; j++) h(i, k + 1 + j) -= tau * v[j] * s;
                }
                for (int i = 0; i < n; i++) {
                    double s = 0.0;
                    for (int j = 0; j < sz; j++) s += qv(i, k + 1 + j) * v[j];
                    for (int j = 0; j < sz; j++) qv(i, k + 1 + j) -= tau * v[j] * s;
                }
            }

            // ── QR iteration with Wilkinson shift and deflation ───────────────
            const double eps = std::numeric_limits<double>::epsilon();
            int maxSteps = 30 * n, ihi = n - 1;
            while (ihi >= 1) {
                int ilo = ihi;
                while (ilo > 0 &&
                       std::abs(h(ilo, ilo - 1)) >
                       eps * (std::abs(h(ilo - 1, ilo - 1)) + std::abs(h(ilo, ilo))))
                    ilo--;
                if (ilo == ihi) { ihi--; continue; }
                if (maxSteps-- < 0)
                    throw std::runtime_error("schurDecomp: QR iteration did not converge");
                double a = h(ihi-1,ihi-1), b = h(ihi-1,ihi);
                double c = h(ihi,ihi-1),   d = h(ihi,ihi);
                double tr2 = (a+d)/2.0, disc = tr2*tr2 - (a*d - b*c);
                double sigma = (disc >= 0.0)
                    ? (std::abs(tr2+std::sqrt(disc)-d) < std::abs(tr2-std::sqrt(disc)-d)
                       ? tr2+std::sqrt(disc) : tr2-std::sqrt(disc))
                    : d;
                int sz = ihi - ilo + 1;
                std::vector<double> tv(sz, 0.0);
                std::vector<std::vector<double>> hv(sz);
                for (int i = ilo; i <= ihi; i++) h(i,i) -= sigma;
                for (int k = ilo; k < ihi; k++) {
                    int ki = k-ilo, rows = ihi-k+1;
                    double xn = 0.0;
                    for (int i = k; i <= ihi; i++) xn += h(i,k)*h(i,k);
                    xn = std::sqrt(xn);
                    if (xn < 1e-14) { hv[ki].assign(rows,0.0); continue; }
                    double alpha = (h(k,k)>=0.0?-1.0:1.0)*xn;
                    std::vector<double> v(rows);
                    for (int i = 0; i < rows; i++) v[i] = h(k+i,k);
                    v[0] -= alpha;
                    double vv = 0.0;
                    for (double vi : v) vv += vi*vi;
                    if (vv < 1e-28) { hv[ki].assign(rows,0.0); continue; }
                    double tau = 2.0/vv;
                    tv[ki] = tau; hv[ki] = v;
                    for (int j = k; j < n; j++) {
                        double s = 0.0;
                        for (int i = 0; i < rows; i++) s += v[i]*h(k+i,j);
                        for (int i = 0; i < rows; i++) h(k+i,j) -= tau*v[i]*s;
                    }
                }
                for (int ki = 0; ki < sz-1; ki++) {
                    int k = ilo+ki;
                    if (tv[ki] == 0.0) continue;
                    const auto& v = hv[ki];
                    int rows = (int)v.size();
                    for (int i = 0; i < n; i++) {
                        double s = 0.0;
                        for (int j = 0; j < rows; j++) s += h(i,k+j)*v[j];
                        for (int j = 0; j < rows; j++) h(i,k+j) -= tv[ki]*v[j]*s;
                    }
                    for (int i = 0; i < n; i++) {
                        double s = 0.0;
                        for (int j = 0; j < rows; j++) s += qv(i,k+j)*v[j];
                        for (int j = 0; j < rows; j++) qv(i,k+j) -= tv[ki]*v[j]*s;
                    }
                }
                for (int i = ilo; i <= ihi; i++) h(i,i) += sigma;
                if (std::abs(h(ihi,ihi-1)) <= eps*(std::abs(h(ihi-1,ihi-1))+std::abs(h(ihi,ihi)))) {
                    h(ihi,ihi-1) = 0.0;
                    ihi--;
                }
            }
            return {H, Q};
        }

        // Shared Doolittle factorisation used by both LU() and det().
        // Returns {packedData, pivotVec} — packed lower/upper triangle + row-swap record.
        std::pair<std::vector<double>, std::vector<int>> luPacked() const {
            if (rowSize != colSize)
                throw std::invalid_argument(
                    "LU: matrix must be square, got " +
                    std::to_string(rowSize) + "x" + std::to_string(colSize));
            if (rowSize < 2)
                throw std::invalid_argument(
                    "LU: matrix must be at least 2x2, got " +
                    std::to_string(rowSize) + "x" + std::to_string(colSize));

            int n = (int)rowSize;
            std::vector<double> packed(n * n);
            for (int k = 0; k < n * n; k++) packed[k] = double(grid[k]);

            auto pat = [&](int i, int j) -> double& { return packed[i * n + j]; };

            std::vector<int> pivotVec(n);
            for (int i = 0; i < n; i++) pivotVec[i] = i;

            for (int k = 0; k < n; k++) {
                int maxRow = k;
                double maxVal = std::abs(pat(k, k));
                for (int i = k + 1; i < n; i++) {
                    double v = std::abs(pat(i, k));
                    if (v > maxVal) { maxVal = v; maxRow = i; }
                }
                if (maxVal == 0.0)
                    throw std::runtime_error(
                        "LU: zero pivot in column " + std::to_string(k) +
                        " — matrix is singular");
                if (maxRow != k)
                    for (int j = 0; j < n; j++)
                        std::swap(pat(k, j), pat(maxRow, j));
                pivotVec[k] = maxRow;
                for (int i = k + 1; i < n; i++) pat(i, k) /= pat(k, k);
                for (int i = k + 1; i < n; i++)
                    for (int j = k + 1; j < n; j++)
                        pat(i, j) -= pat(i, k) * pat(k, j);
            }
            return {packed, pivotVec};
        }

        // Crossover point: matrices smaller than this use naive O(n³) multiplication.
        // 64 is a common empirical choice — below this the Strassen overhead outweighs
        // the asymptotic benefit.
        static constexpr long STRASSEN_THRESHOLD = 64;

        // Returns the smallest power of 2 >= n.
        static long nextPow2(long n) {
            long p = 1;
            while (p < n) p <<= 1;
            return p;
        }

        // Cache-blocked, OpenMP-parallelised matrix multiplication.
        // Tile size: 64 elements × sizeof(datatype) fits comfortably in L1 cache.
        // Each outer ii-tile is an independent OpenMP task, so cores don't share work.
        // Used as the base case for Strassen-Winograd and for rectangular matrices.
        static Matrix naiveMul(const Matrix& A, const Matrix& B) {
            const long M  = A.rowSize;
            const long K  = A.colSize;
            const long N  = B.colSize;
            Matrix ans(M, N);
            constexpr long BLOCK = 64;

            #pragma omp parallel for schedule(dynamic) shared(ans)
            for (long ii = 0; ii < M; ii += BLOCK)
                for (long kk = 0; kk < K; kk += BLOCK)
                    for (long jj = 0; jj < N; jj += BLOCK)
                        for (long i = ii; i < std::min(ii + BLOCK, M); i++)
                            for (long k = kk; k < std::min(kk + BLOCK, K); k++) {
                                const datatype aik = A.grid[i * K + k];
                                for (long j = jj; j < std::min(jj + BLOCK, N); j++)
                                    ans.grid[i * N + j] += aik * B.grid[k * N + j];
                            }
            return ans;
        }

        // Extract the h×h sub-block of M starting at (r0, c0).
        static Matrix subBlock(const Matrix& M, long r0, long c0, long h) {
            Matrix out(h, h);
            for (long i = 0; i < h; i++)
                for (long j = 0; j < h; j++)
                    out.grid[i * h + j] = M.grid[(r0 + i) * M.colSize + (c0 + j)];
            return out;
        }

        // Write block src (h×h) into dst at (r0, c0).
        static void setBlock(Matrix& dst, const Matrix& src, long r0, long c0, long h) {
            for (long i = 0; i < h; i++)
                for (long j = 0; j < h; j++)
                    dst.grid[(r0 + i) * dst.colSize + (c0 + j)] = src.grid[i * h + j];
        }

        // Element-wise addition of two same-size matrices.
        static Matrix addMat(const Matrix& A, const Matrix& B) {
            Matrix out(A.rowSize, A.colSize);
            for (long k = 0; k < A.rowSize * A.colSize; k++)
                out.grid[k] = A.grid[k] + B.grid[k];
            return out;
        }

        // Element-wise subtraction.
        static Matrix subMat(const Matrix& A, const Matrix& B) {
            Matrix out(A.rowSize, A.colSize);
            for (long k = 0; k < A.rowSize * A.colSize; k++)
                out.grid[k] = A.grid[k] - B.grid[k];
            return out;
        }

        // Strassen-Winograd algorithm.
        //
        // Requires A and B to be square with size = power of 2.
        // Recursively splits into h×h quadrants and computes 7 recursive products
        // (versus 8 for standard multiplication).
        //
        // Winograd's variant saves 4 additions over the original Strassen formulation
        // by precomputing row/column auxiliary sums:
        //
        //   s1 = A21 + A22        t1 = B12 - B11
        //   s2 = s1  - A11        t2 = B22 - t1
        //   s3 = A11 - A21        t3 = B22 - B12
        //   s4 = A12 - s2         t4 = B21 - t2
        //
        //   P1 = A11 * B11        P2 = A12 * B21
        //   P3 = s4   * B22       P4 = A22 * t4
        //   P5 = s1   * t1        P6 = s2  * t2
        //   P7 = s3   * t3
        //
        //   C11 = P1 + P2
        //   C12 = P1 + P6 + P5 + P3
        //   C21 = P1 + P4 + P7 - P6   (note: P1 shared by all quadrants)
        //   Wait — the exact Winograd recurrence uses U1..U7 combinations below.
        //
        // Using the standard Winograd-Strassen combination table:
        //   U1 = P1 + P2
        //   U2 = P1 + P6
        //   U3 = U2 + P7
        //   U4 = U2 + P5
        //   U5 = U4 + P3
        //   C11 = U1            C12 = U5
        //   C21 = U3 - P4       C22 = U4 + P4  -- wait, let me use the canonical form
        //
        // Using Coppersmith-Winograd / Laderman variant as commonly implemented:
        static Matrix strassenWinograd(const Matrix& A, const Matrix& B) {
            long n = A.rowSize;

            // Base case: fall back to naive multiplication
            if (n <= STRASSEN_THRESHOLD)
                return naiveMul(A, B);

            long h = n / 2;

            // Partition A into quadrants
            Matrix A11 = subBlock(A,  0,  0, h);
            Matrix A12 = subBlock(A,  0,  h, h);
            Matrix A21 = subBlock(A,  h,  0, h);
            Matrix A22 = subBlock(A,  h,  h, h);

            // Partition B into quadrants
            Matrix B11 = subBlock(B,  0,  0, h);
            Matrix B12 = subBlock(B,  0,  h, h);
            Matrix B21 = subBlock(B,  h,  0, h);
            Matrix B22 = subBlock(B,  h,  h, h);

            // Winograd auxiliary sums (saves additions vs plain Strassen)
            Matrix S1  = addMat(A21, A22);          // A21 + A22
            Matrix S2  = subMat(S1,  A11);          // S1  - A11
            Matrix S3  = subMat(A11, A21);          // A11 - A21
            Matrix S4  = subMat(A12, S2);           // A12 - S2
            Matrix T1  = subMat(B12, B11);          // B12 - B11
            Matrix T2  = subMat(B22, T1);           // B22 - T1
            Matrix T3  = subMat(B22, B12);          // B22 - B12
            Matrix T4  = subMat(B21, T2);           // B21 - T2

            // 7 recursive multiplications
            Matrix P1  = strassenWinograd(A11, B11);
            Matrix P2  = strassenWinograd(A12, B21);
            Matrix P3  = strassenWinograd(S4,  B22);
            Matrix P4  = strassenWinograd(A22, T4);
            Matrix P5  = strassenWinograd(S1,  T1);
            Matrix P6  = strassenWinograd(S2,  T2);
            Matrix P7  = strassenWinograd(S3,  T3);

            // Combine into result quadrants
            //   C11 = P1 + P2
            //   C12 = P1 + P3 + P5 + P6
            //   C21 = P1 + P4 - P6 + P7   (equivalently U3 - P4 with a sign flip variant)
            //   C22 = P1 + P3 + P4 - P5 + (sign variant) -- use the standard table:
            // Standard Winograd-Strassen result quadrants:
            //   U1  = P1 + P2
            //   U2  = P1 + P6
            //   U3  = U2 + P7
            //   U4  = U2 + P5
            //   U5  = U4 + P3
            //   U6  = U3 - P4
            //   U7  = U4 + P4
            //   C11 = U1,  C12 = U5,  C21 = U6,  C22 = U7
            Matrix U1  = addMat(P1, P2);
            Matrix U2  = addMat(P1, P6);
            Matrix U3  = addMat(U2, P7);
            Matrix U4  = addMat(U2, P5);
            Matrix U5  = addMat(U4, P3);
            Matrix U6  = subMat(U3, P4);
            Matrix U7  = addMat(U4, P4);

            // Assemble the n×n result from its four h×h quadrants
            Matrix C(n, n);
            setBlock(C, U1,  0,  0, h);   // C11
            setBlock(C, U5,  0,  h, h);   // C12
            setBlock(C, U6,  h,  0, h);   // C21
            setBlock(C, U7,  h,  h, h);   // C22
            return C;
        }
};

// pow(A, p) — matrix power A^p (not element-wise; use A.pow(p) for that).
//
// Integer p  — binary exponentiation using operator* (Strassen-accelerated).
//              Negative integers use A.inverse() then repeated squaring.
//
// Real p     — Higham Schur-Padé algorithm:
//   1. Schur decompose:  A = Q T Q^T
//   2. Compute T^p via Parlett recurrence on the upper triangular T
//      (diagonal entries λᵢ^p; super-diagonals via the commutativity equation T·F = F·T)
//   3. Return Q * T^p * Q^T
//
// Requires square matrix. For real p, all eigenvalues must be positive
// (negative eigenvalues with non-integer p yield complex results — an exception is thrown).
//
// Usage: auto Ahalf = pow(A, 0.5);   // matrix square root
//        auto Ainv  = pow(A, -1);    // same as A.inverse()
//        auto A3    = pow(A, 3);     // A * A * A  via binary squaring
template<typename datatype, typename scalar>
Matrix<double> pow(const Matrix<datatype>& A, scalar p) {
    if (A.rows() != A.cols())
        throw std::invalid_argument(
            "pow: matrix must be square, got " +
            std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
    int n = (int)A.rows();

    // Convert to double for consistent arithmetic
    Matrix<double> Ad(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Ad(i, j) = double(A(i, j));

    auto identity = [&]() {
        Matrix<double> I(n, n);
        for (int i = 0; i < n; i++) I(i, i) = 1.0;
        return I;
    };

    // ── Integer fast path: binary exponentiation ──────────────────────────────
    long ip = (long)std::round(double(p));
    if (std::abs(double(p) - double(ip)) < 1e-9) {
        if (ip == 0) return identity();
        if (ip == 1) return Ad;
        Matrix<double> base = (ip < 0) ? Ad.inverse() : Ad;
        Matrix<double> result = identity();
        for (long exp = std::abs(ip); exp > 0; exp >>= 1) {
            if (exp & 1) result = result * base;
            if (exp > 1) base = base * base;
        }
        return result;
    }

    // ── Real power: Schur-Padé via Parlett recurrence ─────────────────────────
    // Step 1: Schur decompose Ad = Q * T * Q^T
    auto [Tv, Qv] = Ad.schurDecomp();
    auto T  = [&](int i, int j) -> double  { return Tv[i * n + j]; };
    auto Qm = [&](int i, int j) -> double  { return Qv[i * n + j]; };

    // Step 2: Parlett recurrence for F = T^p (upper triangular)
    // Diagonal: F[i,i] = T[i,i]^p  (eigenvalue must be positive for real result)
    std::vector<double> F(n * n, 0.0);
    auto f = [&](int i, int j) -> double& { return F[i * n + j]; };

    for (int i = 0; i < n; i++) {
        double lam = T(i, i);
        if (lam <= 0.0)
            throw std::domain_error(
                "pow: eigenvalue " + std::to_string(lam) +
                " is non-positive — real matrix power undefined for non-integer exponent");
        f(i, i) = std::pow(lam, double(p));
    }

    // Super-diagonals via commutativity equation T·F = F·T → Parlett recurrence.
    // For distinct eigenvalues (|λᵢ - λⱼ| > ε):
    //   F[i,j] = (T[i,j]·(F[i,i]−F[j,j]) + Σ_{k=i+1}^{j-1}(F[i,k]·T[k,j] − T[i,k]·F[k,j]))
    //            / (T[i,i] − T[j,j])
    // For repeated eigenvalues (|λᵢ−λⱼ| < relative eps), the standard formula
    // has a near-zero denominator. Use the L'Hôpital limit instead:
    //   lim_{λⱼ→λᵢ} (f(λᵢ)−f(λⱼ))/(λᵢ−λⱼ) = f'(λᵢ) = p·λᵢ^(p−1)
    // This is exact for adjacent super-diagonals (d=1) and a good approximation
    // for d>1 because when eigenvalues are equal the inner sum is also zero.
    for (int d = 1; d < n; d++) {
        for (int i = 0; i < n - d; i++) {
            int j = i + d;
            double num = T(i, j) * (f(i, i) - f(j, j));
            for (int k = i + 1; k < j; k++)
                num += f(i, k) * T(k, j) - T(i, k) * f(k, j);
            double denom = T(i, i) - T(j, j);
            double scale = std::max(std::abs(T(i, i)), std::abs(T(j, j)));
            if (std::abs(denom) > std::numeric_limits<double>::epsilon() * 1e4 * scale)
                f(i, j) = num / denom;
            else
                f(i, j) = T(i, j) * double(p) * std::pow(T(i, i), double(p) - 1.0);
        }
    }

    // Step 3: A^p = Q·F·Qᵀ  —  two O(n³) multiplications, not one O(n⁴) loop
    Matrix<double> Fm(n, n), Qmat(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            Fm(i, j)   = f(i, j);
            Qmat(i, j) = Qm(i, j);
        }
    return Qmat * Fm * Qmat.T();
}

// log(A, base) — matrix logarithm in an arbitrary base (free function).
// Distinct from A.log(base) which is the element-wise member function.
//
// Algorithm: Schur-Padé with f(x) = ln(x) / ln(base)
//   1. Schur decompose:  A = Q T Qᵀ
//   2. Compute F = ln(T) via Parlett recurrence on upper triangular T:
//        diagonal:       F[i,i] = ln(T[i,i])           (eigenvalue must be > 0)
//        super-diagonals: same recurrence as pow() with f'(λ) = 1/λ
//   3. Return Q · (F / ln(base)) · Qᵀ
//
// For symmetric (diagonalizable) A the Schur form is diagonal, so Parlett
// collapses to element-wise log on the eigenvalues — exact with no extra cost.
//
// Requires: square matrix, all eigenvalues positive, base > 0 and base ≠ 1.
// Usage: auto L2 = log(A, 2.0);   // log base-2 of matrix A
//        auto Le = log(A, M_E);   // natural matrix logarithm
template<typename datatype, typename scalar>
Matrix<double> log(const Matrix<datatype>& A, scalar base) {
    if (A.rows() != A.cols())
        throw std::invalid_argument(
            "log: matrix must be square, got " +
            std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
    if (double(base) <= 0.0 || double(base) == 1.0)
        throw std::invalid_argument("log: base must be positive and not equal to 1");

    int n = (int)A.rows();
    Matrix<double> Ad(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Ad(i, j) = double(A(i, j));

    // ── Schur decompose: A = Q T Qᵀ ─────────────────────────────────────────
    auto [Tv, Qv] = Ad.schurDecomp();
    auto T  = [&](int i, int j) { return Tv[i * n + j]; };
    auto Qm = [&](int i, int j) { return Qv[i * n + j]; };

    // ── Parlett recurrence for F = ln(T), f(x)=ln(x), f'(x)=1/x ────────────
    std::vector<double> F(n * n, 0.0);
    auto f = [&](int i, int j) -> double& { return F[i * n + j]; };

    for (int i = 0; i < n; i++) {
        double lam = T(i, i);
        if (lam <= 0.0)
            throw std::domain_error(
                "log: eigenvalue " + std::to_string(lam) +
                " is non-positive — matrix logarithm is not real-valued");
        f(i, i) = std::log(lam);
    }

    for (int d = 1; d < n; d++) {
        for (int i = 0; i < n - d; i++) {
            int j = i + d;
            double num = T(i, j) * (f(i, i) - f(j, j));
            for (int k = i + 1; k < j; k++)
                num += f(i, k) * T(k, j) - T(i, k) * f(k, j);
            double denom = T(i, i) - T(j, j);
            double scale = std::max(std::abs(T(i, i)), std::abs(T(j, j)));
            if (std::abs(denom) > std::numeric_limits<double>::epsilon() * 1e4 * scale)
                f(i, j) = num / denom;
            else
                f(i, j) = T(i, j) / T(i, i);   // f'(λ) = 1/λ for ln
        }
    }

    // ── A^log = Q · (F/ln(base)) · Qᵀ  ─────────────────────────────────────
    double logBase = std::log(double(base));
    Matrix<double> Fm(n, n), Qmat(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            Fm(i, j)   = f(i, j) / logBase;
            Qmat(i, j) = Qm(i, j);
        }
    return Qmat * Fm * Qmat.T();
}

// tr(A) — sum of the main diagonal elements. Mirrors mathematical notation.
template<typename datatype>
datatype tr(const Matrix<datatype>& A) { return A.tr(); }

// det(A) — determinant. Mirrors mathematical notation.
template<typename datatype>
datatype det(const Matrix<datatype>& A) { return A.det(); }



// Scalar multiplication with scalar on the left: k * A.
// Complements the member operator A * k so both orderings work.
template<typename datatype, typename scalar>
Matrix<datatype> operator*(const scalar k, Matrix<datatype> A) {
    return A * k;
}


// Stream insertion: allows std::cout << A and writing to any std::ostream.
// Uses default precision (6dp). For custom precision call A.toString(n) directly.
template<typename datatype>
std::ostream& operator<<(std::ostream& os, const Matrix<datatype>& M) {
    return os << M.toString();
}

// Prints two matrices side by side with an operator symbol centred on the middle row.
// A:        left matrix
// op:       operator string shown between them, e.g. "*", "+", "="
// B:        right matrix
// precision: decimal places for floating-point types (default 6)
// Handles mismatched row counts by padding the shorter matrix with blank lines.
template<typename datatype>
void printSideBySide(const Matrix<datatype>& A, const std::string& op,
                     const Matrix<datatype>& B, int precision = 6) {
    auto linesA = A.toLines(precision);
    auto linesB = B.toLines(precision);

    size_t rowsA = linesA.size();
    size_t rowsB = linesB.size();
    size_t totalRows = rowsA > rowsB ? rowsA : rowsB;

    // Width of a blank line matching A's and B's row width
    size_t widthA = rowsA > 0 ? linesA[0].size() : 0;
    size_t widthB = rowsB > 0 ? linesB[0].size() : 0;
    std::string blankA(widthA, ' ');
    std::string blankB(widthB, ' ');

    // op column: symbol on middle row, spaces elsewhere
    size_t midRow = totalRows / 2;
    std::string opPad(op.size(), ' ');

    for (size_t r = 0; r < totalRows; r++) {
        const std::string& rowA = r < rowsA ? linesA[r] : blankA;
        const std::string& rowB = r < rowsB ? linesB[r] : blankB;
        const std::string& sym  = r == midRow ? op : opPad;
        std::cout << rowA << "   " << sym << "   " << rowB << '\n';
    }
}

// ============================================================
// IdentityMatrix — lazy proxy for k*I
//   n == 0  : dynamic  (size inferred when used in an expression)
//   n  > 0  : fixed    (materialized via I(n) assignment)
//   scale   : multiplier (supports k*I, I+I, etc.)
// ============================================================
class IdentityMatrix {
public:
    IdentityMatrix() : n(0), scale(1.0L) {}

    // I(size) — fix the dimension, return a new proxy
    IdentityMatrix operator()(unsigned int size) const {
        if (size == 0)
            throw std::invalid_argument("IdentityMatrix: size must be positive, got 0");
        return IdentityMatrix(size, scale);
    }

    // Materialize to a concrete Matrix<T> (triggered by: Matrix<T> A = I(n);)
    template<typename T>
    operator Matrix<T>() const {
        if (n == 0)
            throw std::invalid_argument(
                "IdentityMatrix: cannot materialize a dynamic identity matrix "
                "without a fixed size — use I(n) to specify one");
        Matrix<T> ans(n, n);
        for (int i = 0; i < n; i++)
            ans(i, i) = static_cast<T>(scale);
        return ans;
    }

    // kI + kI  →  (k1+k2)I      handles I+I+I+... chains
    IdentityMatrix operator+(const IdentityMatrix& rhs) const {
        return IdentityMatrix(resolveSize(n, rhs.n, "operator+"), scale + rhs.scale);
    }

    // kI * kI  →  (k1*k2)I
    IdentityMatrix operator*(const IdentityMatrix& rhs) const {
        return IdentityMatrix(resolveSize(n, rhs.n, "operator*"), scale * rhs.scale);
    }

    // I * scalar  (scalar on right: I * k)
    template<typename scalar>
    IdentityMatrix operator*(const scalar k) const {
        return IdentityMatrix(n, scale * static_cast<long double>(k));
    }

    unsigned int  size()     const { return n; }
    long double   getScale() const { return scale; }

private:
    unsigned int n;
    long double scale;

    IdentityMatrix(unsigned int size, long double s) : n(size), scale(s) {}

    // size resolution rules:
    //   dynamic + dynamic → dynamic (0)
    //   dynamic + fixed   → fixed
    //   fixed   + fixed   → must match, else throw
    static unsigned int resolveSize(unsigned int a, unsigned int b, const char* op) {
        if (a == 0) return b;
        if (b == 0) return a;
        if (a != b)
            throw std::invalid_argument(
                std::string("IdentityMatrix size mismatch in ") + op + ": " +
                std::to_string(a) + " != " + std::to_string(b));
        return a;
    }
};

// scalar * I  (scalar on left)
template<typename scalar>
IdentityMatrix operator*(const scalar k, const IdentityMatrix& Id) {
    return Id * k;
}

// A + I  —  adds scale to each diagonal element; A must be square
template<typename datatype>
Matrix<datatype> operator+(Matrix<datatype> A, const IdentityMatrix& Id) {
    try {
        if (A.rows() != A.cols())
            throw std::invalid_argument(
                "operator+(Matrix, IdentityMatrix): Matrix must be square, got " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
        unsigned int sz = Id.size();
        if (sz != 0 && sz != A.rows())
            throw std::invalid_argument(
                "operator+(Matrix, IdentityMatrix): size mismatch: Matrix is " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
                " but I(" + std::to_string(sz) + ") was requested");
        for (unsigned int i = 0; i < A.rows(); i++)
            A(i, i) += static_cast<datatype>(Id.getScale());
        return A;
    } catch (const std::exception& e) {
        std::cerr << "Matrix + IdentityMatrix error: " << e.what() << std::endl;
        throw;
    }
}

// I + A  —  commutative
template<typename datatype>
Matrix<datatype> operator+(const IdentityMatrix& Id, Matrix<datatype> A) {
    return A + Id;
}

// A * I  —  A must be square; returns scale * A
template<typename datatype>
Matrix<datatype> operator*(Matrix<datatype> A, const IdentityMatrix& Id) {
    try {
        if (A.rows() != A.cols())
            throw std::invalid_argument(
                "operator*(Matrix, IdentityMatrix): Matrix must be square, got " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
        unsigned int sz = Id.size();
        if (sz != 0 && sz != A.cols())
            throw std::invalid_argument(
                "operator*(Matrix, IdentityMatrix): size mismatch: Matrix cols=" +
                std::to_string(A.cols()) + " but I(" + std::to_string(sz) + ")");
        return A * static_cast<datatype>(Id.getScale());
    } catch (const std::exception& e) {
        std::cerr << "Matrix * IdentityMatrix error: " << e.what() << std::endl;
        throw;
    }
}

// I * A  —  A must be square; returns scale * A
template<typename datatype>
Matrix<datatype> operator*(const IdentityMatrix& Id, Matrix<datatype> A) {
    try {
        if (A.rows() != A.cols())
            throw std::invalid_argument(
                "operator*(IdentityMatrix, Matrix): Matrix must be square, got " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
        unsigned int sz = Id.size();
        if (sz != 0 && sz != A.rows())
            throw std::invalid_argument(
                "operator*(IdentityMatrix, Matrix): size mismatch: I(" +
                std::to_string(sz) + ") but Matrix rows=" + std::to_string(A.rows()));
        return A * static_cast<datatype>(Id.getScale());
    } catch (const std::exception& e) {
        std::cerr << "IdentityMatrix * Matrix error: " << e.what() << std::endl;
        throw;
    }
}

// Global instance — include this header and 'I' is ready to use, inline version is for std=C++17 and beyond
inline const IdentityMatrix I;
//const IdentityMatrix I;