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
                throw std::invalid_argument("Matrix: dimensions must be non-negative, got " + std::to_string(i) + "x" + std::to_string(j));
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

        // In-place Hadamard product
        Matrix& operator%= (const Matrix& M){
            *this = (*this) % M;
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

        // Returns the sum of the main diagonal elements. Requires a square matrix.
        datatype trace() const{
            try
            {
                if (!(rowSize == colSize && rowSize > 0))
                    throw std::invalid_argument(
                        "trace() requires a square non-empty matrix, got " +
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
        
        // --------------------------------------------------------
        // LUResult — returned by Matrix::LU()
        // Stores the packed Doolittle factorization and pivot vector.
        // Relation: PA = LU, where P is the permutation matrix.
        //   packed lower triangle (below diag) = L factors (diag of L is implicitly 1)
        //   packed upper triangle (incl. diag)  = U
        // --------------------------------------------------------
        class LUResult {
        public:
            // L and U are always double — LU is a floating-point algorithm.
            // Materialize L: lower triangle with diag = 1
            Matrix<double> getL() const {
                Matrix<double> L(n, n);
                for (int i = 0; i < n; i++) {
                    L(i, i) = 1.0;
                    for (int j = 0; j < i; j++)
                        L(i, j) = packedAt(i, j);
                }
                return L;
            }

            // Materialize U: upper triangle including diagonal
            Matrix<double> getU() const {
                Matrix<double> U(n, n);
                for (int i = 0; i < n; i++)
                    for (int j = i; j < n; j++)
                        U(i, j) = packedAt(i, j);
                return U;
            }

            // Materialize P: permutation matrix such that PA = LU.
            // 0s and 1s only, so datatype is fine here.
            Matrix<datatype> getPivot() const {
                Matrix<datatype> P(n, n);
                for (int i = 0; i < n; i++) P(i, i) = datatype(1);
                for (int k = 0; k < n; k++) {
                    if (pivotVec[k] != k) {
                        for (int j = 0; j < n; j++) {
                            datatype tmp = P(k, j);
                            P(k, j) = P(pivotVec[k], j);
                            P(pivotVec[k], j) = tmp;
                        }
                    }
                }
                return P;
            }

        private:
            // std::vector<double> avoids the incomplete-type error that arises from
            // storing Matrix<double> inside a nested class of Matrix<datatype>.
            std::vector<double> packedData;
            std::vector<int>    pivotVec;
            int                 n;

            double packedAt(int i, int j) const { return packedData[i * n + j]; }
            double& packedAt(int i, int j)       { return packedData[i * n + j]; }

            LUResult(int size, const std::vector<double>& data, const std::vector<int>& pv)
                : packedData(data), pivotVec(pv), n(size) {}

            // pivotSign() is private — only Matrix::det() needs it
            int pivotSign() const {
                int sign = 1;
                for (int k = 0; k < n; k++)
                    if (pivotVec[k] != k) sign = -sign;
                return sign;
            }

            friend class Matrix;
        };

        // Performs LU factorization with partial pivoting (Doolittle's method).
        // Requires a square matrix of at least 2x2.
        // Returns an LUResult from which L, U, and P can be extracted.
        // Throws if the matrix is singular (zero pivot encountered).
        LUResult LU() const {
            try {
                if (rowSize != colSize)
                    throw std::invalid_argument(
                        "LU: matrix must be square, got " +
                        std::to_string(rowSize) + "x" + std::to_string(colSize));
                if (rowSize < 2)
                    throw std::invalid_argument(
                        "LU: matrix too small (" +
                        std::to_string(rowSize) + "x" + std::to_string(colSize) +
                        "), must be at least 2x2");

                int n = (int)rowSize;
                // Always work in double — integer types would truncate the L factors.
                // Store as flat vector to avoid incomplete-type issues with Matrix<double>.
                std::vector<double> packedData(n * n);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        packedData[i * n + j] = double((*this)(i, j));

                auto pat = [&](int i, int j) -> double& { return packedData[i * n + j]; };

                std::vector<int> pivotVec(n);
                for (int i = 0; i < n; i++) pivotVec[i] = i;

                for (int k = 0; k < n; k++) {
                    // Partial pivoting: find row >= k with largest |value| in column k
                    int maxRow = k;
                    double maxVal = std::abs(pat(k, k));
                    for (int i = k + 1; i < n; i++) {
                        double val = std::abs(pat(i, k));
                        if (val > maxVal) { maxVal = val; maxRow = i; }
                    }

                    if (maxVal == 0.0)
                        throw std::runtime_error(
                            "LU: zero pivot in column " + std::to_string(k) +
                            " — matrix is singular");

                    // Swap rows k and maxRow
                    if (maxRow != k)
                        for (int j = 0; j < n; j++)
                            std::swap(pat(k, j), pat(maxRow, j));
                    pivotVec[k] = maxRow;

                    // Doolittle: L column below pivot, then update trailing submatrix
                    for (int i = k + 1; i < n; i++)
                        pat(i, k) /= pat(k, k);
                    for (int i = k + 1; i < n; i++)
                        for (int j = k + 1; j < n; j++)
                            pat(i, j) -= pat(i, k) * pat(k, j);
                }

                return LUResult(n, packedData, pivotVec);
            } catch (const std::exception& e) {
                std::cerr << "LU factorization error: " << e.what() << '\n';
                throw;
            }
        }

        // Returns the determinant of this matrix via LU factorization.
        // Requires a square matrix. Integer types are rounded to avoid floating-point drift.
        datatype det() const {
            try {
                if (rowSize != colSize)
                    throw std::invalid_argument(
                        "det() requires a square matrix, got " +
                        std::to_string(rowSize) + "x" + std::to_string(colSize));
                LUResult lu = this->LU();
                double d = double(lu.pivotSign());
                for (long i = 0; i < rowSize; i++)
                    d *= lu.packedAt(i, i);
                // For integer matrices the determinant is always a whole number;
                // round before casting to avoid floating-point drift (e.g. 2.9999 → 3)
                return std::is_integral<datatype>::value
                    ? datatype(std::round(d))
                    : datatype(d);
            } catch (const std::exception& e) {
                std::cerr << "det() error: " << e.what() << '\n';
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

        // Naive O(n³) matrix multiplication (i-k-j loop order for cache friendliness).
        // Used as the base case for Strassen-Winograd and for rectangular matrices.
        static Matrix naiveMul(const Matrix& A, const Matrix& B) {
            Matrix ans(A.rowSize, B.colSize);
            for (long i = 0; i < A.rowSize; i++)
                for (long k = 0; k < A.colSize; k++) {
                    const datatype aik = A.grid[i * A.colSize + k];
                    for (long j = 0; j < B.colSize; j++)
                        ans.grid[i * B.colSize + j] += aik * B.grid[k * B.colSize + j];
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