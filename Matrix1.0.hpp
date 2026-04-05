// #include <initializer_list>
// #include <utility>
#include <numeric>
#include <stdexcept>
#include <string>

template <typename datatype>
class Matrix{
    public:
        //constructors
        Matrix(){
            rowSize = 0;
            colSize = 0;
            grid = nullptr;
        }
        Matrix(int i, int j){
            rowSize = i;
            colSize = j;
            grid = new datatype[rowSize*colSize]();
        }
        Matrix(const Matrix &M){
            rowSize = M.rowSize;
            colSize = M.colSize;
            grid = new datatype[rowSize*colSize]();
            for (int index = 0; index < rowSize*colSize; index++)
                grid[index] = M.grid[index];
        }
        // Overloaded operators
        Matrix& operator=(const Matrix &M){
            if (this == &M) return *this;  // self-assignment guard

            delete[] grid; // empty the Matrix
            rowSize = M.rowSize;
            colSize = M.colSize;
            grid = new datatype[rowSize*colSize];
            for (int index = 0; index < rowSize*colSize; index++)
                grid[index] = M.grid[index];
            return *this;
        }
        Matrix& operator=(const std::initializer_list<std::initializer_list<datatype>> &M){ // for direct matrix assignment
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

        Matrix operator+(const Matrix &M){
            Matrix ans = *this;
            try{
                if (this->colSize != M.colSize)
                    throw std::invalid_argument(
                        "Column size mismatch in operator+: " + std::to_string(colSize) + " != " + std::to_string(M.colSize));
                if (this->rowSize != M.rowSize)
                    throw std::invalid_argument(
                        "Row size mismatch in operator+: " + std::to_string(rowSize) + " != " + std::to_string(M.rowSize));

                for (int index = 0; index < colSize*rowSize; index++)
                    ans.grid[index] += M.grid[index];

            }catch(const std::exception& e){
                std::cerr << "Matrix addition error: " << e.what() << std::endl;
                throw;
            }
            return ans;
        }

        template <typename scalar>
        Matrix operator*(const scalar num) const {
            Matrix ans = *this;
            for (int index = 0; index < rowSize*colSize; index++)
                ans.grid[index] *= num;
            return ans;
        }

        Matrix operator*(const Matrix &M){
            try{
                if (this->colSize != M.rowSize)
                    throw std::invalid_argument(
                        "Inner dimensions must match for operator*: (" +
                        std::to_string(rowSize) + "x" + std::to_string(colSize) + ") * (" +
                        std::to_string(M.rowSize) + "x" + std::to_string(M.colSize) + ")");
                Matrix <datatype> ans(this->rowSize, M.colSize);
                Matrix <datatype> tmp(1, this->colSize);
                for (int i = 0; i < this->rowSize * M.colSize; i++){
                    tmp = (*this)(i / M.colSize, ':') % M(':', i % colSize.T());
                    ans[i] = std::accumulate(tmp.grid, tmp.grid + colSize, (datatype)0);
                }
                return ans;
            }catch(const std::exception& e){
                std::cerr << "Matrix multiplication error: " << e.what() << std::endl;
                throw;
            }
        }
        //Hadamard product (using %)
        Matrix operator%(const Matrix &M){
            try{
                if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                    throw std::invalid_argument(
                        "Dimension mismatch in Hadamard product: (" +
                        std::to_string(rowSize) + "x" + std::to_string(colSize) + ") vs (" +
                        std::to_string(M.rowSize) + "x" + std::to_string(M.colSize) + ")");
                Matrix <datatype> ans(M.rowSize, M.colSize);
                for (int i = 0; i < M.rowSize * M.colSize; i++){
                    ans.grid[i] = this->grid[i] * M.grid[i];
                }
                return ans;
            }catch(const std::exception& e){
                std::cerr << "Hadamard product error: " << e.what() << std::endl;
                throw;
            }
        }

        Matrix operator-(const Matrix &M){
            return *this + M*-1;
        }

        //Indexing (note, negative indexes will read the matrix backwards)
        //Assume the matrix A is indexed as A_ij
        datatype& operator()(const int& i, const int& j){
            return grid[(i % rowSize) * colSize + (j % colSize)];
        }
        datatype& operator[](const int& i){
            return grid[i % (rowSize * colSize)];
        }
        //vector extraction for the ith row
        Matrix operator()(const int& i, const char& c){
            try{
                switch(c){
                    case ':': {
                        Matrix <datatype> ans(1, this->colSize);
                        for (int j = 0; j < this->colSize; j++)
                            ans[j] = this->grid[(i % rowSize) * colSize + (j % colSize)];
                        return ans;
                    }
                }
                throw std::invalid_argument(
                    std::string("Invalid row specifier '") + c + "': use ':' for row extraction, e.g. A(i, ':')");
            }catch(const std::exception& e){
                std::cerr << "Matrix indexing error: " << e.what() << std::endl;
                throw;
            }
        }
        //vector extraction for the ith column
        Matrix operator()(const char& c, const int& i){
            try{
                switch(c){
                    case ':': {
                        Matrix <datatype> ans(rowSize, 1);
                        for (int j = 0; j < rowSize; j++)
                            ans[j] = grid[(j % rowSize) * colSize + (i % colSize)];
                        return ans;
                    }
                }
                throw std::invalid_argument(
                    std::string("Invalid column specifier '") + c + "': use ':' for column extraction, e.g. A(':', j)");
            }catch(const std::exception& e){
                std::cerr << "Matrix indexing error: " << e.what() << std::endl;
                throw;
            }
        }
        //dot operators
        inline bool empty() const { return !(rowSize * colSize); }
        unsigned int rows() const { return rowSize; }
        unsigned int cols() const { return colSize; }
        void print() const{
            for (int i = 0; i< rowSize*colSize; i++){
                if ( !(i % rowSize)) std::cout << "[ ";
                std::cout << grid[i]<< " ";
                if ( !((i + 1 )% colSize)) std::cout << "]" << std::endl;
            }
        }
        //Transpose
        Matrix T() const{
            Matrix <datatype> ans(colSize, rowSize);
            for (int i = 0; i < colSize*rowSize; i++){
                //By definition of transpose.
                ans.grid[(i % colSize) * rowSize + (i / colSize)] = grid[i];
            }
            return ans;
        }

        datatype trace() const{
            try
            {
                if (!(rowSize == colSize && rowSize > 0))
                    throw std::invalid_argument(
                        "trace() requires a square non-empty matrix, got " +
                        std::to_string(rowSize) + "x" + std::to_string(colSize));
                datatype sum = datatype(0);
                for (int i = 0; i < rowSize; i++) sum += (*this)(i,i);

                return sum;
            }
            catch(const std::exception& e)
            {
                std::cerr << "Matrix trace error: " << e.what() << '\n';
                throw;
            }
        }
        //sum member function
        datatype sum() const{// adds all elements
            datatype total = datatype(0);
            for (int i = 0; i< rowSize * colSize; i++) total += grid[i];
            return total;
        }
        // for 1, you get a column matrix containing the sum of each row.
        // For 0, return a row matrix of the sum of each column.
        Matrix sum(const bool& addcol) const{
            try{
                if (!addcol) {
                    Matrix<datatype> total_vec(1, colSize);
                    for (int i = 0; i < (int)colSize; i++)
                        total_vec[i] = (*this)(':', i).sum();
                    return total_vec;
                } else {
                    Matrix<datatype> total_vec(rowSize, 1);
                    for (int i = 0; i < (int)rowSize; i++)
                        total_vec[i] = (*this)(i, ':').sum();
                    return total_vec;
                }
            }catch(const std::exception& e){
                std::cerr << "Matrix summation error: " << e.what() << std::endl;
                throw;
            }
        }

        Matrix concat(const Matrix& M, const bool& concatCol) const{
            try{
                if (!concatCol) {
                    // vertical concat: stack rows, columns must match
                    if (this->colSize != M.colSize) throw std::invalid_argument(
                        "concat: column size mismatch: " + std::to_string(colSize) +
                        " != " + std::to_string(M.colSize));
                    Matrix<datatype> argumentMatrix(this->rowSize + M.rowSize, colSize);
                    for (int i = 0; i < (int)this->rowSize; i++)
                        argumentMatrix(i, ':') = (*this)(i, ':');
                    for (int i = 0; i < (int)M.rowSize; i++)
                        argumentMatrix(i + this->rowSize, ':') = M(i, ':');
                    return argumentMatrix;
                } else {
                    // horizontal concat: stack columns, rows must match
                    if (this->rowSize != M.rowSize) throw std::invalid_argument(
                        "concat: row size mismatch: " + std::to_string(rowSize) +
                        " != " + std::to_string(M.rowSize));
                    Matrix<datatype> argumentMatrix(rowSize, this->colSize + M.colSize);
                    for (int i = 0; i < (int)this->colSize; i++)
                        argumentMatrix(':', i) = (*this)(':', i);
                    for (int i = 0; i < (int)M.colSize; i++)
                        argumentMatrix(':', i + this->colSize) = M(':', i);
                    return argumentMatrix;
                }
            }catch(const std::exception& e){
                std::cerr << "Matrix concat error: " << e.what() << std::endl;
                throw;
            }
        }
        
        // LU factorization (Doolittle's method: diag(L) = 1).
        // Returns a packed matrix: upper triangle (incl. diagonal) = U,
        // lower triangle (below diagonal) = L.
        // To recover L: take lower triangle and set diagonal to 1.
        // To recover U: take upper triangle including diagonal.
        Matrix LU() const{
            try
            {
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
                Matrix<datatype> packed = *this;  // in-place Doolittle on a copy

                for (int k = 0; k < n; k++) {
                    if (packed(k, k) == datatype(0))
                        throw std::runtime_error(
                            "LU: zero pivot at position (" + std::to_string(k) +
                            "," + std::to_string(k) + ") — matrix may be singular");
                    // Compute L column k (rows below pivot)
                    for (int i = k + 1; i < n; i++)
                        packed(i, k) /= packed(k, k);
                    // Update trailing submatrix
                    for (int i = k + 1; i < n; i++)
                        for (int j = k + 1; j < n; j++)
                            packed(i, j) -= packed(i, k) * packed(k, j);
                }
                return packed;
            }
            catch(const std::exception& e)
            {
                std::cerr << "LU factorization error: " << e.what() << '\n';
                throw;
            }
        }

        // datatype det() const{

        // }
        //deconstructor
        ~Matrix(){
            if (grid) delete[] grid;
        }
    private:
        unsigned int rowSize;
        unsigned int colSize;
        datatype *grid;
};

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

// Global instance — include this header and 'I' is ready to use
inline const IdentityMatrix I;