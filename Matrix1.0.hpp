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
        inline bool empty() const {
            return !(rowSize * colSize);
        }
        void print(){
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

        // datatype const det() const{

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