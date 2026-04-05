#include <initializer_list>
#include <iostream>

template <typename datatype>
class Matrix{
    public:
        Matrix(int i, int j){
            rowSize = i;
            colSize = j;

            grid = new datatype*[rowSize];       // array of row pointers
            for (int r = 0; r < rowSize; r++){
                grid[r] = new datatype[colSize]; // each row's actual data
            }
        }
        Matrix(const Matrix &M){
            rowSize = M.rowSize;
            colSize = M.colSize;
            grid = new datatype*[rowSize];
            for (int r = 0; r < rowSize; r++){
                grid[r] = new datatype[colSize];
                for (int c = 0; c < colSize; c++)
                    grid[r][c] = M.grid[r][c];
            }
        }
        Matrix& operator=(const Matrix &M){ // for matrix copying
            if (this == &M) return *this;  // self-assignment guard

            for (int r = 0; r < rowSize; r++)  // free existing data
                delete[] grid[r];
            delete[] grid;

            rowSize = M.rowSize;
            colSize = M.colSize;

            grid = new datatype*[rowSize];         // deep copy
            for (int r = 0; r < rowSize; r++){
                grid[r] = new datatype[colSize];
                for (int c = 0; c < colSize; c++)
                    grid[r][c] = M.grid[r][c];
            }

            return *this;
        }
        Matrix& operator=(const std::initializer_list<std::initializer_list<datatype>> &M){ // for direct matrix assignment
            try{
                if ( M.empty() ) throw (M.empty());  
            for (int r = 0; r < rowSize; r++)  // free existing data
                delete[] grid[r];
            delete[] grid;

            rowSize = M.size();
            auto itr = M.begin():
            colSize = itr->size();

            grid = new datatype*[rowSize];         // deep copy
            for (int r = 0; r < rowSize; r++){
                grid[r] = new datatype[colSize];
            }

            int i = 0, j = 0;
            for (auto row : M){
                for (auto element : row)
                    grid[i][j++ % colSize] = element;
                i++;
            }
            }catch( bool isEmpty){
                std::cout << "Error, Null Matrix assignment: " << isEmpty << std::endl;
                std::cout << "Exiting Now.";
                exit(0);
            }
            return *this;

        }
        Matrix& operator+(const Matrix &M){
            try{
                if( this->colSize != M.colSize || this->rowSize != M.rowSize)
                    throw;
                Matrix ans = this;
                for (auto row : ans.grid, auto rowM : M.grid){
                    for (auto element : row){
                        &element += 
                    }
                }
                
            }catch(){
                std::cout << "Dimension Error" << std::endl;
                std::cout << "Exiting Now.";
                exit(0);
            }
            return ans;
        }
        ~Matrix(){
            for (int r = 0; r < rowSize; r++){
                delete[] grid[r];
            }
            delete[] grid;
        }
    private:
        int rowSize;
        int colSize;
        datatype **grid;

};
