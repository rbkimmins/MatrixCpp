#include <iostream>
#include </home/ryan/Documents/C++/Matrix1.0.hpp>
using namespace std;
int main(){
    Matrix<double> A(2, 2);
    A.print();
    Matrix<double> B;
    B = {{1,2}, {2,1}};
    Matrix<double> C(3,3);
    (B+C).print();

    return 0;
}
