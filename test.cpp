#include "Matrix1.0.hpp"
using namespace std;
int main(){
    Matrix<int> A(3, 3);
    A.set_Ran_values(0, 1, -7);
    cout << "A =" << endl;
    A.print();
    Matrix<double> B;
    B = {{1,2}, {2,1}};
    Matrix<int> C(3,3);
    C.set_Ran_values(0,1, -4);
    cout << "C = " << endl;
    C.print();
    cout << "Trace of C: " << C.trace() << endl << "Determinate of C: " << C.det() << endl;
    auto lup = C.LU();
    Matrix <double> L = lup.getL(); Matrix <double> U = lup.getU();
    cout << "L = "<< endl;
    L.print(); 
    cout << "U = "<< endl;
    U.print();
    cout << A*C;
    return 0;
}
