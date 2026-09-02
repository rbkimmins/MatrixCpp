// Demo and smoke test for the Plotting Package. Build and run from the repo
// root:
//
//     g++ -std=c++17 -O2 -fopenmp -I. plotting/demo.cpp -o plotdemo && ./plotdemo
//
// Writes plot_1_lines.png ... plot_8_family.png into the working directory.
#include "MatrixPlot.hpp"
#include <cstdio>
int main(){
  Matrix<double> x = linspace(0.0, 10.0, 120);
  // 1. multi-series line plot with labels
  plt::plot(x, x.sin(), "sin"); plt::plot(x, x.cos(), "cos");
  plt::title("lines"); plt::xlabel("x"); plt::legend(":bottomleft"); plt::grid();
  plt::save("plot_1_lines.png"); printf("  lines     ok\n");
  // 2. scatter + limits + size
  Matrix<double> r(80,1); r.set_Ran_values(-1,1,-7);
  plt::scatter(r, "noise"); plt::ylim(-1.5,1.5); plt::size(700,400);
  plt::save("plot_2_scatter.png"); printf("  scatter   ok\n");
  // 3. heatmap of a matrix
  Matrix<double> A(40,60);
  for(int i=0;i<40;i++) for(int j=0;j<60;j++) A(i,j)=std::sin(i*0.2)*std::cos(j*0.15);
  plt::heatmap(A); plt::title("heatmap"); plt::save("plot_3_heat.png"); printf("  heatmap   ok\n");
  // 4. sparsity of an upper-triangular R from QR
  Matrix<double> M(60,60); M.set_Ran_values(-1,1,-5);
  auto [Q,R,P] = M.QR();
  plt::spy(R); plt::title("R from QR is upper triangular"); plt::save("plot_4_spy.png");
  printf("  spy       ok\n");
  // 5. log axes
  Matrix<double> n(30,1), t(30,1);
  for(int i=0;i<30;i++){ n(i,0)=double(i+1); t(i,0)=std::pow(double(i+1),3.0); }
  plt::loglog(n,t,"n^3"); plt::xlabel("n"); plt::ylabel("time"); plt::legend();
  plt::save("plot_5_loglog.png"); printf("  loglog    ok\n");
  // 6. histogram + surface + a column-family plot
  Matrix<double> h(500,1); h.set_Ran_values(-3,3,-11);
  plt::hist(h,30,"uniform"); plt::save("plot_6_hist.png"); printf("  hist      ok\n");
  plt::surface(A); plt::save("plot_7_surf.png"); printf("  surface   ok\n");
  Matrix<double> fam(50,3);
  for(int i=0;i<50;i++){ fam(i,0)=i; fam(i,1)=i*i*0.1; fam(i,2)=std::sqrt(double(i))*5; }
  plt::plot(fam,"col"); plt::legend(); plt::save("plot_8_family.png"); printf("  family    ok\n");
  return 0;
}
