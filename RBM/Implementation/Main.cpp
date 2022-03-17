#include "NeuralDensityOperators.h"
#include <chrono>
#include <iostream>
#include <fstream>

void GetMatrix() {
    unsigned N_v, N_h, N_a;
    N_v = N_h = N_a = 5;

    NeuralDensityOperators DensityOperators(N_v, N_h, N_a);

    DensityOperators.PrintRBMs();

    DensityOperators.GetRoMatrix();
    DensityOperators.PrintRoMatrix();

    DensityOperators.FindEigRoMatrix();
    DensityOperators.PrintEigRoMatrix();
}

void GetTime() {
    std::ofstream fout("times.txt", std::ios_base::app);

    unsigned N_v, N_h, N_a;
    N_v = 400;
    N_h = 400;
    N_a = 400;

    NeuralDensityOperators DensityOperators(N_v, N_h, N_a);
    DensityOperators.GetRoMatrix();

    std::cout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n";
    std::cout << "Matrix size: " << DensityOperators.RoMatrix.size() << "\n";
    std::cout << "Time: " << DensityOperators.work_time << " s\n\n";

    fout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n";
    fout << "Matrix size: " << DensityOperators.RoMatrix.size() << "\n";
    fout << "Time: " << DensityOperators.work_time << " s\n\n";

    //DensityOperators.PrintRoMatrix();
    //DensityOperators.FindEigRoMatrix();
    //DensityOperators.PrintEigRoMatrix();
}

int main() {
    GetTime();
    return 0;
}
