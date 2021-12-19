#include "NeuralDensityOperators.h"

int main() {
    NeuralDensityOperators DensityOperators;

    DensityOperators.PrintRBMs();

    DensityOperators.GetRoMatrix();
    DensityOperators.PrintRoMatrix();

    DensityOperators.FindEigRoMatrix();
    DensityOperators.PrintEigRoMatrix();

    return 0;
}
