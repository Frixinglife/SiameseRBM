#include "NeuralDensityOperators.h"
#include "RandomMatricesForRBM.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include "mkl.h"

#define MAX_N 100
MKL_Complex16 A[MAX_N * MAX_N];
MKL_Complex16 W[MAX_N];
MKL_Complex16 VL[MAX_N * MAX_N];
MKL_Complex16 VR[MAX_N * MAX_N];
MKL_Complex16 Work[2 * MAX_N];
double rwork[2 * MAX_N];

using std::cin;
using std::cout;

NeuralDensityOperators::NeuralDensityOperators() {
    unsigned int N_h, N_v, N_a;
    cout << "Number of neurons in the visible layer: ";
    cin >> N_v;
    cout << "Number of hidden layer neurons: ";
    cin >> N_h;
    cout << "Number of auxiliary layer neurons: ";
    cin >> N_a;

    vector<vector<double>> W_1, W_2, V_1, V_2;

    W_1 = RandomMatricesForRBM::GetRandomMatrix(N_h, N_v, 11);
    V_1 = RandomMatricesForRBM::GetRandomMatrix(N_a, N_v, 22);

    W_2 = RandomMatricesForRBM::GetRandomMatrix(N_h, N_v, 33);
    V_2 = RandomMatricesForRBM::GetRandomMatrix(N_a, N_v, 44);

    vector<double> c_1, c_2, b_1, b_2, d_1, d_2;

    c_1 = RandomMatricesForRBM::GetRandomVector(N_h, 55);
    b_1 = RandomMatricesForRBM::GetRandomVector(N_v, 66);
    d_1 = RandomMatricesForRBM::GetRandomVector(N_a, 77);

    c_2 = RandomMatricesForRBM::GetRandomVector(N_h, 88);
    b_2 = RandomMatricesForRBM::GetRandomVector(N_v, 99);
    d_2 = RandomMatricesForRBM::GetRandomVector(N_a, 110);

    SiameseRBM _FirstSiameseRBM(N_v, N_h, N_a, W_1, V_1, b_1, c_1, d_1);
    SiameseRBM _SecondSiameseRBM(N_v, N_h, N_a, W_2, V_2, b_2, c_2, d_2);

    FirstSiameseRBM = _FirstSiameseRBM;
    SecondSiameseRBM = _SecondSiameseRBM;
}

void NeuralDensityOperators::PrintRBMs() const {
    FirstSiameseRBM.PrintSiameseRBM("First siamese RBM");
    SecondSiameseRBM.PrintSiameseRBM("Second siamese RBM");
}

vector<double> NeuralDensityOperators::VectorsAdd(
    const vector<double>& FirstVec, const vector<double>& SecondVec) {
    
    size_t size = FirstVec.size();
    vector<double> Answer(size);
    for (size_t i = 0; i < size; ++i) {
        Answer[i] = FirstVec[i] + SecondVec[i];
    }
    
    return Answer;
}

vector<double> NeuralDensityOperators::VectorsSub(
    const vector<double>& FirstVec, const vector<double>& SecondVec) {

    size_t size = FirstVec.size();
    vector<double> Answer(size);
    for (size_t i = 0; i < size; ++i) {
        Answer[i] = FirstVec[i] - SecondVec[i];
    }

    return Answer;
}

double NeuralDensityOperators::VectorsScalarMult(
    const vector<double>& FirstVec, const vector<double>& SecondVec) {

    size_t size = FirstVec.size();
    double Answer = 0.0;
    for (size_t i = 0; i < size; ++i) {
        Answer += FirstVec[i] * SecondVec[i];
    }

    return Answer;
}

double NeuralDensityOperators::GetGamma(const vector<double>& FirstSigma, const vector<double>& SecondSigma, char PlusOrMinus) {
    double Answer = 0.0;

    switch (PlusOrMinus) {
    case '+':
        Answer = VectorsScalarMult(FirstSiameseRBM.b, VectorsAdd(FirstSigma, SecondSigma));

        for (size_t i = 0; i < FirstSiameseRBM.N_h; ++i) {
            double FirstScalarMult = VectorsScalarMult(FirstSiameseRBM.W[i], FirstSigma) + FirstSiameseRBM.c[i];
            Answer += std::log(1.0 + std::exp(FirstScalarMult));

            double SecondScalarMult = VectorsScalarMult(FirstSiameseRBM.W[i], SecondSigma) + FirstSiameseRBM.c[i];
            Answer += std::log(1.0 + std::exp(SecondScalarMult));
        }

        Answer *= 0.5;
        break;
    case '-':
        Answer = VectorsScalarMult(SecondSiameseRBM.b, VectorsSub(FirstSigma, SecondSigma));

        for (size_t i = 0; i < SecondSiameseRBM.N_h; ++i) {
            double FirstScalarMult = VectorsScalarMult(SecondSiameseRBM.W[i], FirstSigma) + SecondSiameseRBM.c[i];
            Answer += std::log(1.0 + std::exp(FirstScalarMult));

            double SecondScalarMult = VectorsScalarMult(SecondSiameseRBM.W[i], SecondSigma) + SecondSiameseRBM.c[i];
            Answer -= std::log(1.0 + std::exp(SecondScalarMult));
        }

        Answer *= 0.5;
        break;
    default:
        break;
    }

    return Answer;
}

std::complex<double> NeuralDensityOperators::GetPi(const vector<double>& FirstSigma, const vector<double>& SecondSigma) {
    std::complex<double> Answer(0.0, 0.0);

    for (size_t i = 0; i < FirstSiameseRBM.N_a; ++i) {
        double Re = VectorsScalarMult(FirstSiameseRBM.V[i], VectorsAdd(FirstSigma, SecondSigma)) * 0.5 + FirstSiameseRBM.d[i];
        double Im = VectorsScalarMult(SecondSiameseRBM.V[i], VectorsSub(FirstSigma, SecondSigma)) * 0.5;

        std::complex<double> CurrentAnswer(Re, Im);
        std::complex<double> One(1.0, 0.0);

        Answer += std::log(One + std::exp(CurrentAnswer));
    }
    
    return Answer;
}

std::complex<double> NeuralDensityOperators::GetRo(const vector<double>& FirstSigma, const vector<double>& SecondSigma) {
    std::complex<double> Gamma(GetGamma(FirstSigma, SecondSigma, '+'), GetGamma(FirstSigma, SecondSigma, '-'));
    std::complex<double> Pi = GetPi(FirstSigma, SecondSigma);

    return std::exp(Gamma + Pi);
}

std::vector<std::vector<std::complex<double>>> NeuralDensityOperators::GetRoMatrix() {
    unsigned int N_v = FirstSiameseRBM.N_v;
    std::vector<std::vector<std::complex<double>>> _RoMatrix(N_v, std::vector<std::complex<double>>(N_v, (0.0, 0.0)));
    std::complex<double> Sum(0.0, 0.0);

    for (unsigned int i = 0; i < N_v; ++i) {
        for (unsigned int j = 0; j < N_v; ++j) {
            std::vector<double> FirstSigma(N_v, 0.0);
            std::vector<double> SecondSigma(N_v, 0.0);

            FirstSigma[i] = 1.0;
            SecondSigma[j] = 1.0;

            std::complex<double> Element = GetRo(FirstSigma, SecondSigma);

            if (i == j) {
                Sum += Element;
            }

            _RoMatrix[i][j] = Element;
        }
    }

    for (unsigned int i = 0; i < N_v; ++i) {
        for (unsigned int j = 0; j < N_v; ++j) {
            _RoMatrix[i][j] /= Sum;
        }
    }
    
    RoMatrix = _RoMatrix;

    return RoMatrix;
}

void NeuralDensityOperators::PrintRoMatrix() const {
    size_t size = RoMatrix.size();

    cout << "Ro matrix:" << "\n\n";

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            std::cout << std::setw(30) << RoMatrix[i][j];
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

std::vector<double> NeuralDensityOperators::FindEigRoMatrix() {
    const int N = static_cast<const int>(RoMatrix.size());

    std::vector<double> Answer(N);

    const char jobvl = 'N';
    const char jobvr = 'N';


    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[j + i * N].real = RoMatrix[i][j].real();
            A[j + i * N].imag = RoMatrix[i][j].imag();
        }
    }

    const int lda = N;
    const int ldvl = N;
    const int ldvr = N;
    const int lwork = 2 * N;
    int info;

    zgeev(&jobvl, &jobvr, &N, A, &lda, W, VL, &ldvl, VR, &ldvr, Work, &lwork, rwork, &info);

    for (int i = 0; i < N; i++) {
        Answer[i] = W[i].real;
    }

    EigRoMatrix = Answer;

    return Answer;
}

void NeuralDensityOperators::PrintEigRoMatrix() const {
    cout << "Eigenvalues:\n\n";
    
    for (size_t i = 0; i < EigRoMatrix.size(); ++i) {
        std::cout << EigRoMatrix[i] << "\n";
    }
}
