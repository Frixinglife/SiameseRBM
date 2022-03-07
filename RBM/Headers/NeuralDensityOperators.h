#ifndef _NEURAL_DENSITY_OPERATORS_H_
#define _NEURAL_DENSITY_OPERATORS_H_

#include "SiameseRBM.h"
#include <complex>
#include <vector>

class NeuralDensityOperators {
public:
    SiameseRBM FirstSiameseRBM, SecondSiameseRBM;
    std::vector<std::vector<std::complex<double>>> RoMatrix;
    std::vector<double> EigRoMatrix;
    double work_time;

    NeuralDensityOperators(unsigned int N_v, unsigned int N_h, unsigned int N_a);
    
    void PrintRBMs() const;
    vector<double> VectorsAdd(const vector<double>& FirstVec, const vector<double>& SecondVec);
    vector<double> VectorsSub(const vector<double>& FirstVec, const vector<double>& SecondVec);
    double VectorsScalarMult(const vector<double>& FirstVec, const vector<double>& SecondVec);
    double GetGamma(const vector<double>& FirstSigma, const vector<double>& SecondSigma, char PlusOrMinus);
    std::complex<double> GetPi(const vector<double>& FirstSigma, const vector<double>& SecondSigma);
    std::complex<double> GetRo(const vector<double>& FirstSigma, const vector<double>& SecondSigma);
    std::vector<std::vector<std::complex<double>>> GetRoMatrix();
    void PrintRoMatrix() const;
    std::vector<double> FindEigRoMatrix();
    void PrintEigRoMatrix() const;
};

#endif //_NEURAL_DENSITY_OPERATORS_H_
