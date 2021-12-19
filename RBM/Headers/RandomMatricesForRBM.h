#ifndef _RANDOM_MATRICES_FOR_RBM_H_
#define _RANDOM_MATRICES_FOR_RBM_H_

#include <vector>
using std::vector;

class RandomMatricesForRBM {
public:
    static vector<vector<double>> GetRandomMatrix(unsigned int N, unsigned int M,
        unsigned int seed, double left = 0.0, double right = 1.0);

    static vector<double> GetRandomVector(unsigned int N, unsigned int seed,
        double left = 0.0, double right = 1.0);
};

#endif //_RANDOM_MATRICES_FOR_RBM_H_
