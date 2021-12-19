#include "RandomMatricesForRBM.h"
#include <random>

vector<vector<double>> RandomMatricesForRBM::GetRandomMatrix(unsigned int N, unsigned int M,
    unsigned int seed, double left, double right) {

    std::random_device random_device;
    std::mt19937 generator(random_device());
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(left, right);

    vector<vector<double>> answer(N, vector<double>(M));
    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < M; ++j) {
            answer[i][j] = distribution(generator);
        }
    }

    return answer;
}

vector<double> RandomMatricesForRBM::GetRandomVector(unsigned int N, unsigned int seed,
    double left, double right) {

    std::random_device random_device;
    std::mt19937 generator(random_device());
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(left, right);

    vector<double> answer(N);
    for (unsigned int i = 0; i < N; ++i) {
        answer[i] = distribution(generator);
    }

    return answer;
}
