#include "SiameseRBM.h"
#include <iostream>
#include <fstream>

using std::cin;
using std::cout;

void SiameseRBM::PrintSiameseRBM(const std::string& NameRBM) const {
    cout << "\n" << NameRBM << ":\n\n";
    cout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n\n";

    cout << "W:" << "\n";
    for (size_t i = 0; i < W.size(); ++i) {
        for (size_t j = 0; j < W[0].size(); ++j) {
            cout << W[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    cout << "V:" << "\n";
    for (size_t i = 0; i < V.size(); ++i) {
        for (size_t j = 0; j < V[0].size(); ++j) {
            cout << V[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    cout << "b:" << "\n";
    for (size_t i = 0; i < b.size(); ++i) {
        cout << b[i] << "\n";
    }
    cout << "\n";

    cout << "c:" << "\n";
    for (size_t i = 0; i < c.size(); ++i) {
        cout << c[i] << "\n";
    }
    cout << "\n";

    cout << "d:" << "\n";
    for (size_t i = 0; i < d.size(); ++i) {
        cout << d[i] << "\n";
    }
    cout << "\n";
}
