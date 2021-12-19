#ifndef _SIAMESE_RBM_H_
#define _SIAMESE_RBM_H_

#include <vector>
#include <string>
using std::vector;

class SiameseRBM {
public:
    unsigned int N_v, N_h, N_a;
    vector<vector<double>> W, V;
    vector<double> b, c, d;

    SiameseRBM() : N_v(0), N_h(0), N_a(0) {};

    SiameseRBM(unsigned int _N_v, unsigned int _N_h, unsigned int _N_a,
        const vector<vector<double>>& _W, const vector<vector<double>>& _V,
        const vector<double>& _b, const vector<double>& _c, const vector<double>& _d) :
        N_v(_N_v), N_h(_N_h), N_a(_N_a), W(_W), V(_V), b(_b), c(_c), d(_d) {};

    void PrintSiameseRBM(const std::string& NameRBM = "") const;
};

#endif //_SIAMESE_RBM_H_
