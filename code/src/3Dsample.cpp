// C++ Version of coding a wavefunction solver
#include <iostream>
#include <vector>

#define DIM 3

using namespace std;

// Returns an array that contains the axes
vector<vector<double>> getAxes(double dx, double Lx, double Ly, double Lz){
    vector<vector<double>> axes;

    int Nx = Lx/dx;
    int Ny = Ly/dx;
    int Nz = Lz/dx;

    int N[3] = {Nx, Ny, Nz};

    for (int n=0;n<DIM;n++){
        vector<double> Vn;
        for (int i = 0; i < N[n]; i++){
            Vn.push_back(i * dx);
        }
        axes.push_back(Vn);
    }

    return axes;
}

vector<double> getPsi(vector<double>* axes,float A, float s){

}
