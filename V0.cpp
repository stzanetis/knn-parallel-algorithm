#include <iostream>
#include <vector>
#include <cmath>
#include <cblas.h>

using namespace std;

void calculate_distances(const vector<vector<double>>& C, const vector<vector<double>>& Q, vector<vector<double>>& D) {
    int c_points = C.size();    // Number of points in C
    int q_points = Q.size();    // Number of points in Q
    int d = C[0].size();        // Number of dimensions in a point

    vector<vector<double>> C_squared(c_points, vector<double>(d, 0.0));     // Vector for C^2
    vector<vector<double>> Q_squared(q_points, vector<double>(d));          // Vector for Q^2
    vector<double> C_mul_Q(c_points * q_points);                            // Vector for C * Q^T

    // Calculate C^2
    for (int i = 0; i < c_points; i++) {
        for (int j = 0; j < d; ++j) {
            C_squared[i][j] = C[i][j] * C[i][j];
        }
    }

    // Calculate Q^2T
    for (int i = 0; i < q_points; i++) {
        for (int j = 0; j < d; ++j) {
            Q_squared[i][j] = Q[i][j] * Q[i][j];
        }
    }

    // Calculate C*Q^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c_points, q_points, d, 1.0, &C[0][0], d, &Q[0][0], d, 0.0, &C_mul_Q[0], q_points);

    // Calculate D using sqrt(C^2 - 2C*Q^T + Q^2T)
    D.resize(c_points, vector<double>(q_points));
    for (int i =0; i < c_points; i++) {
        for (int j = 0; j < q_points; j++) {
            D[i][j] = sqrt(C_squared[i][j] - 2*C_mul_Q[i*q_points + j] + Q_squared[i][j]);
        }
    }
}

vector<vector<int>> knn_search(const vector<vector<double>>& C, const vector<vector<double>>& Q, char o, int k) {
    if (o != 'k') k = 1;
    vector<vector<double>> D;
    vector<vector<int>> nearest_neighbors(Q.size());
    calculate_distances(C, Q, D);

    for (int i = 0; i < Q.size(); i++) {
        vector<pair<int,double>> point_pairs;
        for (int j = 0; j < C.size(); j++) {
            point_pairs.emplace_back(j, D[j][i]);
        }

        quick_select(point_pairs, k);
        
        nearest_neighbors[i].resize(k);
        for (int j = 0; j < k; j++) {
            nearest_neighbors[i][j] = point_pairs[j].first;
        }
    }

    return nearest_neighbors;
}

void quick_select(vector<pair<int,double>>& point_pairs, int k) {
    // Returns since the points are less or equal to k, but not in order
    if (point_pairs.size() <= k) return;

    int left = 0, right = point_pairs.size() - 1;
    while (left <= right) {
        int pivotIndex = left + (right - left) / 2;
        pivotIndex = partition(point_pairs, left, right, pivotIndex);
        if (pivotIndex == k) break;
        else if (pivotIndex < k) left = pivotIndex + 1;
        else right = pivotIndex - 1;
    }
}

int partition(vector<pair<int,double>>& point_pairs, int left, int right, int pivotIndex) {
    double pivotValue = point_pairs[pivotIndex].second;
    int storeIndex = left;
    swap(point_pairs[pivotIndex], point_pairs[right]);
    for (int i = left; i < right; i++) {
        if (point_pairs[i].second < pivotValue) {
            swap(point_pairs[storeIndex], point_pairs[i]);
            storeIndex++;
        }
    }
    swap(point_pairs[right], point_pairs[storeIndex]);
    return storeIndex;
}

// For testing
int main() {
    vector<vector<double>> C = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    vector<vector<double>> Q = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    vector<vector<double>> D;

    calculate_distances(C, Q, D);

    //Print D
    for (int i = 0; i < D.size(); i++) {
        for (int j = 0; j < D[0].size(); j++) {
            cout << D[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}