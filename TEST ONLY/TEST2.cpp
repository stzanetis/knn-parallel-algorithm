#include <iostream>
#include <vector>
#include <cblas.h>
#include <random>
#include <chrono>

using namespace std;

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

void calculate_distances(const vector<vector<double>>& C, const vector<vector<double>>& Q, vector<vector<double>>& D) {
    int c_points = C.size();    // Number of points in C
    int q_points = Q.size();    // Number of points in Q
    int d = C[0].size();        // Number of dimensions in a point

    vector<double> CSquared(c_points);          // Vector for C^2
    vector<double> QSquared(q_points);          // Vector for Q^2
    vector<double> CQT(c_points * q_points);    // Vector for C * Q^T

    // Create a flat vector for C and Q (Used for cblas_dgemm)
    vector<double> CFlat(c_points * d), QFlat(q_points * d);
    for (int i = 0; i < c_points; ++i) {
        for (int j = 0; j < d; ++j) {
            CFlat[i * d + j] = C[i][j];
        }
    }
    for (int i = 0; i < q_points; ++i) {
        for (int j = 0; j < d; ++j) {
            QFlat[i * d + j] = Q[i][j];
        }
    }

    // Calculate C^2
    for (int i = 0; i < c_points; i++) {
        double sum = 0.0;
        for (int j = 0; j < d; ++j) {
            sum += C[i][j] * C[i][j];
        }
        CSquared[i] = sum;
    }

    // Calculate Q^2
    for (int i = 0; i < q_points; i++) {
        double sum = 0.0;
        for (int j = 0; j < d; ++j) {
            sum += Q[i][j] * Q[i][j];
        }
        QSquared[i] = sum;
    }

    // Calculate C*Q^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c_points, q_points, d, 1.0, CFlat.data(), d, QFlat.data(), d, 0.0, CQT.data(), q_points);

    // Calculate D using sqrt(C^2 - 2C*Q^T + Q^2T)
    D.resize(c_points, vector<double>(q_points));
    for (int i =0; i < c_points; i++) {
        for (int j = 0; j < q_points; j++) {
            D[i][j] = sqrt(CSquared[i] + QSquared[j] - 2 * CQT[i * q_points + j]);
        }
    }
}

pair<vector<vector<int>>, vector<vector<double>>> knn_search(const vector<vector<double>>& C, const vector<vector<double>>& Q, int k) {
    vector<vector<double>> D;
    vector<vector<int>> idx(Q.size());
    vector<vector<double>> dist(Q.size());
    calculate_distances(C, Q, D);

    for (int i = 0; i < Q.size(); i++) {
        vector<pair<int,double>> point_pairs;
        for (int j = 0; j < C.size(); j++) {
            point_pairs.emplace_back(j, D[j][i]);
        }
        
        quick_select(point_pairs, k);
        
        idx[i].resize(k);
        dist[i].resize(k);
        for (int j = 0; j < k; j++) {
            idx[i][j] = point_pairs[j].first;
            dist[i][j] = point_pairs[j].second;
        }
    }

    return {idx, dist};
}

// For testing
int main() {
    // Generate random points for C and Q
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);
    vector<vector<double>> C(100000, vector<double>(10));
    vector<vector<double>> Q(5000, vector<double>(10));
    for (auto& point : C) {
        for (auto& coord : point) {
            coord = dis(gen);
        }
    }
    for (auto& point : Q) {
        for (auto& coord : point) {
            coord = dis(gen);
        }
    }

    // Perform k-NN search while measuring time
    int k=2;
    auto start = chrono::high_resolution_clock::now();
    auto [idx, dist] = knn_search(C, Q, k);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "k-NN search took " << elapsed.count() << " seconds.\n";

    // Display results
    //for (int i = 0; i < idx.size(); i++) {
    //    std::cout << "Query " << i << ":\n";
    //    for (int j = 0; j < k; j++) {
    //        std::cout << "Neighbor " << j << ": index = " << idx[i][j] << ", distance = " << dist[i][j] << "\n";
    //    }
    //}

    return 0;
}