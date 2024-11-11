#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cblas.h>

using namespace std;

void printResults(const vector<vector<int>>& idx, const vector<vector<double>>& dist) {
    for (int i = 0; i < idx.size(); i++) {
        std::cout << "Query " << i << ":\n";
        for (int j = 0; j < idx[i].size(); j++) {
            std::cout << "Neighbor " << j << ": index = " << idx[i][j] << ", distance = " << dist[i][j] << "\n";
        }
    }
}

void calculateDistances(const vector<vector<double>>& C, const vector<vector<double>>& Q, vector<vector<double>>& D) {
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

void quickSelect(vector<pair<int,double>>& point_pairs, int k) {
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

void knnSearch(const vector<vector<double>>& C, const vector<vector<double>>& Q, int k, vector<vector<int>>& idx, vector<vector<double>>& dist) {
    vector<vector<double>> D;
    calculateDistances(C, Q, D);

    for (int i = 0; i < Q.size(); i++) {
        vector<pair<int,double>> point_pairs;
        for (int j = 0; j < C.size(); j++) {
            point_pairs.emplace_back(j, D[j][i]);
        }
        
        quickSelect(point_pairs, k);
        
        idx[i].resize(k);
        dist[i].resize(k);
        for (int j = 0; j < k; j++) {
            idx[i][j] = point_pairs[j].first;
            dist[i][j] = point_pairs[j].second;
        }
    }
}

void projectToLowerDimension(const vector<vector<double>>& input, vector<vector<double>>& output) {
    int points = input.size();
    int old_dim = input[0].size();

    // Calculate new_dim automatically to balance speed and accuracy
    int new_dim = static_cast<int>(sqrt(old_dim));

    // Random projection matrix
    vector<vector<double>> projection_matrix(old_dim, vector<double>(new_dim));
    for (int i = 0; i < old_dim; ++i) {
        for (int j = 0; j < new_dim; ++j) {
            projection_matrix[i][j] = ((double) rand() / (RAND_MAX));
        }
    }

    // Project input to lower dimension
    output.resize(points, vector<double>(new_dim));
    for (int i = 0; i < points; ++i) {
        for (int j = 0; j < new_dim; ++j) {
            output[i][j] = 0.0;
            for (int k = 0; k < old_dim; ++k) {
                output[i][j] += input[i][k] * projection_matrix[k][j];
            }
        }
    }
}

pair<vector<vector<int>>, vector<vector<double>>> knnSearchParallel(const vector<vector<double>>& C, const vector<vector<double>>& Q, int k) {
    vector<vector<double>> C_proj, Q_proj;
    vector<vector<int>> idx(Q.size());
    vector<vector<double>> dist(Q.size());

    // Project C and Q to lower dimensions
    auto start = omp_get_wtime();
    projectToLowerDimension(C, C_proj);
    projectToLowerDimension(Q, Q_proj);
    auto end = omp_get_wtime();
    cout << "Projected to lower dimensions in " << (end - start) << " seconds" << endl;

    int c_points = C_proj.size();
    int q_points = Q_proj.size();
    int d = C_proj[0].size();
    int num_threads = omp_get_max_threads();
    int chunk_size = (q_points + num_threads - 1) / num_threads;
    vector<vector<vector<int>>> idx_chunks(num_threads);
    vector<vector<vector<double>>> dist_chunks(num_threads);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = min(start + chunk_size, q_points);

        vector<vector<double>> subQ(Q_proj.begin() + start, Q_proj.begin() + end);
        vector<vector<int>> sub_idx(subQ.size(), vector<int>(k));
        vector<vector<double>> sub_dist(subQ.size(), vector<double>(k));

        knnSearch(C_proj, subQ, k, sub_idx, sub_dist);

        idx_chunks[thread_id] = sub_idx;
        dist_chunks[thread_id] = sub_dist;
    }

    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = min(start + chunk_size, q_points);
        for (int i = start; i < end; ++i) {
            idx[i] = idx_chunks[t][i - start];
            dist[i] = dist_chunks[t][i - start];
        }
    }

    return {idx, dist};
}

// For testing
int main() {
    int k = 2;
    int n = 70000; // Number of points for Corpus
    int q = 600;   // Number of Queries
    int d = 30;     // Dimensions

    // Generate random C and Q matrices
    std::vector<std::vector<double>> C(n, std::vector<double>(d));
    std::vector<std::vector<double>> Q(q, std::vector<double>(d));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            C[i][j] = rand() % 100;
        }
    }
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < d; j++) {
            Q[i][j] = rand() % 100;
        }
    }

    // Perform k-NN search while measuring time
    auto start = omp_get_wtime();
    auto [idx, dist] = knnSearchParallel(C, Q, k);
    auto end = omp_get_wtime();
    std::cout << "knnsearch took " << (end - start) << " seconds" << std::endl;

    //printResults(idx, dist);

    return 0;
}
