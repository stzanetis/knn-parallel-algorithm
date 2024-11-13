#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cblas.h>
#include <random>
#include <unordered_map>

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
            D[i][j] = CSquared[i] + QSquared[j] - 2 * CQT[i * q_points + j];
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

// Function to generate random projection vectors
vector<vector<double>> generateRandomProjections(int original_dim, int reduced_dim) {
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 1.0);

    vector<vector<double>> projections(reduced_dim, vector<double>(original_dim));
    for (int i = 0; i < reduced_dim; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            projections[i][j] = distribution(generator);
        }
    }
    return projections;
}

// Function to project points onto lower-dimensional space
vector<vector<double>> projectPoints(const vector<vector<double>>& points, const vector<vector<double>>& projections) {
    int num_points = points.size();
    int reduced_dim = projections.size();
    int original_dim = points[0].size();

    vector<vector<double>> projected_points(num_points, vector<double>(reduced_dim));
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < reduced_dim; ++j) {
            double projection = 0.0;
            for (int k = 0; k < original_dim; ++k) {
                projection += points[i][k] * projections[j][k];
            }
            projected_points[i][j] = projection;
        }
    }
    return projected_points;
}

// Function to hash points into buckets
unordered_map<int, vector<int>> hashPoints(const vector<vector<double>>& projected_points, double w) {
    unordered_map<int, vector<int>> hash_table;
    for (int i = 0; i < projected_points.size(); ++i) {
        int hash_value = 0;
        for (int j = 0; j < projected_points[i].size(); ++j) {
            hash_value += static_cast<int>(floor(projected_points[i][j] / w));
        }
        hash_table[hash_value].push_back(i);
    }
    return hash_table;
}

void knnSearchLSH(const vector<vector<double>>& C, const vector<vector<double>>& Q, int k, int reduced_dim, double w, vector<vector<int>>& idx, vector<vector<double>>& dist) {
    // Generate random projections
    vector<vector<double>> projections = generateRandomProjections(C[0].size(), reduced_dim);
    vector<vector<double>> projected_C = projectPoints(C, projections);
    vector<vector<double>> projected_Q = projectPoints(Q, projections);

    unordered_map<int, vector<int>> hash_table = hashPoints(projected_C, w);

    for (int i = 0; i < Q.size(); ++i) {
        int hash_value = 0;
        for (int j = 0; j < projected_Q[i].size(); ++j) {
            hash_value += static_cast<int>(floor(projected_Q[i][j] / w));
        }

        vector<pair<int, double>> point_pairs;
        if (hash_table.find(hash_value) != hash_table.end()) {
            vector<vector<double>> subC;
            for (int idx : hash_table[hash_value]) {
            subC.push_back(C[idx]);
            }
            vector<vector<double>> subQ = {Q[i]};
            vector<vector<double>> D;
            calculateDistances(subC, subQ, D);
            for (int j = 0; j < subC.size(); ++j) {
            point_pairs.emplace_back(hash_table[hash_value][j], D[j][0]);
            }
        }

        quickSelect(point_pairs, k);

        idx[i].resize(k);
        dist[i].resize(k);
        for (int j = 0; j < k; ++j) {
            idx[i][j] = point_pairs[j].first;
            dist[i][j] = sqrt(point_pairs[j].second);
        }
    }
    cout << projected_C[0].size() << endl;
}

pair<vector<vector<int>>, vector<vector<double>>> knnSearchParallel(const vector<vector<double>>& C, const vector<vector<double>>& Q, int k) {

    int c_points = C.size();
    int q_points = Q.size();
    int d = C[0].size();
    int dl = sqrt(d);
    int num_threads = omp_get_max_threads();
    int chunk_size = (q_points + num_threads - 1) / num_threads;
    vector<vector<vector<int>>> idx_chunks(num_threads);
    vector<vector<vector<double>>> dist_chunks(num_threads);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = min(start + chunk_size, q_points);

        vector<vector<double>> subQ(Q.begin() + start, Q.begin() + end);
        int sq_points = subQ.size();
        vector<vector<int>> sub_idx(sq_points, vector<int>(k));
        vector<vector<double>> sub_dist(sq_points, vector<double>(k));

        knnSearchLSH(C, subQ, k, dl , 10, sub_idx, sub_dist);

        idx_chunks[thread_id] = sub_idx;
        dist_chunks[thread_id] = sub_dist;
    }

    vector<vector<int>> idx(q_points, vector<int>(k));
    vector<vector<double>> dist(q_points, vector<double>(k));
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
    int c = 10000; // Number of points for Corpus
    int q = 300;   // Number of Queries
    int d = 784;     // Dimensions

    vector<vector<double>> C, Q;

    int option;
    cout << "1.Random matrices   3.Small matrices for printing  Select and option: ";
    cin >> option;

    if (option == 1) {
        // Generate random C and Q matrices
        C.resize(c, vector<double>(d));
        Q.resize(q, vector<double>(d));
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < d; j++) {
                C[i][j] = rand() % 100;
            }
        }
        for (int i = 0; i < q; i++) {
            for (int j = 0; j < d; j++) {
                Q[i][j] = rand() % 100;
            }
        }
    } else if(option == 3) {
        C = {
            {1.0, 2.0},
            {4.0, 5.0},
            {7.0, 8.0},
            {10.0, 11.0},
            {13.0, 14.0},
            {16.0, 17.0},
            {19.0, 20.0},
            {22.0, 23.0},
            {25.0, 26.0},
            {28.0, 29.0}
        };

        Q = {
            {1.1, 2.6},
            {4.5, 5.2},
            {11.0, 8.0},
            {14.0, 15.0},
            {17.0, 18.0},
            {20.0, 21.0},
            {23.0, 24.0},
            {26.0, 27.0},
            {29.0, 30.0},
            {32.0, 33.0}
        };
    }   else {
        cout << "Invalid option" << endl;
        return 1;
    }

    // Perform k-NN search while measuring time
    auto start = omp_get_wtime();
    auto [idx, dist] = knnSearchParallel(C, Q, k);
    auto end = omp_get_wtime();
    cout << "knnsearch took " << (end - start) << " seconds" << endl;

    //printResults(idx, dist);

    return 0;
}
