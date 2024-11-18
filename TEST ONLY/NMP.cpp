#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cblas.h>
#include <random>
#include <ctime>

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
    //#pragma omp parallel for
    for (int i = 0; i < c_points; i++) {
        double sum = 0.0;
        for (int j = 0; j < d; ++j) {
            sum += C[i][j] * C[i][j];
        }
        CSquared[i] = sum;
    }

    // Calculate Q^2
    //#pragma omp parallel for
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
    #pragma omp parallel for
    for (int i =0; i < c_points; i++) {
        for (int j = 0; j < q_points; j++) {
            D[i][j] = CSquared[i] + QSquared[j] - 2 * CQT[i * q_points + j];
            if (D[i][j] < 0) {
                D[i][j] = 0;
            }
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
    if (point_pairs.size() <= k) return;    // Returns since the points are less or equal to k, but not in order

    int left = 0, right = point_pairs.size() - 1;
    while (left <= right) {
        int pivotIndex = left + (right - left) / 2;
        pivotIndex = partition(point_pairs, left, right, pivotIndex);
        if (pivotIndex == k) break;
        else if (pivotIndex < k) left = pivotIndex + 1;
        else right = pivotIndex - 1;
    }
}

vector<vector<double>> generateRandomProjections(int original_dim, int reduced_dim) {
    vector<vector<double>> projections(reduced_dim, vector<double>(original_dim));
    for (int i = 0; i < reduced_dim; i++) {
        for (int j = 0; j < original_dim; ++j) {
            projections[i][j] = (rand()% 2 == 0 ? -1 : 1) / sqrt((double)reduced_dim);
        }
    }

    return projections;
}

vector<vector<double>> projectPoints(const vector<vector<double>>& points, const vector<vector<double>>& projections) {
    int num_points = points.size();
    int reduced_dim = projections.size();
    int original_dim = points[0].size();

    vector<vector<double>> projected_points(num_points, vector<double>(reduced_dim));
    vector<double> points_flat(num_points * original_dim);
    vector<double> projections_flat(reduced_dim * original_dim);
    vector<double> projected_points_flat(num_points * reduced_dim);

    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            points_flat[i * original_dim + j] = points[i][j];
        }
    }

    for (int i = 0; i < reduced_dim; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            projections_flat[i * original_dim + j] = projections[i][j];
        }
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_points, reduced_dim, original_dim, 1.0, points_flat.data(), original_dim, projections_flat.data(), original_dim, 0.0, projected_points_flat.data(), reduced_dim);

    // Convert the flattened result back to a 2D vector
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < reduced_dim; ++j) {
            projected_points[i][j] = projected_points_flat[i * reduced_dim + j];
        }
    }
    
    return projected_points;
}

pair<vector<vector<int>>, vector<vector<double>>> knnSearch(const vector<vector<double>>& C, const vector<vector<double>>& Q, int k) {
    vector<vector<double>> D;
    vector<vector<int>> idx(Q.size());
    vector<vector<double>> dist(Q.size());
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
           dist[i][j] = sqrt(point_pairs[j].second);
        }
    }
    return {idx, dist};
}

pair<vector<vector<int>>, vector<vector<double>>> knnSearchParallel(const vector<vector<double>>& C, const vector<vector<double>>& Q, int k) {
    int c_points = C.size();
    int q_points = Q.size();
    int d = C[0].size();
    vector<vector<int>> idx(q_points, vector<int>(k));
    vector<vector<double>> dist(q_points, vector<double>(k));

    int num_subQs = (q_points + 49) / 50; // Number of sub-Queries needed
    vector<vector<vector<double>>> subQs(num_subQs);

    #pragma omp parallel for
    for (int i = 0; i < num_subQs; ++i) {
        int start_idx = i * 50;
        int end_idx = min(start_idx + 50, q_points);
        subQs[i].assign(Q.begin() + start_idx, Q.begin() + end_idx);
    }

    #pragma omp parallel for
    for (int i = 0; i < num_subQs; ++i) {
        auto [subidx, subdist] = knnSearch(C, subQs[i], k);
        for (int j = 0; j < subQs[i].size(); ++j) {
            idx[i * 50 + j] = subidx[j];
            dist[i * 50 + j] = subdist[j];
        }
    }

    return {idx, dist};
}

int main() {
    srand(time(0));
    int c, q, d, k;
    double const e = 0.3;

    vector<vector<double>> C, Q;

    int option;
    cout << "1.Random matrices   3.Small matrices for printing  Select and option: ";
    cin >> option;

    if (option == 1) {
        //importData(C, Q);
    } else if (option == 2) {
        // Ask for matrices size, dimensions and number of neighbors
        cout << "Enter the number of points for Corpus: ";
        cin >> c;
        cout << "Enter the number of Queries: ";
        cin >> q;
        cout << "Enter the number of dimensions: ";
        cin >> d;
        cout << "Enter the value of k: ";
        cin >> k;

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
        k = 2;
        C = {
            {1.0, 2.0, 3.0, 4.0, 5.0},
            {6.0, 7.0, 8.0, 9.0, 10.0},
            {11.0, 12.0, 13.0, 14.0, 15.0},
            {16.0, 17.0, 18.0, 19.0, 20.0},
            {21.0, 22.0, 23.0, 24.0, 25.0},
            {26.0, 27.0, 28.0, 29.0, 30.0},
            {31.0, 32.0, 33.0, 34.0, 35.0},
            {36.0, 37.0, 38.0, 39.0, 40.0},
            {1.1, 2.6, 3.1, 4.6, 5.1},
            {46.0, 47.0, 48.0, 49.0, 50.0}
        };

        Q = {
            {1.1, 2.6, 3.1, 4.6, 5.1},
            {6.5, 7.2, 8.5, 9.2, 10.5},
            {11.0, 12.0, 13.0, 14.0, 15.0},
            {16.0, 17.0, 18.0, 19.0, 20.0},
            {21.0, 22.0, 23.0, 24.0, 25.0},
            {26.0, 27.0, 28.0, 29.0, 30.0},
            {31.0, 32.0, 33.0, 34.0, 35.0},
            {36.0, 37.0, 38.0, 39.0, 40.0},
            {41.0, 42.0, 43.0, 44.0, 45.0},
            {46.0, 47.0, 48.0, 49.0, 50.0}
        };

    } else {
        cout << "Invalid option" << endl;
        return 1;
    }

    // Perform k-NN search while measuring time
    int cp = C.size();
    int qp = Q.size();
    int dim = C[0].size();
    int dl = log(cp) / (e*e);

    if(cp < 1000 && dim < dl) {
        auto start = omp_get_wtime();
        auto [idx, dist] = knnSearchParallel(C, Q, k);
        auto end = omp_get_wtime();

        cout << "knnsearch took " << (end - start) << " seconds" << endl;
        printResults(idx, dist);

    } else if (cp >= 1000 && dim > dl) {
        auto start = omp_get_wtime();
        dl = sqrt(dim) + 1;
        vector<vector<double>> CS, QS;
        vector<vector<double>> projections = generateRandomProjections(dim, dl);
        CS = projectPoints(C, projections);
        QS = projectPoints(Q, projections);

        auto [idx, dist] = knnSearchParallel(CS, QS, k);
        auto end = omp_get_wtime();

        cout << "knnsearch took " << (end - start) << " seconds" << endl;

    } else {
        auto start = omp_get_wtime();
        vector<vector<double>> CS, QS;
        vector<vector<double>> projections = generateRandomProjections(dim, dl);
        CS = projectPoints(C, projections);
        QS = projectPoints(Q, projections);
        auto [idx, dist] = knnSearchParallel(CS, QS, k);
        auto end = omp_get_wtime();

        cout << "knnsearch took " << (end - start) << " seconds" << endl;
    }

    return 0;
}