#include <iostream>
#include <vector>
#include <cblas.h>
#include <H5Cpp.h>

// For testing
#include <random>
#include <chrono>

using namespace std;

void printResults(const vector<vector<int>>& idx, const vector<vector<double>>& dist) {
    for (int i = 0; i < idx.size(); i++) {
        std::cout << "Query " << i << ":\n";
        for (int j = 0; j < idx[i].size(); j++) {
            std::cout << "Neighbor " << j << ": index = " << idx[i][j] << ", distance = " << dist[i][j] << "\n";
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

void exportResults(const vector<vector<int>>& idx, const vector<vector<double>>& dist) {
    hsize_t queries = idx.size();
    hsize_t k = idx[0].size();
    hsize_t dims[2] = {queries, k};

    try {
        // Create a new HDF5 file
        H5::H5File file("results.h5", H5F_ACC_TRUNC);

        // Create idx dataset
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset_idx = file.createDataSet("idx", H5::PredType::NATIVE_INT, dataspace);
        vector<int> flat_idx;
        for (const auto& row : idx) {
            flat_idx.insert(flat_idx.end(), row.begin(), row.end());
        }
        dataset_idx.write(flat_idx.data(), H5::PredType::NATIVE_INT);

        // Create dist dataset
        H5::DataSet dataset_dist = file.createDataSet("dist", H5::PredType::NATIVE_DOUBLE, dataspace);
        vector<double> flat_dist;
        for (const auto& row : dist) {
            flat_dist.insert(flat_dist.end(), row.begin(), row.end());
        }
        dataset_dist.write(flat_dist.data(), H5::PredType::NATIVE_DOUBLE);

        cout << "Results exported to results.h5" << endl;

        // Close the file
        file.close();
    } catch (H5::Exception& error) {
        cerr << "Error: " << error.getDetailMsg() << endl;
    }
}

void importData(vector<vector<double>>& C, vector<vector<double>>& Q) {
    try {
        // Open the HDF5 file
        cout << "Enter the filename: ";
        string filename;
        cin >> filename;
        H5::H5File file(filename, H5F_ACC_RDONLY);

        // Read C dataset
        H5::DataSet dataset_C = file.openDataSet("C");
        H5::DataSpace dataspace_C = dataset_C.getSpace();
        hsize_t dims_C[2];
        dataspace_C.getSimpleExtentDims(dims_C, NULL);
        C.resize(dims_C[0], vector<double>(dims_C[1]));
        dataset_C.read(C[0].data(), H5::PredType::NATIVE_DOUBLE);

        // Read Q dataset
        H5::DataSet dataset_Q = file.openDataSet("Q");
        H5::DataSpace dataspace_Q = dataset_Q.getSpace();
        hsize_t dims_Q[2];
        dataspace_Q.getSimpleExtentDims(dims_Q, NULL);
        Q.resize(dims_Q[0], vector<double>(dims_Q[1]));
        dataset_Q.read(Q[0].data(), H5::PredType::NATIVE_DOUBLE);

        cout << "Data imported from data.h5" << endl;

        // Close the file
        file.close();
    } catch (H5::Exception& error) {
        cerr << "Error: " << error.getDetailMsg() << endl;
    }
}

// For testing
int main() {
    int k = 2;
    int c = 5000; // Number of points for Corpus
    int q = 50;    // Number of Queries
    int d = 3;      // Dimensions

    vector<vector<double>> C, Q;

    int option;
    cout << "1.Import matrices from .h5 file    2.Random matrices   3.Small matrices for printing\nSelect and option: ";
    cin >> option;

    if (option == 1) {
        importData(C, Q);
    } else if (option == 2) {
        // Generate random C and Q matrices
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.0, 10.0);
        vector<vector<double>> C(c, vector<double>(5));
        vector<vector<double>> Q(q, vector<double>(5));
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
    } else if(option == 3) {
        C = {
            {1.0, 2.0},
            {4.0, 5.0},
            {7.0, 8.0},
            {10.0, 11.0},
            {13.0, 14.0}
        };

        Q = {
            {1.1, 2.6},
            {4.5, 5.2},
            {11.0, 8.0}
        };
    }   else {
        cout << "Invalid option" << endl;
        return 1;
    }

    // Perform k-NN search while measuring time
    auto start = chrono::high_resolution_clock::now();
    auto [idx, dist] = knn_search(C, Q, k);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "knnsearch took " << elapsed.count() << " seconds." << endl;

    printResults(idx, dist);
    exportResults(idx, dist);

    return 0;
}