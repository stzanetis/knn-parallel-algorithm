#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cblas.h>
#include <random>
#include <H5Cpp.h>
#include <ctime>
#include <algorithm>

using namespace std;

void printResults(const vector<vector<int>>& idx, const vector<vector<float>>& dist) {
    for (int i = 0; i < idx.size(); i++) {
        std::cout << "Query " << i << ":\n";
        for (int j = 0; j < idx[i].size(); j++) {
            std::cout << "Neighbor " << j << ": index = " << idx[i][j] << ", distance = " << dist[i][j] << "\n";
        }
    }
}

void calculateDistances(const vector<vector<float>>& C, const vector<vector<float>>& Q, vector<vector<float>>& D) {
    int c_points = C.size();    // Number of points in C
    int q_points = Q.size();    // Number of points in Q
    int d = C[0].size();        // Number of dimensions in a point

    vector<float> CSquared(c_points);          // Vector for C^2
    vector<float> QSquared(q_points);          // Vector for Q^2
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
    #pragma omp parallel for
    for (int i = 0; i < c_points; i++) {
        double sum = 0.0;
        for (int j = 0; j < d; ++j) {
            sum += C[i][j] * C[i][j];
        }
        CSquared[i] = sum;
    }

    // Calculate Q^2
    #pragma omp parallel for
    for (int i = 0; i < q_points; i++) {
        float sum = 0.0;
        for (int j = 0; j < d; ++j) {
            sum += Q[i][j] * Q[i][j];
        }
        QSquared[i] = sum;
    }

    // Calculate C*Q^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c_points, q_points, d, 1.0, CFlat.data(), d, QFlat.data(), d, 0.0, CQT.data(), q_points);

    // Calculate D^2 using (C^2 - 2C*Q^T + Q^2T)
    D.resize(c_points, vector<float>(q_points));
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

int partition(vector<pair<int,float>>& point_pairs, int left, int right, int pivotIndex) {
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

void quickSelect(vector<pair<int,float>>& point_pairs, int k) {
    if (point_pairs.size() <= k) return;    // Returns since the points are less or equal to k, but not in order

    int left = 0, right = point_pairs.size() - 1;
    while (left <= right) {
        int pivotIndex = left + (right - left) / 2;
        pivotIndex = partition(point_pairs, left, right, pivotIndex);
        if (pivotIndex == k) break;
        else if (pivotIndex < k) left = pivotIndex + 1;
        else right = pivotIndex - 1;
    }

    // Sort the first k elements
    sort(point_pairs.begin(), point_pairs.begin() + k, [](const pair<int, float>& a, const pair<int, float>& b) {
        return a.second < b.second;
    });
}

void randomProjection(const vector<vector<float>>& C, const vector<vector<float>>& Q, int dl, vector<vector<float>>& CS, vector<vector<float>>& QS) {
    int c_points = C.size();
    int q_points = Q.size();
    int d = C[0].size();

    vector<double> CSTemp(c_points * dl);
    vector<double> QSTemp(q_points * dl);

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


    vector<double> projections(dl * d);
    for (int i = 0; i < dl * d; i++) {
        projections[i] = (rand()% 2 == 0 ? -1 : 1) / sqrt((double)dl);
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c_points, dl, d, 1.0, CFlat.data(), d, projections.data(), dl, 0.0, CSTemp.data(), dl);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, q_points, dl, d, 1.0, QFlat.data(), d, projections.data(), dl, 0.0, QSTemp.data(), dl);

    CS.resize(c_points, vector<float>(dl));
    for (int i = 0; i < c_points; ++i) {
        for (int j = 0; j < dl; ++j) {
            CS[i][j] = static_cast<float>(CSTemp[i * dl + j]);
        }
    }

    QS.resize(q_points, vector<float>(dl));
    for (int i = 0; i < q_points; ++i) {
        for (int j = 0; j < dl; ++j) {
            QS[i][j] = static_cast<float>(QSTemp[i * dl + j]);
        }
    }
}

pair<vector<vector<int>>, vector<vector<float>>> knnSearch(const vector<vector<float>>& C, const vector<vector<float>>& Q, int k) {
    vector<vector<float>> D;
    vector<vector<int>> idx(Q.size());
    vector<vector<float>> dist(Q.size());
    calculateDistances(C, Q, D);

    for (int i = 0; i < Q.size(); i++) {
        vector<pair<int,float>> point_pairs;
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

pair<vector<vector<int>>, vector<vector<float>>> knnSearchParallel(const vector<vector<float>>& C, const vector<vector<float>>& Q, int k) {
    int c_points = C.size();
    int q_points = Q.size();
    int d = C[0].size();
    vector<vector<int>> idx(q_points, vector<int>(k));
    vector<vector<float>> dist(q_points, vector<float>(k));

    int chunk_size = 300;
    int num_subQs = (q_points + (chunk_size-1)) / chunk_size; // Number of sub-Queries needed
    vector<vector<vector<float>>> subQs(num_subQs);

    #pragma omp parallel for
    for (int i = 0; i < num_subQs; ++i) {
        int start_idx = i * chunk_size;
        int end_idx = min(start_idx + chunk_size, q_points);
        subQs[i].assign(Q.begin() + start_idx, Q.begin() + end_idx);
    }

    #pragma omp parallel for
    for (int i = 0; i < num_subQs; ++i) {
        auto [subidx, subdist] = knnSearch(C, subQs[i], k);
        for (int j = 0; j < subQs[i].size(); ++j) {
            idx[i * chunk_size + j] = subidx[j];
            dist[i * chunk_size + j] = subdist[j];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < q_points; ++i) {
        for (int j = 0; j < k; ++j) {
            idx[i][j] += 1;
        }
    }

    return {idx, dist};
}

void exportResults(const vector<vector<int>>& idx, const vector<vector<float>>& dist) {
    hsize_t queries = idx.size();
    hsize_t k = idx[0].size();
    hsize_t dims[2] = {queries, k};

    try {
        // Create a new HDF5 file
        H5::H5File file("omp-results.h5", H5F_ACC_TRUNC);

        // Create idx dataset
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset_idx = file.createDataSet("idx", H5::PredType::NATIVE_INT, dataspace);
        vector<int> flat_idx;
        for (const auto& row : idx) {
            flat_idx.insert(flat_idx.end(), row.begin(), row.end());
        }
        dataset_idx.write(flat_idx.data(), H5::PredType::NATIVE_INT);

        // Create dist dataset
        H5::DataSet dataset_dist = file.createDataSet("dist", H5::PredType::NATIVE_FLOAT, dataspace);
        vector<float> flat_dist;
        for (const auto& row : dist) {
            flat_dist.insert(flat_dist.end(), row.begin(), row.end());
        }
        dataset_dist.write(flat_dist.data(), H5::PredType::NATIVE_FLOAT);

        cout << "Results exported to omp-results.h5" << endl;

        // Close the file
        file.close();
    } catch (H5::Exception& error) {
        cerr << "Error: " << error.getDetailMsg() << endl;
    }
}

void importData(vector<vector<float>>& C, vector<vector<float>>& Q) {
    try {
        // Open the HDF5 file
        string filename;
        cout << "Enter the filename(file should be located in the test folder): ";
        cin >> filename;
        H5::H5File file("../../test/" + filename, H5F_ACC_RDONLY);

        // Read C dataset
        H5::DataSet datasetC = file.openDataSet("train");
        H5::DataSpace dataspaceC = datasetC.getSpace();
        hsize_t dimsC[2];
        dataspaceC.getSimpleExtentDims(dimsC, NULL);
        C.resize(dimsC[0], vector<float>(dimsC[1]));
        vector<float> flatC(dimsC[0] * dimsC[1]);
        datasetC.read(flatC.data(), H5::PredType::NATIVE_FLOAT);

        // Copy data from the flatC array to matrix C
        for (hsize_t i = 0; i < dimsC[0]; ++i) {
            for (hsize_t j = 0; j < dimsC[1]; ++j) {
                C[i][j] = flatC[i * dimsC[1] + j];
            }
        }

        // Read Q dataset
        H5::DataSet datasetQ = file.openDataSet("test");
        H5::DataSpace dataspaceQ = datasetQ.getSpace();
        hsize_t dimsQ[2];
        dataspaceQ.getSimpleExtentDims(dimsQ, NULL);
        Q.resize(dimsQ[0], vector<float>(dimsQ[1]));
        vector<float> flatQ(dimsQ[0] * dimsQ[1]);
        datasetQ.read(flatQ.data(), H5::PredType::NATIVE_FLOAT);

        // Copy data from the flatQ array to matrix Q
        for (hsize_t i = 0; i < dimsQ[0]; ++i) {
            for (hsize_t j = 0; j < dimsQ[1]; ++j) {
                Q[i][j] = flatQ[i * dimsQ[1] + j];
            }
        }

        cout << "Data imported" << endl;

        // Close the file
        file.close();
    } catch (H5::Exception& error) {
        cerr << "Error: " << error.getDetailMsg() << endl;
    }
}

int main() {
    srand(time(0));
    int c, q, d, k;
    float const e = 0.3;
    vector<vector<float>> C, Q;

    int option;
    cout << "1.Import matrices from .h5 file    2.Random matrices   3.Small matrices for printing\nSelect an option: ";
    cin >> option;

    cout << "Enter the value for k nearest neighbors: ";
    cin >> k;

    if (option == 1) {
        importData(C, Q);
    } else if (option == 2) {
        // Ask for matrices size, dimensions and number of neighbors
        cout << "Enter the number of points for Corpus: ";
        cin >> c;
        cout << "Enter the number of Queries: ";
        cin >> q;
        cout << "Enter the number of dimensions: ";
        cin >> d;
        
        // Generate random C and Q matrices
        C.resize(c, vector<float>(d));
        Q.resize(q, vector<float>(d));
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
            {1.0, 2.0, 3.0, 4.0, 5.0},
            {45.4, 35.2, 8.0, 9.78, 0.1},
            {13.0, 14.35, 89.0, 14.0, 15.0},
            {16.0, 17.0, 18.0, 19.0, 20.0},
            {21.0, 22.0, 28.0, 243.0, 25.0},
            {26.0, 223.0, 28.0, 29.0, 30.0},
            {31.0, 32.0, 33.0, 34.0, 35.0},
            {36.0, 39.0, 38.0, 19.0, 4.0},
            {1.1, 2.6, 3.1, 4.6, 53.1},
            {41.0, 0.0, 48.0, 49.0, 50.0}
        };

        Q = {
            {1.1, 2.6, 3.1, 5.6, 5.1},
            {6.5, 7.2, 11.5, 9.2, 10.5},
            {11.0, 198.0, 13.0, 14.0, 15.0},
            {16.0, 13.0, 18.0, 13.0, 21.0},
            {25.0, 72.0, 23.0, 24.0, 85.0},
            {26.0, 27.0, 48.0, 49.0, 37.0},
            {1.0, 0.0, 3.0, 14.0, 35.0},
            {36.0, 37.0, 38.0, 576.0, 40.0},
            {41.0, 42.0, 4.0, 0.544, 8.0},
            {41.0, 47.0, 8.0, 49.0, 599.0}
        };
    } else {
        cout << "Invalid option" << endl;
        return 1;
    }

    // Perform k-NN search while measuring time
    int cp = C.size();
    int qp = Q.size();
    int dim = C[0].size();
    int dl = 3 * (log(cp) / (e*e));

    if((cp > 1000 && qp > 1000) && dl < dim) {
        auto start = omp_get_wtime();
        vector<vector<float>> CS, QS;
        randomProjection(C, Q, dl, CS, QS);
        auto [idx, dist] = knnSearchParallel(CS, QS, k);
        auto end = omp_get_wtime();

        cout << "knnsearch took " << (end - start) << " seconds" << endl;
        
        exportResults(idx, dist);

    } else {
        auto start = omp_get_wtime();
        auto [idx, dist] = knnSearchParallel(C, Q, k);
        auto end = omp_get_wtime();

        cout << "knnsearch took " << (end - start) << " seconds" << endl;
        
        exportResults(idx, dist);

    }

    return 0;
}
