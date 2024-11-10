#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cblas.h>

// Συνάρτηση για την υπολογισμό της απόστασης μεταξύ δύο σημείων
double compute_distance(const std::vector<double>& C, const std::vector<double>& Q) {
    double dist = 0.0;
    for (size_t i = 0; i < C.size(); i++) {
        dist += std::pow(C[i] - Q[i], 2);
    }
    return std::sqrt(dist);
}

// Συνάρτηση quickselect για την εύρεση των k κοντινότερων γειτόνων
void quickselect(std::vector<std::pair<double, int>>& distances, int k) {
    std::nth_element(distances.begin(), distances.begin() + k, distances.end());
    distances.resize(k);  // Κρατάμε μόνο τους k κοντινότερους
}

// Υπορουτίνα για την αναδρομική αναζήτηση των k κοντινότερων γειτόνων
void knn_recursive(const std::vector<std::vector<double>>& C, 
                   const std::vector<std::vector<double>>& Q, 
                   int k, 
                   std::vector<std::vector<int>>& idx, 
                   std::vector<std::vector<double>>& dist, 
                   int start, 
                   int end) {
    // Υπολογισμός των αποστάσεων για το υποσύνολο των σημείων
    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        std::vector<std::pair<double, int>> distances;
        for (size_t j = 0; j < C.size(); j++) {
            double dist = compute_distance(C[j], Q[i]);
            distances.push_back({dist, j});
        }

        // Quickselect για να βρούμε τους k κοντινότερους γείτονες
        quickselect(distances, k);

        // Αποθήκευση των αποτελεσμάτων
        for (int j = 0; j < k; j++) {
            idx[i][j] = distances[j].second;
            dist[i][j] = distances[j].first;
        }
    }
}

int main() {
    int k = 2; // Αριθμός κοντινότερων γειτόνων
    int n = 100000; // Αριθμός σημείων
    int d = 10;  // Διαστάσεις
    int q = 5000;   // Αριθμός ερωτημάτων

    // Δημιουργία τυχαίων δεδομένων για το σύνολο σημείων (C) και τα ερωτήματα (Q)
    std::vector<std::vector<double>> C(n, std::vector<double>(d));
    std::vector<std::vector<double>> Q(q, std::vector<double>(d));

    // Αρχικοποίηση των σημείων και των ερωτημάτων με τυχαίες τιμές
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

    // Πίνακες για τους δείκτες και τις αποστάσεις
    std::vector<std::vector<int>> idx(q, std::vector<int>(k));
    std::vector<std::vector<double>> dist(q, std::vector<double>(k));

    // Μέτρηση του χρόνου εκτέλεσης της συνάρτησης knn_recursive
    double start_time = omp_get_wtime();
    knn_recursive(C, Q, k, idx, dist, 0, q);
    double end_time = omp_get_wtime();

    // Εκτύπωση του χρόνου εκτέλεσης
    std::cout << "Time taken for knn_recursive: " << (end_time - start_time) << " seconds" << std::endl;

    // Εκτύπωση των αποτελεσμάτων
    //for (int i = 0; i < q; i++) {
    //    std::cout << "Query point " << i << " neighbors: ";
    //    for (int j = 0; j < k; j++) {
    //        std::cout << "(" << idx[i][j] << ", " << dist[i][j] << ") ";
    //    }
    //    std::cout << std::endl;
    //}

    return 0;
}
