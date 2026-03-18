#ifndef PLMDCA_BACKEND_H
#define PLMDCA_BACKEND_H

    // Imports -----------------------------------------------------------------
    #include <cstdio>
    #include <cstdlib>
    #include <filesystem>
    #include <fstream>
    #include <iostream>
    #include <sstream>
    #include <string>
    #include <vector>
    #include <unordered_map>
    #include <unordered_set>
    #include <algorithm>
    #include <cmath>
    #include <numeric>
    #include <cstdint>  // For uint8_t
    #include <thread>   // Standard C++ multi-threading
    #include <mutex>    // Lock globals for multi-threading
    #include <chrono>   // To time code execution
    #include "../lbfgs/include/lbfgs.h"

    // Main: PlmDCA ------------------------------------------------------------
    class PlmDCA {
    public:
        
        // Constructor
        PlmDCA(
            const int m_n_states,
            const char* m_msa_path,
            const int m_msa_length,
            const bool* couplings_cutoff_flat,
            const float m_lambda_h, 
            const float m_lambda_J,
            const float m_lambda_asymptotic,
            const bool m_exclude_gaps,
            const float m_theta_regularization,
            const bool m_count_target_sequence,
            const bool m_use_weights,
            const float m_weights_seqid,
			const float* m_pos_weights,
            const int m_num_threads,
            const char* m_weights_cache_path,
            const bool m_verbose
        );

        // Constructor Methods
        std::vector<std::vector<bool>> unflattenCouplingFilter(
            const bool* couplings_cutoff_flat
        );
        std::vector<std::vector<uint8_t>> readSequences();
        std::vector<float> computeWeights();
        void countClustersForIndicies(
            const std::vector<int>& indices,
            std::vector<int>& counts,
            const int start_loop
        );
        void saveWeights();
        std::vector<float> readWeights();
        std::vector<std::vector<float>> computeCounts();
        std::vector<std::vector<float>> computeFrequencies();
        std::vector<std::vector<float>> computeHStar();
		float computeJijSums(const float*, int, std::vector<float>&, std::vector<float>&);
        std::vector<std::vector<int>> computeIjIndex();
        void initCoefficients(
            float* hJ
        );

        // Gradiend Descent Method
        float gradient(
            const float* hJ,
            float* grad
        );
        void update_position_gradient(
            const int i,
            const float* hJ,
            float* grad,
            float& loss,
            std::mutex& mtx
        );

        // Utility
        void logCouplingMatrix();
        void logCouplingList();
        void logTime(const char* log_str);

        // Properties
        int n_couplings;
        int n_h;
        int n_J;
        int n_hJ;
        float Neff;
        float Neff_inv;
            
    private:

        // Input Properties
        const int n_states;
        const char* msa_path;
        const int msa_length;
        const float lambda_h;
        const float lambda_J;
        const float lambda_asymptotic;
        float lambda_h_corrected;
        float lambda_J_corrected;
        const bool exclude_gaps;
        const float theta_regularization;
        const bool count_target_sequence;
        const bool use_weights;
        const float weights_seqid;
		const float* pos_weights;
        const int num_threads;
        const char* weights_cache_path;
        const bool verbose;
        std::vector<std::vector<int>> coupling_list;
        std::vector<std::vector<int>> coupling_list_left;
        std::vector<std::vector<int>> coupling_list_right;
        const int A;
        const int A_nogap;
        const int A2;
        const uint8_t gap_state;

        // Properties
        float dt;
        std::vector<std::vector<bool>> couplings_cutoff;
        std::vector<std::vector<uint8_t>> sequences;
        int msa_depth;
        std::vector<float> weights;
        std::vector<std::vector<float>> counts;
        std::vector<std::vector<float>> frequencies;
        std::vector<std::vector<float>> h_star;
        std::vector<std::vector<int>> ij_index;

    };

#endif
