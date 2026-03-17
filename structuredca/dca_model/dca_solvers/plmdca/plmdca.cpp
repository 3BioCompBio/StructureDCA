
// Header ----------------------------------------------------------------------
#if defined(_OPENMP)
    #include <omp.h>
#endif
#include "include/plmdca.h"

/*
[Sparse plmDCA solver in C++]
This part is the plmDCA solver itself (where we do all the maths)
Author: Matsvei Tsishyn
*/

// Constructor -----------------------------------------------------------------
PlmDCA::PlmDCA(
    const int m_n_states,
    const char* m_msa_path,
    const int m_msa_length,
    const bool* m_couplings_cutoff_flat,
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
):
n_states(m_n_states),
msa_path(m_msa_path),
msa_length(m_msa_length), 
lambda_h(m_lambda_h),   
lambda_J(m_lambda_J),
lambda_asymptotic(m_lambda_asymptotic),
exclude_gaps(m_exclude_gaps),
theta_regularization(m_theta_regularization),
count_target_sequence(m_count_target_sequence),
use_weights(m_use_weights),
weights_seqid(m_weights_seqid),
pos_weights(m_pos_weights),
num_threads(m_num_threads),
weights_cache_path(m_weights_cache_path),
verbose(m_verbose),
coupling_list(m_msa_length),
coupling_list_left(m_msa_length),
coupling_list_right(m_msa_length),
A(m_n_states),                                        // Number of states: 21
A_nogap(m_exclude_gaps ? m_n_states - 1 : m_n_states) // Number of states, eventually excluding gaps: 21 or 20
{

    // Init time metric
    this->dt = 0.0f;

    // Unflatten couplings_cutoff and counts h and J coefficients
    this->n_couplings = 0;
    this->couplings_cutoff = unflattenCouplingFilter(m_couplings_cutoff_flat);
    this->n_h = this->msa_length * this->n_states;
    this->n_J = this->n_couplings * (this->n_states * this->n_states);
    this->n_hJ = this->n_h + this->n_J;

    // Log counts
    //std::cout << n_couplings << std::endl;
    //std::cout << n_h << std::endl;
    //std::cout << n_J << std::endl;
    //std::cout << n_hJ << std::endl;

    // Read MSA
    if(this->verbose) {
        std::cout << " * plmDCA: read sequences from fasta file: '" << this->msa_path << "'" << std::endl;
    }
    this->sequences = readSequences();
    this->msa_depth = this->sequences.size();

    // Set weights for MSA sequences
    // Read weights from cache (weights_cache_path)
    if (!std::string(this->weights_cache_path).empty() && std::filesystem::exists(this->weights_cache_path)) {
        if (this->verbose) {
            std::cout << " * plmDCA: read MSA sequences weights from weights_cache_path: '" << this->weights_cache_path  << "'" << std::endl;
        }
        this->weights = this->readWeights();
    // Compute weights
    } else {
        if (this->verbose && this->use_weights) {
            std::cout << " * plmDCA: compute MSA sequences weights" << std::endl;
        }
        this->weights = this->computeWeights(); // NOTE: still need to initialize weights even if use_weights=false
    }
    // Compute Neff
    this->Neff = std::accumulate(this->weights.begin(), this->weights.end(), 0.0f);
    if(this->verbose && this->use_weights) {
        std::cout << "    - Neff: " << this->Neff << std::endl;
    }
    // Write weights to cache
    if(!std::string(this->weights_cache_path).empty() && !std::filesystem::exists(this->weights_cache_path)){
        if (this->verbose) {
            std::cout << " * plmDCA: save MSA sequences weights to weights_cache_path: '" << this->weights_cache_path  << "'" << std::endl;
        }
        this->saveWeights();
    }

    // Compute Counts
    if(this->verbose) {
        std::cout << " * plmDCA: compute counts, frequencies and h* (single-site, pseudolikelihood model)" << std::endl;
    }
    this->counts = this->computeCounts();

    // Compute Frequencies
    this->frequencies = this->computeFrequencies();

    // Compute h* (single-site Boltzman solution)
    this->h_star = this->computeHStar();

    // Initialize start coupling index of (i, j) pair
    this->ij_index = this->computeIjIndex();

    // Correct lambda_h and lambda_J: asymptotic correction for L2 regularization
    // -> So that there is still some small regularization even when Neff tends to infinity
    this->lambda_h_corrected = this->lambda_h * (1.0f + this->Neff * this->lambda_asymptotic);
    this->lambda_J_corrected = this->lambda_J * (1.0f + this->Neff * this->lambda_asymptotic);
    if(this->verbose) {
        std::cout << " * plmDCA: L2-regularization asymptotic correction: lambda_asymptotic = " << this->lambda_asymptotic << std::endl;
        std::cout << "    - lambda_h: " << this->lambda_h << " -> " << this->lambda_h_corrected  << std::endl;
        std::cout << "    - lambda_J: " << this->lambda_J << " -> " << this->lambda_J_corrected  << std::endl;
    }

    // Logs
    //this->logCouplingMatrix();
    //this->logCouplingList();
}


// Methods ---------------------------------------------------------------------

// Unflatter couplings_cutoff: List to symmetric matrix
std::vector<std::vector<bool>> PlmDCA::unflattenCouplingFilter(const bool* couplings_cutoff_flat)
{
    std::vector<std::vector<bool>> couplings_cutoff(this->msa_length, std::vector<bool>(this->msa_length, true));
    const int L = this->msa_length;
    int index = 0;
    for(int i = 0; i < L - 1; ++i){
        for(int j = i + 1; j < L; ++j){
            auto is_coupling = couplings_cutoff_flat[index];
            if(is_coupling){
                this->n_couplings ++;
                this->coupling_list[i].push_back(j);
                this->coupling_list[j].push_back(i);
                this->coupling_list_right[i].push_back(j);
                this->coupling_list_left[j].push_back(i);
            }
            couplings_cutoff[i][j] = is_coupling;
            couplings_cutoff[j][i] = is_coupling;
            index ++;
        }
    }
    return couplings_cutoff;
}

// Read sequences from MSA (.fasta) file
std::vector<std::vector<uint8_t>> PlmDCA::readSequences()
{

    // Init residues mapping to int
    std::unordered_map<char, uint8_t> res_mapping;
    res_mapping['A'] = 0;  res_mapping['C'] = 1;  res_mapping['D'] = 2;
    res_mapping['E'] = 3;  res_mapping['F'] = 4;  res_mapping['G'] = 5;
    res_mapping['H'] = 6;  res_mapping['I'] = 7;  res_mapping['K'] = 8;
    res_mapping['L'] = 9;  res_mapping['M'] = 10; res_mapping['N'] = 11;
    res_mapping['P'] = 12; res_mapping['Q'] = 13; res_mapping['R'] = 14;
    res_mapping['S'] = 15; res_mapping['T'] = 16; res_mapping['V'] = 17;
    res_mapping['W'] = 18; res_mapping['Y'] = 19; res_mapping['-'] = 20;
    res_mapping['.'] = 20; res_mapping['~'] = 20; res_mapping['B'] = 20;
    res_mapping['J'] = 20; res_mapping['O'] = 20; res_mapping['U'] = 20;
    res_mapping['X'] = 20; res_mapping['Z'] = 20;

    // Init
    std::vector<std::vector<uint8_t>> sequences;
    std::unordered_set<std::string> unique_sequences_set; // For redundency cheking
    std::ifstream msa_path_stream(this->msa_path);
    std::string current_line;
    int unique_seq_counter = 0;
    int seq_counter = 0;

    // Check file streaming
    if(msa_path_stream.fail()){
        std::cerr << "ERROR in StructureDCA::plmDCA.saveWeights(): Unable to open MSA file: " << this->msa_path << std::endl;
        throw std::runtime_error("Unable to open file containing the MSA data");
    }

    // Loop on lines of the file
    while(std::getline(msa_path_stream, current_line)){
        if(!current_line.empty() && current_line[0] != '>') { // Skip header and empty lines
            ++seq_counter;
            if (unique_sequences_set.find(current_line) == unique_sequences_set.end()) { // Skip redundent lines
                std::vector<uint8_t> current_seq;
                current_seq.reserve(this->msa_length); // optimize by putting the vector in the correct size which is known
                for (char c : current_line) {
                    current_seq.push_back(res_mapping.at(toupper(c)));
                }
                sequences.push_back(current_seq);
                unique_sequences_set.insert(current_line);
                ++unique_seq_counter;
            }
        }
    }

    // Log and return
    if(this->verbose) {
        std::cout << "    - MSA [L, N] = [" << this->msa_length << ", " << unique_seq_counter << "]" << std::endl;
        //std::cout << "    - target sequence length: " << this->msa_length << std::endl;
        //std::cout << "    - total sequences: " << seq_counter << std::endl;
        //std::cout << "    - unique sequences: " << unique_seq_counter << std::endl;
    }
    return sequences;
}

// Compute sequences weight
std::vector<float> PlmDCA::computeWeights()
{

    // Init
    auto t1 = std::chrono::high_resolution_clock::now();

    // No weighting case: Set all weights to 1
    if (!this->use_weights){
        std::vector<float> weights(this->msa_depth, 1.0f);
        if(!this->count_target_sequence) {weights[0] = 0.0f;}
        return weights;
    }
    
    // Count or ignore first sequence for weights computations by starting loop at 0 or 1
    int start_loop = this->count_target_sequence ? 0 : 1;

    // Initialize the per-thread cluster counts vectors
    std::vector<std::vector<int>> cluster_counts_by_thread(
        num_threads, std::vector<int>(this->msa_depth, 0)
    );

    // Separate indices in chunks for each thread
    // * Trick: Since we only loop on half (i, j)-matrix (j < i), first i iterations will stop much earlier than last,
    //          so we distribute i indices evenly across threads, so they all terminate approximatively at the same time
    std::vector<std::vector<int>> indicies_by_thread(num_threads);
    for (int i = start_loop; i < this->msa_depth; ++i) {
        int thread_id = i % num_threads;
        indicies_by_thread[thread_id].push_back(i);
    }

    // Manage multi-threading
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back( // ok here some magic
            [this, &indicies_by_thread, &cluster_counts_by_thread, t, start_loop]() {
            countClustersForIndicies(indicies_by_thread[t], cluster_counts_by_thread[t], start_loop); // compute cluster by chunks
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    // Merge thread counts into global counts
    std::vector<int> cluster_counts(this->msa_depth, 1);
    for (const auto& thread_cluster_counts : cluster_counts_by_thread) {
        for (int i = 0; i < this->msa_depth; ++i) {
            cluster_counts[i] += thread_cluster_counts[i];
        }
    }
    
    // Convert counts to weights
    std::vector<float> weights(this->msa_depth);
    for(int i = 0; i < this->msa_depth; ++i){
        weights[i] = 1.0f/ static_cast<float>(cluster_counts[i]);
    }

    // Remove first sequences weight (that was initally assigned to 1.0)
    if(!this->count_target_sequence) {
        weights[0] = 0.0f;
    }

    // Log time
    auto t2 = std::chrono::high_resolution_clock::now();
    this->dt = std::chrono::duration<float>(t2 - t1).count();
    if(this->verbose){
        this->logTime("weighting time: ");
    }

    // Return
    return weights;
}

void PlmDCA::countClustersForIndicies(
    const std::vector<int>& indices,
    std::vector<int>& counts,
    const int start_loop
)
{

    /* Given a set of indices (for positions in the MSA),computes the clusters generated by those positions with all other positions.
        Is designed to be multi-threaded on MSA positions indexes.
    */

    // Init
    char char_i;
    char char_j;
    int num_aligned_residues;
    int num_identical_residues;
    //int identical_residues_thr = static_cast<int>(this->weights_seqid * this->msa_length);

    // Loop on range
    for (auto i : indices) {
        const auto& seq_i = this->sequences[i];
        
        // Loop on other sequences j < i (half matrix because (i, i)=(j, i))
        for (int j = start_loop; j < i; ++j) {
            const auto& seq_j = this->sequences[j];

            // Compute weights_seqid(i, j)
            num_aligned_residues = 0;
            num_identical_residues = 0;
            for (int site = 0; site < this->msa_length; ++site) {
                char_i = seq_i[site];
                char_j = seq_j[site];
                if(char_i != 20 && char_j != 20) {
                    num_aligned_residues += 1;
                    num_identical_residues += char_i == char_j;
                }
            }
            
            // Update if (i, j) in same cluster
            if(static_cast<float>(num_identical_residues) > this->weights_seqid * static_cast<float>(num_aligned_residues)) {
                ++counts[i];
                ++counts[j];                    
            }
            //if (num_identical_residues > identical_residues_thr) {
            //    ++counts[i];
            //    ++counts[j];
            //}
        }
    }
}

void PlmDCA::saveWeights(){

    // Init
    std::ofstream weights_file(this->weights_cache_path);
    if(!weights_file.is_open()){
        std::cerr << "ERROR in StructureDCA::plmDCA.saveWeights(): Unable to open weights_cache_path: " << this->weights_cache_path << std::endl;
        throw std::runtime_error("Unable to open weights_cache_path");
    }

    // Write to file
    for(int i = 0; i < this->msa_depth; ++i){
        weights_file << this->weights[i] << '\n';
    }
    weights_file.close();
}

std::vector<float> PlmDCA::readWeights(){

    // Init
    std::ifstream weights_file(this->weights_cache_path);
    if(!weights_file.is_open()){
        std::cerr << "ERROR in StructureDCA::plmDCA.readWeights(): Unable to open weights_cache_path: " << this->weights_cache_path << std::endl;
        throw std::runtime_error("Unable to open weights_cache_path");
    }

    // Read
    std::vector<float> weights(this->msa_depth);
    std::string line;
    float weight_i;
    for(int i = 0; i < this->msa_depth; ++i){
        if(!std::getline(weights_file, line)){
            std::cerr << "ERROR in StructureDCA::plmDCA.readWeights(): Not enough lines in weights_cache_path file: " << this->weights_cache_path << std::endl;
            throw std::runtime_error("Not enough lines in weights_cache_path file");
        }
        // Convert to float and savety check (little magic here)
        std::istringstream iss(line);
        if(!(iss >> weight_i)){
            std::cerr << "ERROR in StructureDCA::plmDCA.readWeights(): Invalid line in file: " << this->weights_cache_path << std::endl;
            throw std::runtime_error("Invalid line in file");
        }
        weights[i] = weight_i;
    }

    // Check coherence and return
    if(std::getline(weights_file, line)){
        std::cerr << "ERROR in StructureDCA::plmDCA.readWeights(): Too much lines in weights_cache_path file: " << this->weights_cache_path << std::endl;
        throw std::runtime_error("Too much lines in weights_cache_path file");
    }
    return weights;
}

std::vector<std::vector<float>> PlmDCA::computeCounts()
{
    std::vector<std::vector<float>> counts(this->msa_length, std::vector<float>(this->n_states));
    for(int n = 0; n < msa_depth; ++n){
        auto& seq = sequences[n];
        float weight = weights[n];
        for(int i = 0; i < msa_length; ++i){
            auto aa = seq[i];
            counts[i][aa] += weight;
        }
    }
    return counts;
}

std::vector<std::vector<float>> PlmDCA::computeFrequencies()
{

    // Init regularization terms
    float reg_term = this->theta_regularization / this->A;
    float reg_factor = 1.0f - this->theta_regularization;

    // Compute frequencies
    std::vector<std::vector<float>> frequencies(this->msa_length, std::vector<float>(this->n_states));
    for(int i = 0; i < this->msa_length; ++i){
        auto& current_counts = this->counts[i];
        auto& current_frequencies = frequencies[i];
        for(int aa = 0; aa < this->n_states; ++aa){
            float freq_theta_reg = (reg_factor * (current_counts[aa] / this->Neff)) + reg_term;
            current_frequencies[aa] = freq_theta_reg;
        }
    }
    return frequencies;
}

std::vector<std::vector<float>> PlmDCA::computeHStar()
{
    std::vector<std::vector<float>> h_star(this->msa_length, std::vector<float>(this->n_states));
    for(int i = 0; i < this->msa_length; ++i){
        const auto w_i = this->pos_weights[i];
        std::vector<float> h_star_i(this->n_states, 0.0f);
        for(int a = 0; a < this->n_states; ++a){
            h_star_i[a] = std::log(this->frequencies[i][a]);
        }
        float h_star_sum = std::accumulate(h_star_i.begin(), h_star_i.end(), 0.0f);
        float h_star_mean = h_star_sum / this->n_states;
        for(int a = 0; a < this->A_nogap; ++a){
            h_star_i[a] = (h_star_i[a] - h_star_mean) * w_i;
        }
        if(this->exclude_gaps){
            h_star_i[this->n_states-1] = 0.0f;
        }
        h_star[i] = h_star_i;
    }
    return h_star;
}

std::vector<std::vector<int>> PlmDCA::computeIjIndex()
{
    /*Create cache for mapping position-pair [i, j] -> index in hJ coefficients vector (that is flat).*/
    const auto L = this->msa_length;
    const auto A = this->n_states;
    const auto A2 = A * A;
    const auto LA = L * A;
    std::vector<std::vector<int>> ij_index(this->msa_length, std::vector<int>(this->msa_length));
    int index = 0;
    for(int i = 0; i < L - 1; ++i){
        const auto& couplings_cutoff_i = this->couplings_cutoff[i];
        for(int j = i + 1; j < L; ++j){
            if(couplings_cutoff_i[j]) { // Only assign indicies for couplings positions
                int k = LA + index * A2;
                ij_index[i][j] = k;
                ij_index[j][i] = k;
                index ++;
            }
        }
    }
    return ij_index;
}

float PlmDCA::computeJijSums(const float* hJ, int start_index, std::vector<float>& Jij_sums_a, std::vector<float>& Jij_sums_b)
{
    /*Compute \sum_b Jij(a,b), \sum_a Jij(a,b), \sum_a \sum b Jij(a,b).*/
	const auto A = this->A;
	const auto A_nogap = this->A_nogap;
	Jij_sums_a.assign(A, 0.0f);
	Jij_sums_b.assign(A, 0.0f);
	float Jij_sum = 0.0f;
	for (int a = 0; a < A_nogap; ++a){
		for (int b = 0; b < A_nogap; ++b){
			float Jijab = hJ[start_index + a*A + b];
			Jij_sums_a[a] += Jijab;
			Jij_sums_b[b] += Jijab;
			Jij_sum += Jijab;
		}
	}
	return Jij_sum;
}


// Initialize fields and couplings
void PlmDCA::initCoefficients(float* hJ)
{
    // Init h and J to zero
    for(int i = 0; i < this->n_hJ; ++i){
        hJ[i] = 0.0f;
    }

    // Set h as h* (single-site Boltzmann)
    int index = 0;
    for(int i = 0; i < this->msa_length; ++i){
        auto& h_star_i = this->h_star[i];
        for(int a = 0; a < this->n_states; ++a){
            hJ[index] = h_star_i[a];
            index ++;
        }
    }
}


// Main Methods: gradient --------------------------------------------------------------------------

// Compute gradients from site probabilities
float PlmDCA::gradient(const float* hJ, float* grad)
{
    /*Computes the gradient of the negative psuedolikelihood from alignment data and L2-regularization
    Parameters
        hJ      : Array of fields and couplings
        grad    : Array of gradients 
    Returns
        loss    : Value of objective function
    */
   
    // Init values ---------------------------------------------------------------------------------
    auto t1 = std::chrono::high_resolution_clock::now();
    const auto L = this->msa_length;
    const auto A = this->A;
    const auto A_nogap = this->A_nogap;
    const auto lh = this->lambda_h_corrected;
    const auto lJ = this->lambda_J_corrected;
    const auto A2 = A * A;
    const auto Neff_inv = 1.0f / this->Neff;

    // Init objective function value
    float loss = 0.0f;


    // L2 regularization terms ---------------------------------------------------------------------
    const auto lh_Neff_inv = lh * Neff_inv;
    const auto two_lh_Neff_inv = 2.0f * lh_Neff_inv;
    const auto lJ_Neff_inv = 2.0f * lJ * Neff_inv;
    const auto two_lJ_Neff_inv = 2.0f * lJ_Neff_inv;
    
    // L2 for h
    int index_h = 0;
    for(int i = 0; i < L; ++i){
        for(int a = 0; a < A; ++a){
            const float hia = hJ[index_h];
            grad[index_h] = two_lh_Neff_inv * hia;
            loss += lh_Neff_inv * hia * hia;
            index_h ++;
        }
    }

    // L2 for J (just loop on remaining indexes)
    for(int index_J=index_h; index_J < this->n_hJ; ++index_J){
        const float Jijab = hJ[index_J];
        grad[index_J] = two_lJ_Neff_inv * Jijab;
        loss += lJ_Neff_inv * Jijab * Jijab;
    }


    // Gradients of log-pseudolikelihood from alignment data ---------------------------------------
    #pragma omp parallel for num_threads(this->num_threads)
    for(int i = 0; i < L; ++i){ // Loop on positions

        // Init local variables for position i
        std::vector<float> prob_ni(A, 0.0f);                       // Prabability for each a (at position i for sequence n)
        std::vector<float> w_prob_ni(A, 0.0f);                     // Weighted Probability
        std::vector<float> fields_gradient_i(A, 0.0f);             // Grad(h)
        std::vector<float> couplings_gradient_i(L * A * A, 0.0f);  // Grad(J)
        float lossi = 0.0f;                                        // Objective function for position i
        const int Ai = A * i;
        //const auto& coupling_list_i = this->coupling_list[i];
        const auto& coupling_list_left_i = this->coupling_list_left[i];
        const auto& coupling_list_right_i = this->coupling_list_right[i];
		const auto w_i = this->pos_weights[i];


        // Init ij_index map at position i
        const auto& ij_index_i = this->ij_index[i];
        
        for(int n = 0; n < this->msa_depth; ++n){ // Loop on sequences

            // Init probability at position i for seq n
            std::vector<float> prob_ni(A, 0.0f);                     // Prabability for each a
            for(int a = 0; a < A; ++a){
                prob_ni[a] = 0.0f;
            }

            // Init
            const auto& current_seq = this->sequences[n];
            const auto w_n = this->weights[n];
			const auto w_ni = w_n * w_i;
            const auto a_ni = current_seq[i];

            // Ignore optimization target aa a_ni is a gap
            if(this->exclude_gaps && a_ni == A-1){
                continue;
            }


            // Compute Pi(n) -----------------------------------------------------------------------

            // Compute for h
            for(int a = 0; a < A_nogap; ++a){
                prob_ni[a] += hJ[Ai + a];
            }
            
            // Compute for J (j<i)
            int k_j;
            for(int j : coupling_list_left_i){
				if(this->exclude_gaps && current_seq[j] == A_nogap){
					continue;
				}
                k_j = ij_index_i[j]+ A*current_seq[j];
                for(int a = 0; a < A_nogap; ++a){
                    prob_ni[a] += hJ[k_j + a];
                }
            }

            // Compute for J (i<j)
            for(int j : coupling_list_right_i){
				if(this->exclude_gaps && current_seq[j] == A_nogap){
					continue;
				}
                k_j = ij_index_i[j] + current_seq[j];
                for(int a = 0; a < A_nogap; ++a){
                    prob_ni[a] += hJ[k_j + A*a];
                }
            }

            // Exponentiate
            for(int a = 0; a <  A_nogap; ++a){
                prob_ni[a] = std::exp(prob_ni[a]);
            }
            
            // Normalize to sum=1
            const float sum_prob_ni = std::accumulate(prob_ni.begin(), prob_ni.end(), 0.0f);
            for(int a = 0; a < A_nogap; ++a){
                prob_ni[a] /= sum_prob_ni;
            }

            // Accumulate log(P) in objective function for current a_ni of seq n
            lossi -= w_ni * std::log(prob_ni[a_ni]);

            // Weighted probability
            for(int a = 0; a < A_nogap; ++a){
                w_prob_ni[a] = w_ni * prob_ni[a];
            }

            
            // Compute Gradianst ot position i -----------------------------------------------------

            // Set Grad(h): Grad(h_ni(a)) = w_n*w_i*(P_ni(a) - delta(a=a_ni)) (summed over all sequences n)
            fields_gradient_i[a_ni] -= w_ni;
            for(int a = 0; a < A_nogap; ++a){
                fields_gradient_i[a] += w_prob_ni[a];
            }
            
            // Set Grad(J): Grad(J_nij(a, b)) = w*(P_ni(a) - delta(a=a_ni)) (summed over all sequences n)
            for(int j : coupling_list_left_i){
				if(this->exclude_gaps && current_seq[j] == A_nogap){
					continue;
				}
                k_j = A2*j + A*current_seq[j];
                couplings_gradient_i[k_j + a_ni] -= w_ni; // Case a = a_ni
                for(int a = 0; a < A_nogap; ++a){
                    couplings_gradient_i[k_j + a] += w_prob_ni[a]; // All cases
                }
            }
            for(int j : coupling_list_right_i){
				if(this->exclude_gaps && current_seq[j] == A_nogap){
					continue;
				}
                k_j = A2*j + current_seq[j];
                couplings_gradient_i[k_j + A*a_ni] -= w_ni; // Case a = a_ni
                for(int a = 0; a < A_nogap; ++a){
                   couplings_gradient_i[k_j + A*a] += w_prob_ni[a]; // All cases
                }
            }

        } // End: Loop on positions


        // Aggregate gradien and objective function ------------------------------------------------
        #pragma omp critical
        {
        
            // Aggregate object function
            loss += lossi * Neff_inv;
        
            // Aggregate Grad(h)
            for(int a = 0; a < A; ++a){
                grad[Ai + a] += fields_gradient_i[a] * Neff_inv;
            }
        
            // Aggregate Grad(J)
            int k;
            int k_2;
            int k_plus_Aa;
            int A2j;
            for(int j : coupling_list_left_i){
                k = ij_index_i[j];
                A2j = A2*j;
                for(int a = 0; a < A; ++a){
                    k_2 = A2j + A*a;
                    k_plus_Aa = k + A*a;
                    for(int b = 0; b < A; ++b){
                        grad[k_plus_Aa + b] += couplings_gradient_i[k_2 + b] * Neff_inv;
                    }
                }
            }
            for(int j : coupling_list_right_i){
                k = ij_index_i[j];
                A2j = A2*j;
                for(int a = 0; a < A; ++a){
                    k_2 = A2j + A*a;
                    k_plus_Aa = k + A*a;
                    for(int b = 0; b < A; ++b){
                        grad[k_plus_Aa + b] += couplings_gradient_i[k_2 + b] * Neff_inv;
                    }
                }
            }

        } // End: Aggregation of gradiens and objective function
    } // End: Loop on positions

    auto t2 = std::chrono::high_resolution_clock::now();
    this->dt += std::chrono::duration<float>(t2 - t1).count();

    return loss;
}


// Utils Methods -----------------------------------------------------------------------------------

// Log matrix of couplings
void PlmDCA::logCouplingMatrix(){
    int L = this->msa_length;
    for(int i = 0; i < L; ++i){
        for(int j = 0; j < L; ++j){
             std::cout << this->couplings_cutoff[i][j];
        }
        std::cout << "\n";
    }
}

// Log lists of couplings by line
void PlmDCA::logCouplingList(){
    int L = this->msa_length;
    for(int i = 0; i < L; ++i){
        std::cout << i << " : ";
        for (int j : this->coupling_list[i]) {
            std::cout << j << " ";
        }
        std::cout << "\n";
    }
}

// Log method for accumulated times
void PlmDCA::logTime(const char* log_str){
    std::cout << "    - " << log_str << " dt=" << this->dt << " sec." << std::endl;
    this->dt=0.0f;
}
