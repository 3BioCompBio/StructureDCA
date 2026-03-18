#include "include/plmdca.h"

/*
[Sparse plmDCA solver in C++]
This part is responsible for:
    - The bridge between Python and C++: it provides C++ functions to communicate with Python
    - The Gradient Descent-based optimizer
Author: Matsvei Tsishyn
*/

static std::ios_base::Init _force_iostream_init; // force global iostream (std::cout etc.) initialization

// Optimized for h and J: Gradient Descent on plm of the DCA model -------------

class ObjectiveFunction{
    /*Objective Function for lbfgs input.     
    Attributes
        hJ               : A dynamic array containing fields and couplings.
        plmdca_inst      : PlmDCA 
        verbose          : bool 
        max_iterations   : int 
    */
   
    // Constructor -------------------------------------------------------------
    protected:
        float* hJ;
        const bool verbose;
        const bool log_gd_steps;
        const int max_iterations;
        int n_iterations;
        PlmDCA plmdca_inst;
        
    public:
        ObjectiveFunction(
            const short n_states, 
            const char* msa_path,
            const int msa_length,
            const bool* couplings_cutoff_flat,
            const float lambda_h, 
            const float lambda_J,
            const float lambda_asymptotic,
            const bool exclude_gaps,
            const float theta_regularization,
            const bool count_target_sequence,
            const bool use_weights,
            const float weights_seqid,
			const float* pos_weights,
            const int m_max_iterations,
            const int num_threads,
            const char* weights_cache_path,
            const bool m_verbose,
            const bool m_log_gd_steps
        ):
            hJ(NULL), 
            verbose(m_verbose),
            log_gd_steps(m_log_gd_steps),
            max_iterations(m_max_iterations),
            plmdca_inst(
                n_states,
                msa_path,
                msa_length,
                couplings_cutoff_flat,
                lambda_h,
                lambda_J,
                lambda_asymptotic,
                exclude_gaps,
                theta_regularization,
                count_target_sequence,
                use_weights,
                weights_seqid,
				pos_weights,
                num_threads,
                weights_cache_path,
                verbose
            )
        {
            // Constructor
        }        

        // Main Methods --------------------------------------------------------

        // Return h and J array
        float* getFieldsAndCouplings(){
            return this->hJ;
        }

        // Run Optimized
        int run(){
            /*Performs plmDCA computation using LBFGS optimization.
            Returns
                ret   : Exit status of LBFGS optimization.
            */

            // Init
            float loss;
            int n_hJ = this->plmdca_inst.n_hJ;
            this->hJ = lbfgs_malloc(n_hJ);
            this->n_iterations = 0;

            // Memory allocation error
            if (this->hJ == NULL) {
                std::cerr << "ERROR in StructureDCA(): DCASolver.PlmDCA C++ backend." << std::endl;
                std::cerr << " -> Failed to allocate a memory block for DCA coefficients h and J. (maybe out of RAM ?)" << std::endl;
                std::abort();  // Force the code to crash at runtime
                return 1; // Return 1 just for style
            }

            // Initialize parameters
            lbfgs_parameter_t param;
            lbfgs_parameter_init(&param);
            param.epsilon = 1E-6;
            param.max_iterations = this->max_iterations;
            param.max_linesearch = 5;
            param.ftol = 1E-4;
            //param.wolfe = 0.2;
            param.m = 5 ;

            // Init hJ coeff
            this->plmdca_inst.initCoefficients(hJ);

            // Skip GD optimization if max_iterations is zero
            if(this->verbose && this->max_iterations == 0){
                std::cout << " * plmDCA WARNING: max_iterations=0, so no h, J parameters optimization was perfomed" << std::endl;
                return 0;
            }
            
            // Optimize
            if(this->verbose) {
                std::cout << "   - plmDCA: solve DCA model (h and J coefficients)" << std::endl;
            }
            int ret = lbfgs(n_hJ, hJ, &loss, _evaluate, _progress, this, &param);
            if(this->verbose){
                if (ret == -1001){ // return status value ret == -1001 corresponds with convergence for a given precision
                    std::cout << " * plmDCA: L-BFGS optimization completed (" << this->n_iterations << " / " << this->max_iterations << " GD-iterations)" << std::endl;
                }else{
                    std::cout << "L-BFGS optimization terminated with status code: " << ret << std::endl;
                    std::cout << " -> max_iterations for GD reached: " << this->max_iterations << std::endl;
                    std::cout << " -> loss: " << loss << std::endl;
                }
            }

            // Log total accumulated time after DCA is solved
            if(this->verbose){
                plmdca_inst.logTime("plmDCA GD time: ");
            }

            return ret;
        }

        float getNeff(){
            return this->plmdca_inst.Neff;
        }

    // Dependencies ------------------------------------------------------------
    protected:
        static float _evaluate(void* instance, const float*hJ, float* grad, const int n_hJ, const float step)
        {
            /*Computes the gradient of the regularized negative pseudolikelihood function for the alignment
            Parameters
                instance     : An instance of ObjectiveFunction class. 
                hJ           : Array of fields and couplings.
                grad         : Array of gradient of the negative log pseudolikelihood of the conditional probablity
                n_hJ         : Number of fields and couplings?
                step         : The step size for gradient descent.
            Returns
                loss         : Value of plmDCA objective function (Loss Function / Error)
            */
            auto* obj = static_cast<ObjectiveFunction*>(instance);
            auto loss = obj->plmdca_inst.gradient(hJ, grad);
            return loss;
        }

        static int _progress(
            void* instance,
            const float* hJ,
            const float* grad,
            const float loss, 
            const float xnorm,
            const float gnorm,
            const float step,
            const int n_hJ,
            int k,
            int ls
        )
        {
            /*Function to run after each iteration of GD.*/
            auto* obj = static_cast<ObjectiveFunction*>(instance);
            if(obj->log_gd_steps){
                fprintf(stdout, "      * Iteration %d: loss = %f, xnorm = %f, gnorm = %f, step = %fn_hJ \n", k, loss, xnorm, gnorm, step);
            }
            obj->n_iterations ++;
            return 0;
        }
};
 

// Main ------------------------------------------------------------------------
// These functions are exported to be used by Python
extern "C" float* plmdcaBackend(
    const short n_states,
    const char* msa_path,
    const int msa_length,
    const bool* couplings_cutoff_flat,
    const float lambda_h, 
    const float lambda_J,
    const float lambda_asymptotic,
    const bool exclude_gaps,
    const float theta_regularization,
    const bool count_target_sequence,
    const bool use_weights,
    const float weights_seqid,
	const float* pos_weights,
    const int max_iterations,
    const int num_threads,
    const char* weights_cache_path,
    const bool verbose,
    const bool log_gd_steps,
    const char* neff_tmp_path
)
{  

    // Init h and J Optimized
    ObjectiveFunction objective_function(
        n_states,
        msa_path,
        msa_length,
        couplings_cutoff_flat,
        lambda_h,
        lambda_J,
        lambda_asymptotic,
        exclude_gaps,
        theta_regularization,
        count_target_sequence,
        use_weights,
        weights_seqid,
		pos_weights,
        max_iterations,
        num_threads,
        weights_cache_path,
        verbose,
        log_gd_steps
    );

    // Solve h and h and J
    objective_function.run();
    auto h_and_J = objective_function.getFieldsAndCouplings();

    // Save Neff in tmp file
    const float Neff = objective_function.getNeff();
    std::ofstream neff_tmp_file;
    neff_tmp_file.open(neff_tmp_path);
    neff_tmp_file << Neff;
    neff_tmp_file.close();

    return h_and_J;
}


extern "C" void freeFieldsAndCouplings(void* h_and_J)
{  
    /*Frees memory that has been used to store fields and couplings before they are captured in the Python interface.
        - h_and_J : Pointer to the fields and couplings vector 
    */
    float* h_and_J_casted = static_cast<float*>(h_and_J);  
    if(h_and_J_casted !=nullptr){
        delete [] h_and_J_casted;
        h_and_J_casted = nullptr;
    }
}
