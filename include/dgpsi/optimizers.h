#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#define OPTIM_ENABLE_EIGEN_WRAPPERS

#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include <string>
#include <chrono>

#include <dgpsi/base.h>
#include <optim/optim.hpp>

extern "C" {
void setulb_wrapper(int *n, int *m, double x[], double l[], double u[], int nbd[], double *f,
                    double g[], double *factr, double *pgtol, double wa[], int iwa[], int *itask,
                    int *iprint, int *icsave, bool *lsave0, bool *lsave1, bool *lsave2, bool *lsave3,
                    int isave[], double dsave[]);
}

namespace dgpsi::optimizers {
	enum TSolver { TLBFGSB, TPSO, TCG, TRprop };

	namespace utilities {
		template<typename F>
		TVector numerical_gradient(F &functor, const TVector& X, const TVector& lb, const TVector& ub, double grid_spacing = 1e-3) {
			// check consistency of the dimensions
			int inputDimension = X.size();
			if (inputDimension != lb.size() || inputDimension != ub.size()) {
				throw std::invalid_argument("The size of x does not match the bound's dimensions");
			}
			// check x is within bounds
			for (int i = 0; i < inputDimension; i++) {
				if (X[i] > ub[i] || X[i] < lb[i]) {
					throw std::runtime_error("x is not contained within [lb, ub]");
				}
			}

			TVector grad(X);
			TVector workX(X);
			for (int i = 0; i < inputDimension; i++) {
				double effectiveGridOver = ((X[i] + grid_spacing) > ub[i]) ? ub[i] - X[i] : grid_spacing;
				workX[i] = X[i] + effectiveGridOver;
				double valueOver = functor.objective_value(workX);
				double effectiveGridBelow = ((X[i] - grid_spacing) < lb[i]) ? X[i] - lb[i] : grid_spacing;
				workX[i] = X[i] - effectiveGridBelow;
				grad[i] = (valueOver - functor.objective_value(workX)) / (effectiveGridOver + effectiveGridBelow);
				// restore original value
				workX[i] = X[i];
			}
			return grad;
		}		
		
		class Problem {
		public:
			Problem(unsigned int input_dim) : input_dim(input_dim) {}
			Problem(unsigned int input_dim, const TVector& lb, const TVector& ub) : input_dim(input_dim), lower_bound(lb), upper_bound(ub) {}
			virtual double operator()(const TVector& X, TVector& grad) = 0;
			virtual double objective_value(const TVector& X) = 0;
			virtual void approx_gradient(const TVector& X, TVector& grad)  {
				grad = numerical_gradient((*this), X, lower_bound, upper_bound);
			}
			virtual void gradient(TVector& grad) = 0;
			
			void set_lower_bound(const TVector& lb) {
				if (lb.size() != input_dim){
					throw std::runtime_error("lb.size() != input_dim");
				}
				lower_bound = lb;
			}			
			void set_upper_bound(const TVector& ub) {
				if (ub.size() != input_dim){
					throw std::runtime_error("ub.size() != input_dim");
				}				
				upper_bound = ub;
			}
			void set_bounds(const TVector& lb, const TVector& ub) {
				set_lower_bound(lb);
				set_upper_bound(ub);
			}			
			
			VectorPair get_bounds(){ 
				return std::make_pair(lower_bound, upper_bound);
			}

			const unsigned int input_dim;
			TVector Xopt;
			double fopt;
		private:
			TVector lower_bound;				
			TVector upper_bound;
		};
		
	}

	using SolverSettings = optim::algo_settings_t;	
	
	struct OptimData {};
	using OptimFxn = std::function<double(const TVector& x, TVector* grad, void* optdata)>;
	using Problem = utilities::Problem;	

	struct Solver {
		int verbosity = -1;	
		bool from_optim = false;
		Solver() = default;
		Solver(const Solver& solver) {
			verbosity = solver.verbosity;
			from_optim = solver.from_optim;
		}
		Solver(bool from_optim) : from_optim(from_optim) {}
		Solver(const int& verbosity) : verbosity(verbosity) {}
		Solver(const int& verbosity, bool from_optim) : verbosity(verbosity), from_optim(from_optim) {}
		virtual void solve(TVector& XX, Problem& problem) = 0;
		virtual void solve(TVector& theta, OptimFxn objective, OptimData optdata) = 0;
		protected:
			virtual SolverSettings settings() const { SolverSettings settings_; return settings_; }

	};

	/* MODIFIED LBFGSB WRAPPER */
	struct LBFGSB : public Solver {

		int MM; // Memory Size
		double pgtol{1e-9}; // Projected Gradient Tolerance
		unsigned int max_iter{15000};
		unsigned int max_fun{15000};
		double factr{1e7}; // Machine Precision Factor
		double gscale = 1.0; // <= 1 used to scale the gradient for explosive functions

		LBFGSB() : Solver(), MM(16) {}
		LBFGSB(const int& verbosity) : Solver(verbosity), MM(16) {}	
		void solve(TVector& theta, OptimFxn objective, OptimData optdata) override {}
		void solve(TVector& XX, Problem& problem)  override
		{
			run_check();
			int NN = problem.input_dim;
			// Setup Bounds
			VectorPair bounds = problem.get_bounds();
			std::vector<double> LB(NN);
			std::vector<double> UB(NN);
			TVector::Map(&LB[0], NN) = bounds.first;
			TVector::Map(&UB[0], NN) = bounds.second;

			std::vector<int> BND(NN);
			std::vector<int> IWA(3*NN);
			std::vector<double> WA(2 * MM * NN + 5 * NN + 11 * MM * MM + 8 * MM);

			bool hasLowerBound, hasUpperBound; 
			for (Eigen::Index i = 0; i < XX.size(); ++i){
				hasLowerBound = !std::isinf(LB[i]);
				hasUpperBound = !std::isinf(UB[i]);
				if (hasLowerBound) {
					if (hasUpperBound) {
						BND[i] = 2;
					} else {
						BND[i] = 1;
					}
				} else if (hasUpperBound) {
					BND[i] = 3;
				} else {
					BND[i] = 0;
				}
			}

			// TVector grad(XX);
			// double fobj = problem.objective_value(XX);
			// problem.approx_gradient(XX, grad);
			//
			TVector grad;
			double fobj = problem(XX, grad);
			if (gscale != 1.0) {
				scale_gradient(grad, NN);
			}	
			int i = 0;
			int itask = 0;
			int icsave = 0;
			bool test = false;					

			while ((i < max_iter) && ((itask == 0) || (itask == 1) || (itask == 2) || (itask == 3) ))	
			{
				setulb_wrapper(&NN, &MM, &XX[0], &LB[0], &UB[0], &BND[0], &fobj, 
							   &grad[0], &factr, &pgtol, &WA[0], &IWA[0], &itask, 
							   &verbosity, &icsave, &itfbool[0], &itfbool[1], &itfbool[2], &itfbool[3],
							   &itfint[0], &itfdbl[0]);
				// assert that impossible values do not occur
				assert(icsave <= 14 && icsave >= 0);
				assert(itask <= 12 && itask >= 0);

				if (itask == 2 || itask == 3) {
					// fobj = problem.objective_value(XX);
					// problem.approx_gradient(XX, grad);
					//
					fobj = problem(XX, grad);
					if (gscale != 1.0) {
						scale_gradient(grad, NN);
					}
				}
				i = itfint[29];
			}
			problem.Xopt = XX;
			problem.fopt = fobj;
		}
	
	private:
		// Interface to Fortran code
		bool itfbool[4];
		int itfint[44];
		double itfdbl[29];	

		void scale_gradient(TVector& gradient, int gradientSize) {
			for (int i = 0; i < gradientSize; i++) {
				gradient[i] *= gscale;
			}
		}	
		void run_check(){
			if (MM < 1) {
				throw std::invalid_argument("memory size (MM) should be >= 1");
			}
			if (max_iter < 1) {
				throw std::invalid_argument("max_iter should be >= 1");
			}	
			if (factr <= 0) {
				throw std::invalid_argument("factr should be > 0");
			}		
			if (pgtol < 0) {
				throw std::invalid_argument("pgtol should be >= 0");
			}	
			if (gscale <= 0 || gscale > 1) {
				throw std::invalid_argument("gscale should be > 0 and <= 1");
			}											
		}
		SolverSettings settings() const override {
			SolverSettings settings_;
			settings_.lbfgsb_settings.MM = MM;
			settings_.lbfgsb_settings.pgtol = pgtol;
			settings_.lbfgsb_settings.max_iter = max_iter;
			settings_.lbfgsb_settings.max_fun = max_fun;
			settings_.lbfgsb_settings.factr = factr;
			settings_.lbfgsb_settings.gscale = gscale;			
			return settings_;
		}
	
	};

	/* REWRITE OF BLUM & RIEDMILLER (2013) LIBGP->RPROP */
	/* REFERENCE: https://github.com/mblum/libgp        */
	struct Rprop : public Solver {
		Rprop() : Solver() {}
		void solve(TVector& theta, OptimFxn objective, OptimData optdata) override {}
		void solve(TVector& XX, Problem& problem)  override {
			TVector Delta = TVector::Ones(problem.input_dim) * Delta0;
			problem.Xopt = XX;
			TVector grad_old = TVector::Ones(problem.input_dim);

			auto sign = [](double& x) {
				if (x > 0) { return 1.0; }
				else if (x < 0) { return -1.0; }
				else { return 0.0; }
			};


			double best = log(0);
			TVector grad(XX);
			for (unsigned int i = 0; i < n_iter; ++i) {
				problem.gradient(grad);
				grad_old = grad_old.cwiseProduct(grad);
				for (int j = 0; j < grad_old.size(); ++j) {
					if (grad_old(j) > 0) {
						Delta(j) = std::min(Delta(j) * etaplus, Deltamax);
					}
					else if (grad_old(j) < 0) {
						Delta(j) = std::max(Delta(j) * etaminus, Deltamin);
						grad(j) = 0;
					}
					XX(j) += -sign(grad(j)) * Delta(j);
				}
				grad_old = grad;
				if (grad_old.norm() < eps_stop) { break; }
				double nll = problem(XX, grad);
				if (verbosity > 0) std::cout << i << " " << nll << std::endl;
				if (nll > best) {
					best = nll;
					problem.Xopt = XX;
					problem.fopt = best;
				}
			}
		}
		
		double Delta0 = 0.1;
		double Deltamin = 1e-6;
		double Deltamax = 50.0;
		double etaminus = 0.5;
		double etaplus = 1.2;
		double eps_stop = 0.0;
		unsigned int n_iter = 100;
	private:
		SolverSettings settings() const override {
			SolverSettings settings_;
			settings_.Rprop_settings.Delta0 = Delta0;
			settings_.Rprop_settings.Deltamin = Deltamin;
			settings_.Rprop_settings.Deltamax = Deltamax;
			settings_.Rprop_settings.etaminus = etaminus;
			settings_.Rprop_settings.etaplus = etaplus;
			settings_.Rprop_settings.eps_stop = eps_stop;
			settings_.Rprop_settings.n_iter = n_iter;
			return settings_;
		}
	};
	
	
	struct OptimSolver : public Solver {
		OptimSolver() : Solver(true) {}
		OptimSolver(const int& verbosity) : Solver(verbosity, true) {}

		int conv_failure_switch = 0;
		int iter_max = 2000;
		double err_tol = 1E-08;
		bool vals_bound = false;		
	};

	struct PSO : public OptimSolver {
		PSO() : OptimSolver() {}
		PSO(const int& verbosity) : OptimSolver(verbosity) {}
		void solve(TVector& theta, OptimFxn objective, OptimData optdata) override
		{
			SolverSettings settings_ = settings();
			bool success = optim::pso(theta, objective, &optdata, settings_);
		}
		void solve(TVector& XX, Problem& problem)  override {}
		bool center_particle = true;
		int n_pop = 100;
		int n_gen = 1000;
		int inertia_method = 1; // 1 for linear decreasing between w_min and w_max; 2 for dampening
		double par_initial_w = 1.0;
		double par_w_damp = 0.99;
		double par_w_min = 0.10;
		double par_w_max = 0.99;
		int velocity_method = 1; // 1 for fixed; 2 for linear
		double par_c_cog = 2.0;
		double par_c_soc = 2.0;
		double par_initial_c_cog = 2.5;
		double par_final_c_cog = 0.5;
		double par_initial_c_soc = 0.5;
		double par_final_c_soc = 2.5;
	private:
		TVector initial_lb;
		TVector initial_ub;
		SolverSettings settings() const override {
			SolverSettings settings_;
			settings_.conv_failure_switch = conv_failure_switch;
			settings_.iter_max = iter_max;
			settings_.grad_err_tol = err_tol;
			settings_.vals_bound = vals_bound;
			settings_.print_level = verbosity;
			settings_.pso_settings.center_particle = center_particle;
			settings_.pso_settings.n_pop = n_pop;
			settings_.pso_settings.n_gen = n_gen;
			settings_.pso_settings.inertia_method = inertia_method;
			settings_.pso_settings.velocity_method = velocity_method;
			settings_.pso_settings.par_initial_w = par_initial_w;
			settings_.pso_settings.par_w_damp = par_w_damp;
			settings_.pso_settings.par_w_min = par_w_min;
			settings_.pso_settings.par_w_max = par_w_max;
			settings_.pso_settings.par_c_cog = par_c_cog;
			settings_.pso_settings.par_c_soc = par_c_soc;
			settings_.pso_settings.par_initial_c_cog = par_initial_c_cog;
			settings_.pso_settings.par_final_c_cog = par_final_c_cog;
			settings_.pso_settings.par_initial_c_soc = par_initial_c_soc;
			settings_.pso_settings.par_final_c_soc = par_final_c_soc;
			if (initial_lb.size()) { settings_.pso_settings.initial_lb = initial_lb; }
			if (initial_ub.size()) { settings_.pso_settings.initial_ub = initial_ub; }
			return settings_;
		}
	};

	struct ConjugateGradient : public OptimSolver {
		ConjugateGradient() : OptimSolver() {}
		ConjugateGradient(int& verbosity) : OptimSolver(verbosity) {}
		void solve(TVector& theta, OptimFxn objective, OptimData optdata) override
		{
			SolverSettings settings_ = settings();
			bool success = optim::cg(theta, objective, &optdata, settings_);
		}		

		double restart_threshold = 0.1;

	private:
		SolverSettings settings() const override {
			SolverSettings settings_;
			settings_.conv_failure_switch = conv_failure_switch;
			settings_.iter_max = iter_max;
			settings_.grad_err_tol = err_tol;
			settings_.vals_bound = vals_bound;
			settings_.print_level = verbosity;
			settings_.cg_settings.restart_threshold = restart_threshold;
			return settings_;
		}	
	};


}
#endif


/* OPTIM SOLVERS

	struct DifferentialEvolution : public Solver {
		DifferentialEvolution() : Solver("DE") {}
		DifferentialEvolution(const int& verbosity, const int& n_restarts) :
			Solver(verbosity, n_restarts, "DE") {}
		DifferentialEvolution(const int& verbosity, const int& n_restarts, const std::string& sampling_method) :
			Solver(verbosity, n_restarts, sampling_method, "DE") {}
		DifferentialEvolution(const int& verbosity, const int& n_restarts, const std::string& sampling_method,
			const TVector& initial_lb, const TVector& initial_ub) :
			Solver(verbosity, n_restarts, sampling_method, "DE"),
			initial_lb(initial_lb), initial_ub(initial_ub) {}

		SolverSettings settings() const override {
			SolverSettings settings_;
			settings_.conv_failure_switch = conv_failure_switch;
			settings_.iter_max = iter_max;
			settings_.grad_err_tol = err_tol;
			settings_.vals_bound = vals_bound;
			settings_.print_level = verbosity;
			settings_.de_settings.n_pop = n_pop;
			settings_.de_settings.n_pop_best = n_pop_best;
			settings_.de_settings.n_gen = n_gen;
			settings_.de_settings.pmax = pmax;
			settings_.de_settings.max_fn_eval = max_fn_eval;
			settings_.de_settings.mutation_method = mutation_method;
			settings_.de_settings.check_freq = check_freq;
			settings_.de_settings.par_F = par_F;
			settings_.de_settings.par_CR = par_CR;
			settings_.de_settings.par_F_l = par_F_l;
			settings_.de_settings.par_F_u = par_F_u;
			settings_.de_settings.par_tau_F = par_tau_F;
			settings_.de_settings.par_tau_CR = par_tau_CR;
			if (initial_lb.size()) { settings_.de_settings.initial_lb = initial_lb; }
			if (initial_ub.size()) { settings_.de_settings.initial_ub = initial_ub; }
			return settings_;
		}
		bool solve
		(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
		{
			return optim::de(theta, objective, &optdata, settings);
		}

		int n_pop = 200;
		int n_pop_best = 6;
		int n_gen = 100;
		int pmax = 4;
		int max_fn_eval = 100000;
		int mutation_method = 1; // 1 = rand; 2 = best
		int check_freq = -1;
		double par_F = 0.8;
		double par_CR = 0.9;
		double par_F_l = 0.1;
		double par_F_u = 1.0;
		double par_tau_F = 0.1;
		double par_tau_CR = 0.1;
		TVector initial_lb; // this will default to -0.5
		TVector initial_ub; // this will default to  0.5
	};

	struct GradientDescent : public Solver {
		GradientDescent(const int& method) : Solver("GD"), method(method) {}
		GradientDescent(const int& method, const int& verbosity, const int& n_restarts) :
			Solver(verbosity, n_restarts, "GD"), method(method) {}
		GradientDescent(const int& method, const int& verbosity, const int& n_restarts, const std::string& sampling_method) :
			Solver(verbosity, n_restarts, sampling_method, "GD"), method(method) {}

		SolverSettings settings() const override {
			SolverSettings settings_;
			settings_.conv_failure_switch = conv_failure_switch;
			settings_.iter_max = iter_max;
			settings_.grad_err_tol = err_tol;
			settings_.vals_bound = vals_bound;
			settings_.print_level = verbosity;
			settings_.gd_settings.method = method;
			settings_.gd_settings.par_step_size = step_size;
			settings_.gd_settings.step_decay = step_decay;
			settings_.gd_settings.step_decay_periods = step_decay_periods;
			settings_.gd_settings.step_decay_val = step_decay_val;
			settings_.gd_settings.par_momentum = momentum;
			settings_.gd_settings.par_ada_norm_term = ada_norm;
			settings_.gd_settings.ada_max = ada_max;
			settings_.gd_settings.par_adam_beta_1 = adam_beta_1;
			settings_.gd_settings.par_adam_beta_2 = adam_beta_2;
			return settings_;
		}
		bool solve
		(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
		{
			return optim::gd(theta, objective, &optdata, settings);
		}

		// GD method
		int method;
		// step size, or 'learning rate'
		double step_size = 0.1;
		// decay
		bool step_decay = false;
		optim::uint_t step_decay_periods = 10;
		double step_decay_val = 0.5;
		// momentum parameter
		double momentum = 0.9;
		// Ada parameters
		double ada_norm = 10e-08;
		double ada_rho = 0.9;
		bool ada_max = false;
		// Adam parameters
		double adam_beta_1 = 0.9;
		double adam_beta_2 = 0.999;

	};

	struct ConjugateGradient : public Solver {
		ConjugateGradient() : Solver("CG") {}
		ConjugateGradient(int& verbosity) :
			Solver(verbosity, n_restarts, "CG") {}
		ConjugateGradient(int& verbosity, int& n_restarts, std::string& sampling_method) :
			Solver(verbosity, n_restarts, sampling_method, "CG") {}

		SolverSettings settings() const override {
			SolverSettings settings_;
			settings_.conv_failure_switch = conv_failure_switch;
			settings_.iter_max = iter_max;
			settings_.grad_err_tol = err_tol;
			settings_.vals_bound = vals_bound;
			settings_.print_level = verbosity;
			settings_.cg_settings.restart_threshold = restart_threshold;
			return settings_;
		}
		bool solve
		(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
		{
			return optim::cg(theta, objective, &optdata, settings);
		}
		double restart_threshold = 0.1;
	};

	struct NelderMead : public Solver {
		NelderMead() : Solver("NM") {}
		NelderMead(int& verbosity, int& n_restarts) :
			Solver(verbosity, n_restarts, "NM") {}
		NelderMead(int& verbosity, int& n_restarts, std::string& sampling_method) :
			Solver(verbosity, n_restarts, sampling_method, "NM") {}

		SolverSettings settings() const override {
			SolverSettings settings_;
			settings_.conv_failure_switch = conv_failure_switch;
			settings_.iter_max = iter_max;
			settings_.grad_err_tol = err_tol;
			settings_.vals_bound = vals_bound;
			settings_.print_level = verbosity;
			settings_.nm_settings.adaptive_pars = adaptive_pars;
			settings_.nm_settings.par_alpha = par_alpha;
			settings_.nm_settings.par_beta = par_beta;
			settings_.nm_settings.par_gamma = par_gamma;
			settings_.nm_settings.par_delta = par_delta;
			return settings_;
		}

		bool solve
		(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
		{
			return optim::nm(theta, objective, &optdata, settings);
		}

		bool adaptive_pars = true;
		double par_alpha = 1.0; // reflection parameter
		double par_beta = 0.5; // contraction parameter
		double par_gamma = 2.0; // expansion parameter
		double par_delta = 0.5; // shrinkage parameter
	};


*/