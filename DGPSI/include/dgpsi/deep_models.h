#ifndef DEEPMODELS_H
#define DEEPMODELS_H
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#include <dgpsi/utilities.h>
#include <dgpsi/kernels.h>
#include <dgpsi/base_models.h>
#include <chrono>
#include <dgpsi/optimizers.h>

#include <chrono>

namespace dgpsi::deep_models {
	using KernelPtr = std::shared_ptr<Kernel>;
	using SolverPtr = std::shared_ptr<Solver>;
	enum State { Input, InputConnected, Hidden, Observed, Unchanged };
	enum Task { Init, Train, LinkedPredict };
	enum LLF { Gaussian, Heteroskedastic };

	using dgpsi::optimizers::LBFGSB;
	using dgpsi::optimizers::PSO;
	using dgpsi::optimizers::ConjugateGradient;
	using dgpsi::optimizers::Rprop;

	struct Likelihood {

		Likelihood() = default;
		Likelihood(const LLF& likelihood) : likelihood(likelihood) {}

		void log_likelihood() {

		}

		TMatrix	 X;
		TMatrix	 Y;
		TVector  alpha;
		TLLT	 chol;
		TMatrix	 K;

		LLF likelihood = LLF::Gaussian;

	};



	namespace gaussian_process {
		using namespace dgpsi::kernels;
		using namespace dgpsi::utilities;
		using namespace dgpsi::base_models;
		using namespace dgpsi::base_models::gaussian_process;

		class Node : public GP {
		private:
			Node& evalK(bool with_scale = true) {
				K = kernel->K(inputs, inputs, likelihood_variance.value());
				if (with_scale) K.array() *= scale.value();
				return *this;
			}
			Node& evalK(const TMatrix& Xpad, bool with_scale = true) {
				TMatrix tmp(inputs.rows(), inputs.cols() + Xpad.cols());
				tmp << inputs, Xpad;
				K.noalias() = kernel->K(tmp, tmp, likelihood_variance.value());
				if (with_scale) K.array() *= scale.value();
				return *this;
			}
			TMatrix sample_mvn() {
				TVector mean = TVector::Zero(K.rows());
				Eigen::setNbThreads(1);
				Eigen::SelfAdjointEigenSolver<TMatrix> eigenSolver(K);
				//TMatrix transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
				TMatrix transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
				std::normal_distribution<> dist;
				return mean + transform * TVector{ mean.size() }.unaryExpr([&](auto x) {return dist(rng); });
			}
			TMatrix sample_mvn(const TVector& mean, const TMatrix& cov) {
				Eigen::setNbThreads(1);
				Eigen::SelfAdjointEigenSolver<TMatrix> eigenSolver(cov);
				TMatrix transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
				std::normal_distribution<> dist;
				return mean + transform * TVector{ mean.size() }.unaryExpr([&](auto x) {return dist(rng); });
			}
			TVector gradients() override {
				auto log_prior_gradient = [=]() {
					// Gamma Distribution
					// self.gfod = lambda x : (self.prior_coef[0] - 1) - self.prior_coef[1] * x
					const double shape = 1.6;
					const double rate = 0.3;
					TVector lpg;
					if (!(*kernel->length_scale.is_fixed)) {
						lpg = (shape - 1.0) - (rate * kernel->length_scale.value().array()).array();
					}
					if (!(*likelihood_variance.is_fixed)) {
						lpg.conservativeResize(lpg.size() + 1);
						lpg.tail(1)(0) = (shape - 1.0) - (rate * likelihood_variance.value());
					}
					return lpg;
				};
				std::vector<TMatrix> grad_;
				kernel->gradients(inputs, grad_);
				TVector grad = TVector::Zero(grad_.size());
				if (!(*likelihood_variance.is_fixed))
					grad_.push_back(likelihood_variance.value() * TMatrix::Identity(inputs.rows(), inputs.rows()));
				for (int i = 0; i < grad.size(); ++i) {
					TMatrix KKT = chol.solve(grad_[i]);
					double trace = KKT.trace();
					double YKKT = (outputs.transpose() * KKT * alpha).coeff(0);
					double P1 = -0.5 * trace;
					double P2 = 0.5 * YKKT;
					grad[i] = -P1 - (P2 / scale.value());
				}
				grad -= log_prior_gradient();
				return grad;
			}
			TVector get_params() override {
				TVector params = kernel->get_params();
				if (!(*likelihood_variance.is_fixed)) {
					likelihood_variance.transform_value();
					params.conservativeResize(params.rows() + 1);
					params.tail(1)(0) = likelihood_variance.value();
				}
				return params;
			}
			double  log_likelihood() {
				switch (likelihood) {
				case Heteroskedastic:
				{
					if (inputs.cols() != 2) throw std::runtime_error("Heteroskedastic GP requires 2D inputs");
					TMatrix mu = inputs.col(0);
					TMatrix var = exp(inputs.col(1).array());
					double ll = (-0.5 * (log(2 * PI * var.array()) + pow((outputs - mu).array(), 2) / var.array())).sum();
					return ll;
				}
				default:
				{
					chol = K.llt();
					double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
					double quad = (outputs.array() * (chol.solve(outputs)).array()).sum();
					double ll = -0.5 * (logdet + quad);
					return ll;
				}
				}

			}
			double  log_marginal_likelihood() override {
				double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
				double YKinvY = (outputs.transpose() * alpha)(0);
				double NLL = 0.0;
				if (*scale.is_fixed) NLL = 0.5 * (logdet + (YKinvY / scale.value()));
				else NLL = 0.5 * (logdet + (inputs.rows() * log(scale.value())));
				NLL -= log_prior();
				return NLL;
			}
			double  log_prior() {
				// Gamma Distribution
				// self.g = lambda x : (self.prior_coef[0] - 1) * np.log(x) - self.prior_coef[1] * x			
				const double shape = 1.6;
				const double rate = 0.3;
				double lp = 0.0;
				if (!(*kernel->length_scale.is_fixed)) {
					lp += (((shape - 1.0) * log(kernel->length_scale.value().array())) - (rate * kernel->length_scale.value().array())).sum();
				}
				if (!(*likelihood_variance.is_fixed)) {
					lp += ((shape - 1.0) * log(likelihood_variance.value())) - (rate * likelihood_variance.value());
				}
				return lp;
			}
			void    set_params(const TVector& new_params) override
			{
				kernel->set_params(new_params);
				if (!(*likelihood_variance.is_fixed)) likelihood_variance.transform_value(new_params.tail(1)(0));
				update_cholesky();
			}
			void    get_bounds(TVector& lower, TVector& upper) {
				kernel->get_bounds(lower, upper);
				if (!(*likelihood_variance.is_fixed)) {
					std::pair<double, double> llvb = likelihood_variance.get_bounds();
					double lb = llvb.first;
					double ub = llvb.second;
					if (likelihood_variance.get_transform() == "logexp") {
						if (std::isinf(lb)) lb = -23.025850929940457;
						if (std::isinf(ub)) ub = 0.0;
					}
					lower.conservativeResize(lower.rows() + 1);
					upper.conservativeResize(upper.rows() + 1);
					lower.tail(1)(0) = lb;
					upper.tail(1)(0) = ub;
				}
			}
			void    train() override {
				if (likelihood != LLF::Gaussian) return;
				TVector lower_bound, upper_bound;
				get_bounds(lower_bound, upper_bound);
				TVector theta = get_params();
				if (solver->from_optim) {
					auto objective = [this](const TVector& x, TVector* grad, void* opt_data)
					{return objective_(x, grad, nullptr, opt_data); };
					opt::OptimData optdata;
					solver->solve(theta, objective, optdata);
				}
				else {
					// LBFGSB/ Rprop
					Objective objective(this, static_cast<int>(lower_bound.size()));
					objective.set_bounds(lower_bound, upper_bound);
					solver->solve(theta, objective);
					theta = objective.Xopt;
				}
				set_params(theta);
				if (store_parameters) {
					TVector params(theta.size() + 1);
					params << theta, scale.value();
					history.push_back(params);
				}
			}
			// GP Predict
			void    predict(const TMatrix& X, Eigen::Ref<TVector> latent_mu, Eigen::Ref<TVector> latent_var) {
				update_cholesky();
				TMatrix Ks = kernel->K(inputs, X);
				TMatrix Kss = kernel->diag(X);
				TMatrix V = chol.solve(Ks);
				latent_mu = Ks.transpose() * alpha;
				latent_var = abs((scale.value() * (Kss - (Ks.transpose() * V).diagonal()).array()));
			}
			// Linked Predict (Default)
			void    predict(const MatrixPair& linked, Eigen::Ref<TVector> latent_mu, Eigen::Ref<TVector> latent_var) {

				update_cholesky();
				switch (likelihood) {
				case Heteroskedastic:
				{
					/*
					* y_mean = m[:,0]
					* y_var = np.exp(m[:,1]+v[:,1]/2)+v[:,0]
					* return y_mean.flatten(),y_var.flatten()
					*/
					latent_mu = linked.first.col(0);
					latent_var = exp((linked.first.col(1) + (linked.second.col(1) / 2)).array()) + linked.second.col(0).array();
				}
				default:
				{
					const Eigen::Index nrows = linked.first.rows();
					kernel->expectations(linked.first, linked.second);
					if (n_thread == 1) {
						for (Eigen::Index i = 0; i < nrows; ++i) {
							TMatrix I = TMatrix::Ones(inputs.rows(), 1);
							TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
							kernel->IJ(I, J, linked.first.row(i), linked.second.row(i), inputs, i);
							double trace = (K.llt().solve(J)).trace();
							double Ialpha = (I.cwiseProduct(alpha)).array().sum();
							latent_mu[i] = (Ialpha);
							latent_var[i] = (abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum() - (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
						}
					}
					else {
						Eigen::initParallel(); // /openmp (MSVC) or -fopenmp (GCC) flag
						thread_pool pool;
						int split = int(nrows / n_thread);
						const int remainder = int(nrows) % n_thread;
						auto task = [=, &latent_mu, &latent_var](int begin, int end)
						{
							for (Eigen::Index i = begin; i < end; ++i) {
								TMatrix I = TMatrix::Ones(inputs.rows(), 1);
								TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
								kernel->IJ(I, J, linked.first.row(i), linked.second.row(i), inputs, i);
								double trace = (K.llt().solve(J)).trace();
								double Ialpha = (I.cwiseProduct(alpha)).array().sum();
								latent_mu[i] = (Ialpha);
								latent_var[i] = (abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum() - (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
							}
						};
						for (int s = 0; s < n_thread; ++s) {
							pool.push_task(task, int(s * split), int(s * split) + split);
						}
						pool.wait_for_tasks();
						if (remainder > 0) {
							task(nrows - remainder, nrows);
						}
						pool.reset();
					}

				}
				}


			}
			// Linked Predict (InputConnected)
			void    predict(const MatrixPair& XX, const MatrixPair& linked, Eigen::Ref<TVector> latent_mu, Eigen::Ref<TVector> latent_var) {
				TMatrix X_train = XX.first;
				TMatrix X_test = XX.second;
				evalK(X_train, false);
				chol = K.llt();
				alpha = chol.solve(outputs);
				const Eigen::Index nrows = linked.first.rows();
				kernel->expectations(linked.first, linked.second);
				if (n_thread == 1) {
					for (Eigen::Index i = 0; i < nrows; ++i) {
						TMatrix I = TMatrix::Ones(inputs.rows(), 1);
						TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
						kernel->IJ(I, J, linked.first.row(i), linked.second.row(i), inputs, i);
						TMatrix Iz = kernel->K(X_train, X_test.row(i));
						TMatrix Jz = Iz * Iz.transpose();
						I.array() *= Iz.array(); J.array() *= Jz.array();
						double trace = (K.llt().solve(J)).trace();
						double Ialpha = (I.cwiseProduct(alpha)).array().sum();
						latent_mu[i] = (Ialpha);
						latent_var[i] = (abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum() - (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
					}
				}
				else {
					Eigen::initParallel(); // /openmp (MSVC) or -fopenmp (GCC) flag
					thread_pool pool;
					int split = int(nrows / n_thread);
					const int remainder = int(nrows) % n_thread;
					auto task = [=, &X_train, &X_test, &latent_mu, &latent_var](int begin, int end)
					{
						for (Eigen::Index i = begin; i < end; ++i) {
							TMatrix I = TMatrix::Ones(inputs.rows(), 1);
							TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
							kernel->IJ(I, J, linked.first.row(i), linked.second.row(i), inputs, i);
							TMatrix Iz = kernel->K(X_train, X_test.row(i));
							TMatrix Jz = Iz * Iz.transpose();
							I.array() *= Iz.array(); J.array() *= Jz.array();
							double trace = (K.llt().solve(J)).trace();
							double Ialpha = (I.cwiseProduct(alpha)).array().sum();
							latent_mu[i] = (Ialpha);
							latent_var[i] = (abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum() - (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
						}
					};
					for (int s = 0; s < n_thread; ++s) {
						pool.push_task(task, int(s * split), int(s * split) + split);
					}
					pool.wait_for_tasks();
					if (remainder > 0) {
						task(nrows - remainder, nrows);
					}
					pool.reset();
				}

			}
			// Non-Gaussian Likelihoods
			TMatrix	posterior(const TMatrix& K_prev) {
				// Zero Mean posterior
				TMatrix gamma = exp(inputs.col(1).array()).matrix().asDiagonal();
				TVector tmp1 = (gamma + K_prev).fullPivLu().solve(outputs);
				TVector mean = (K_prev.array().colwise() * tmp1.array()).rowwise().sum().matrix();
				TMatrix cov = K_prev * (gamma + K_prev).fullPivLu().solve(gamma);
				return sample_mvn(mean, cov);
			}

		public:
			Node(double likelihood_variance = 1E-6) : GP(likelihood_variance) {}
			void set_kernel(const KernelPtr& rkernel) {
				kernel = std::move(rkernel);
			}
			void set_solver(const SolverPtr& rsolver) {
				solver = std::move(rsolver);
			}
			void set_likelihood_variance(const double& lv) {
				if (lv <= 0.0) likelihood_variance.transform_value(lv);
				else likelihood_variance = lv;
			}
		private:
			friend class Layer;
			friend class DGPSI;
			unsigned int n_thread = 1;
			double objective_(const TVector& x, TVector* grad, TVector* hess, void* opt_data) {
				set_params(x);
				if (grad) { (*grad) = gradients() * 1.0; }
				return log_marginal_likelihood();
			}
			TMatrix get_parameter_history() {
				if (history.size() == 0) throw std::runtime_error("No Parameters Saved, set store_parameters = true");
				Eigen::Index param_size = get_params().size() + 1;
				TMatrix h(history.size(), param_size);
				for (std::vector<TVector>::size_type i = 0; i != history.size(); ++i) {
					h.row(i) = history[i];
				}
				return h;
			}
			void    update_cholesky() {
				K = kernel->K(inputs, inputs, likelihood_variance.value());
				chol = K.llt();
				alpha = chol.solve(outputs);
				//scale is not considered a variable in optimization, it is directly linked to chol
				if (*scale.is_fixed) return;
				else scale = (outputs.transpose() * alpha)(0) / outputs.rows();
			}
		public:
			LLF  likelihood = LLF::Gaussian;
			bool store_parameters = true;
			std::vector<TVector> history;
			Parameter<double> scale = { "scale", 1.0, "none" };

		private:
			State    cstate;
			TVector  alpha;
			TLLT	 chol;
			TMatrix	 K;
		};

		class Layer {
		public:
			Layer() = default;
			Layer(const State& layer_state, const unsigned int& n_nodes) : cstate(layer_state), n_nodes(n_nodes) {
				m_nodes.resize(n_nodes);
				for (unsigned int nn = 0; nn < n_nodes; ++nn) {
					m_nodes[nn] = Node();
				}
			}

			void set_input(const TMatrix& input) {
				if (cstate == State::Input) {
					observed_input.noalias() = input;
					set_output(input);
				}
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					nn->inputs = input;
					nn->cstate = cstate;
					if (ARD) {
						Eigen::Index ndim = input.cols();
						double val = nn->kernel->length_scale.value()(0);
						TVector new_ls = TVector::Constant(ndim, val);
						nn->kernel->length_scale = new_ls;
					}
				}
			}
			void set_input(const TMatrix& input, const Eigen::Index& col) {
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					nn->inputs.col(col) = input;
				}

			}
			void set_output(const TMatrix& output) {

				if (!locked) {
					if (cstate == State::Hidden)
					{
						ostate = State::Observed;
						std::swap(cstate, ostate);
					}
				}

				if (cstate == State::Observed)
					observed_output.noalias() = output;
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					nn->outputs = output.col(nn - m_nodes.begin());
					nn->cstate = cstate;
				}
			}

			TMatrix get_input() {
				if (cstate == State::Input) return observed_input;
				else return m_nodes[0].inputs;
			}
			TMatrix get_output() {
				if (cstate == State::Observed) return observed_output;
				else {
					TMatrix output(m_nodes[0].outputs.rows(), static_cast<Eigen::Index>(n_nodes));
					for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
						output.col(nn - m_nodes.begin()) = nn->outputs;
					}
					return output;
				}
			}

			void set_kernels(const TKernel& kernel) {
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					switch (kernel) {
					case TSquaredExponential:
						nn->set_kernel(KernelPtr(new SquaredExponential));
						continue;
					case TMatern52:
						nn->set_kernel(KernelPtr(new Matern52));
						continue;
					default:
						nn->set_kernel(KernelPtr(new SquaredExponential));
						continue;
					}
				}
			}
			void set_kernels(const TKernel& kernel, TVector& length_scale) {
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					switch (kernel) {
					case TSquaredExponential:
						nn->set_kernel(KernelPtr(new SquaredExponential(length_scale)));
						continue;
					case TMatern52:
						nn->set_kernel(KernelPtr(new Matern52(length_scale)));
						continue;
					default:
						nn->set_kernel(KernelPtr(new SquaredExponential(length_scale)));
						continue;
					}
				}
			}
			void set_solvers(const TSolver& solver) {
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					switch (solver) {
					case TLBFGSB:
						nn->set_solver(SolverPtr(new LBFGSB));
						continue;
					case TPSO:
						nn->set_solver(SolverPtr(new PSO));
						continue;
						//case TCG:
						//	nn->set_solver(SolverPtr(new ConjugateGradient));
						//	continue;
					case TRprop:
						nn->set_solver(SolverPtr(new Rprop));
						continue;
					}
				}
			}
			void set_likelihood(const LLF& likelihood) {
				this->likelihood = likelihood;
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					nn->likelihood = likelihood;
				}
			}

			void set_likelihood_variance(const double& value) {
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					nn->set_likelihood_variance(value);
				}
			}
			void fix_likelihood_variance() {
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					nn->likelihood_variance.fix();
				}
			}
			void fix_scale() {
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					nn->scale.fix();
				}
			}
			//
			void add_node(const Node& node) {
				if (locked) throw std::runtime_error("Layer Locked");
				std::vector<Node> intrm(n_nodes + 1);
				intrm.insert(intrm.end(),
					std::make_move_iterator(m_nodes.begin() + n_nodes),
					std::make_move_iterator(m_nodes.end()));
				intrm.back() = node;
				m_nodes = intrm;
				n_nodes += 1;
			}
			void remove_nodes(const unsigned int& xnodes) {
				if (locked) throw std::runtime_error("Layer Locked");
				if (xnodes > n_nodes) throw std::runtime_error("xnodes > n_nodes");
				m_nodes.erase(m_nodes.begin(), m_nodes.begin() + xnodes);
				n_nodes -= xnodes;
			}
			//
		private:
			void train() {
				for (std::vector<Node>::iterator node = m_nodes.begin(); node != m_nodes.end(); ++node) {
					if (!node->store_parameters) { node->store_parameters = true; }
					node->train();
				}
			}
			void predict(const TMatrix& X, bool store = false) {
				if (store) Xtmp.noalias() = X;
				latent_output = std::make_pair(
					TMatrix::Zero(X.rows(), static_cast<Eigen::Index>(n_nodes)),
					TMatrix::Zero(X.rows(), static_cast<Eigen::Index>(n_nodes)));

				for (std::vector<Node>::iterator node = m_nodes.begin(); node != m_nodes.end(); ++node) {
					Eigen::Index cc = static_cast<Eigen::Index>(node - m_nodes.begin());
					node->predict(X, latent_output.first.col(cc), latent_output.second.col(cc));
				}
			}
			void predict(const MatrixPair& linked) {
				latent_output = std::make_pair(
					TMatrix::Zero(linked.first.rows(), static_cast<Eigen::Index>(n_nodes)),
					TMatrix::Zero(linked.first.rows(), static_cast<Eigen::Index>(n_nodes)));
				for (std::vector<Node>::iterator node = m_nodes.begin(); node != m_nodes.end(); ++node) {
					Eigen::Index cc = static_cast<Eigen::Index>(node - m_nodes.begin());
					node->n_thread = n_thread;
					if (cstate == State::InputConnected)
						node->predict(std::make_pair(observed_input, Xtmp),
							linked, latent_output.first.col(cc),
							latent_output.second.col(cc));
					else
						node->predict(linked,
							latent_output.first.col(cc),
							latent_output.second.col(cc));
				}
			}
			void connect(const TMatrix& Ginput) {
				if (locked) throw std::runtime_error("Layer Locked");
				if (cstate != State::InputConnected)
				{
					ostate = State::InputConnected;
					std::swap(cstate, ostate);
				}
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					if (nn->kernel->length_scale.value().size() > 1) throw std::runtime_error("Input Connections Only Available with Non ARD");
					nn->cstate = cstate;
				}
				observed_input = Ginput;
			}
			Layer& evalK(bool with_scale = true) {
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					if (cstate == State::InputConnected) nn->evalK(observed_input, with_scale);
					else nn->evalK(with_scale);
				}
				return *this;
			}
			double log_likelihood() {
				double ll = 0.0;
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					ll += nn->log_likelihood();
				}
				return ll;
			}
			double log_likelihood(const TMatrix& input, const Eigen::Index& col) {
				double ll = 0.0;
				for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
					nn->inputs.col(col) = input;
					nn->evalK();
					ll += nn->log_likelihood();
				}
				return ll;
			}
			void estimate_parameters(const Eigen::Index& n_burn) {
				for (std::vector<Node>::iterator node = m_nodes.begin(); node != m_nodes.end(); ++node) {
					if (node->likelihood != LLF::Gaussian) continue;
					TMatrix history = node->get_parameter_history();
					TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
					if (*(node->scale.is_fixed)) node->scale.unfix();
					node->scale = theta.tail(1)(0);
					node->scale.fix();
					TVector tmp = theta.head(theta.size() - 1);
					node->set_params(tmp);
				}
			}
		public:
			State cstate; // Current Layer State
			unsigned int n_nodes;
			bool ARD = false;
		private:
			LLF likelihood = LLF::Gaussian;
			std::vector<Node> m_nodes;
			State ostate = State::Unchanged; // Old Layer State;
			bool locked = false;
			unsigned int n_thread = 1;
			TMatrix Xtmp;
			TMatrix observed_input;
			TMatrix observed_output;
			MatrixPair latent_output;
			friend struct Graph;
			friend class DGPSI;
		};

		struct Graph {
			unsigned int n_thread = 1;
			unsigned int n_hidden;
			const int n_layers;

			Graph(const MatrixPair& data, int n_hidden) :
				n_hidden(n_hidden), n_layers(n_hidden + 2)
			{
				unsigned int n_nodes = static_cast<unsigned int>(data.first.cols());
				m_layers.resize(n_layers);
				for (unsigned int ll = 0; ll < n_layers; ++ll) {
					if (ll == 0) {
						m_layers[ll] = Layer(State::Input, n_nodes);
						m_layers[ll].set_input(data.first);
					}
					else if (ll == n_layers - 1) {
						m_layers[ll] = Layer(State::Observed, static_cast<int>(data.second.cols()));
						m_layers[ll].set_output(data.second);
					}
					else {
						m_layers[ll] = Layer(State::Hidden, n_nodes);
					}
				}
			}

			const std::vector<Layer>::iterator layer(const int& ll) {
				if (ll > n_layers) throw std::runtime_error("index > n_layers");
				std::vector<Layer>::iterator lit = m_layers.begin();
				if (ll < 0) {
					return (lit + n_layers) + ll;
				}
				else {
					return lit + ll;
				}

				return lit;
			}
			const std::vector<Node>::iterator operator()(const unsigned int& nn, const unsigned int& ll) {
				if (ll > n_layers) throw std::runtime_error("index > n_layers");
				std::vector<Node>::iterator nit = m_layers[ll].m_nodes.begin() + nn;
				return nit;
			}
			void connect_inputs(const std::size_t& layer_idx) {
				if (layer(layer_idx)->cstate == State::Input) throw std::runtime_error("Invalid Connection: InputLayer");
				m_layers[layer_idx].connect(m_layers[0].observed_input);
			}

		private:
			std::vector<Layer> m_layers;
			void check_connected(const TMatrix& X) {
				for (std::vector<Layer>::iterator cp = m_layers.begin() + 1; cp != m_layers.end(); ++cp) {
					if (cp->cstate == State::InputConnected) cp->Xtmp = X;
				}
			}

			void lock() {
				for (std::vector<Layer>::iterator ll = m_layers.begin(); ll != m_layers.end(); ++ll) {
					ll->locked = true;
				}
			}
			void propagate(const Task& task) {
				// cp : CurrentLayer(PreviousLayer)
				for (std::vector<Layer>::iterator cp = m_layers.begin() + 1; cp != m_layers.end(); ++cp) {
					switch (task) {
					case (Init):
						if (cp->n_nodes == std::prev(cp)->n_nodes) {
							cp->set_input(std::prev(cp)->get_output());
							cp->set_output(cp->get_input());
						}
						else if (cp->n_nodes < std::prev(cp)->n_nodes) {
							if ((cp - m_layers.begin()) == n_layers - 1) {
								// Output Layer
								cp->set_input(std::prev(cp)->get_output());
							}
							else {
								kernelpca::KernelPCA pca(cp->n_nodes, "sigmoid");
								cp->set_input(std::prev(cp)->get_output());
								cp->set_output(pca.transform(std::prev(cp)->get_output()));
							}
						}
						else {
							/* Dimension Expansion */
							// idx = np.random.choice(input.shape[1], n_nodes - input.shape[1])
							// col = input[:, idx]
							// output = np.hstack([input, col])
							TMatrix cinputs = std::prev(cp)->get_output();
							TMatrix cols = cinputs.col(0).replicate(1, cp->n_nodes - std::prev(cp)->n_nodes);;
							TMatrix tmp(cinputs.rows(), cp->n_nodes);
							tmp << cinputs, cols;
							cp->set_input(std::prev(cp)->get_output());
							cp->set_output(tmp);
						}
						continue;
					case (Train):
						if (cp == m_layers.begin() + 1) std::prev(cp)->train();
						cp->train();
						continue;
					case(LinkedPredict):
						cp->n_thread = n_thread;
						cp->predict(std::prev(cp)->latent_output);
						continue;
					}
				}
			}
			friend class DGPSI;


		};

		class DGPSI {
		private:
			TMatrix update_f(const TMatrix& f, const TMatrix& nu, const double& params) {
				TVector mean = TVector::Zero(f.rows());
				return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
			}
			void sample(unsigned int n_burn = 1) {
				auto rand_u = [](const double& a, const double& b) {
					std::uniform_real_distribution<> uniform_dist(a, b);
					return uniform_dist(rng);
				};

				double log_y, theta, theta_min, theta_max;
				for (unsigned int nb = 0; nb < n_burn; ++nb) {
					for (std::vector<Layer>::iterator cl = graph.m_layers.begin(); cl != graph.m_layers.end() - 1; ++cl) {
						if (cl->cstate == State::Observed) continue; // TODO: missingness
						auto linked_layer = std::next(cl);
						for (std::vector<Node>::iterator cn = cl->m_nodes.begin(); cn != cl->m_nodes.end(); ++cn) {
							const Eigen::Index col = static_cast<Eigen::Index>((cn - cl->m_nodes.begin()));
							TMatrix nu(cn->inputs.rows(), 1);
							if (cl->cstate == State::InputConnected) nu = cn->evalK(cl->observed_input).sample_mvn();
							else nu = cn->evalK().sample_mvn();

							if (col == 0 && linked_layer->likelihood == LLF::Heteroskedastic) {
								TMatrix ff = linked_layer->m_nodes[0].posterior(cn->K); // cn->K
								linked_layer->set_input(ff, 0);
								cn->outputs = ff;
								continue;
							}
							log_y = linked_layer->evalK().log_likelihood() + log(rand_u(0.0, 1.0));
							//
							if (!std::isfinite(log_y)) { throw std::runtime_error("log_y is not finite"); }
							//
							theta = rand_u(0.0, 2.0 * PI);
							theta_min = theta - 2.0 * PI;
							theta_max = theta;
							while (true) {
								TMatrix fp = update_f(cn->outputs, nu, theta);
								linked_layer->set_input(fp, col);
								double log_yp = linked_layer->evalK().log_likelihood();
								//
								if (!std::isfinite(log_yp)) { throw std::runtime_error("log_yp is not finite"); }
								//
								if (log_yp > log_y) { cn->outputs = fp; break; }
								else {
									if (theta < 0) { theta_min = theta; }
									else { theta_max = theta; }
									theta = rand_u(theta_min, theta_max);
								}
							}
						}
					}
				}
			}
			void initialize_layers() {
				graph.lock();
				graph.propagate(Task::Init);
			}
		public:
			DGPSI(const Graph& graph) : graph(graph) {
				initialize_layers();
				sample(10);
			}

			//
			void train(int n_iter = 50, int ess_burn = 10, Eigen::Index n_burn = 0) {
				train_iter += n_iter;
				auto train_start = std::chrono::system_clock::now();
				std::time_t train_start_t = std::chrono::system_clock::to_time_t(train_start);
				std::cout << "START: " << std::put_time(std::localtime(&train_start_t), "%F %T") << std::endl;
				ProgressBar* train_prog = new ProgressBar(std::clog, 70u, "[TRAIN]");
				for (int i = 0; i < n_iter; ++i) {
					//double progress = double(i) * 100.0 / double(n_iter);
					train_prog->write((double(i) / double(n_iter)));
					// I-step
					sample(ess_burn);
					// M-step
					graph.propagate(Task::Train);
				}
				delete train_prog;
				auto train_end = std::chrono::system_clock::now();
				std::time_t train_end_t = std::chrono::system_clock::to_time_t(train_end);
				std::cout << "END: " << std::put_time(std::localtime(&train_end_t), "%F %T") << std::endl;
				std::cout << std::endl;
				// Estimate Parameters
				if (n_burn == 0) n_burn = std::size_t(0.75 * train_iter);
				else if (n_burn > train_iter) throw std::runtime_error("n_burn > train_iter");
				for (std::vector<Layer>::iterator layer = graph.m_layers.begin(); layer != graph.m_layers.end(); ++layer) {
					layer->estimate_parameters(n_burn);
				}
			}
			MatrixPair predict(const TMatrix& X, unsigned int n_predict = 50, unsigned int n_thread = 1) {
				sample(50);
				TMatrix mean = TMatrix::Zero(X.rows(), 1);
				TMatrix variance = TMatrix::Zero(X.rows(), 1);
				std::vector<MatrixPair> predictions;

				auto pred_start = std::chrono::system_clock::now();
				std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
				std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
				ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "[PREDICT]");
				graph.n_thread = n_thread;
				for (int i = 0; i < n_predict; ++i) {
					sample();
					graph.layer(0)->predict(X);
					graph.propagate(Task::LinkedPredict);
					MatrixPair output = graph.layer(-1)->latent_output;
					mean.noalias() += output.first;
					variance.noalias() += (square(output.first.array()).matrix() + output.second);
					pred_prog->write((double(i) / double(n_predict)));
				}
				delete pred_prog;

				auto pred_end = std::chrono::system_clock::now();
				std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
				std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
				std::cout << std::endl;
				mean.array() /= double(n_predict);
				variance.array() /= double(n_predict);
				variance.array() -= square(mean.array());

				return std::make_pair(mean, variance);
			}
			MatrixPair predict(const TMatrix& X, TMatrix& Yref, bool& nanflag, unsigned int n_predict = 50, unsigned int n_thread = 1) {
				sample(50);
				TMatrix mean = TMatrix::Zero(X.rows(), 1);
				TMatrix variance = TMatrix::Zero(X.rows(), 1);
				std::vector<MatrixPair> predictions;

				auto pred_start = std::chrono::system_clock::now();
				std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
				std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
				ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "");
				graph.n_thread = n_thread;
				graph.check_connected(X);
				for (int i = 0; i < n_predict; ++i) {
					sample();
					graph.layer(0)->predict(X);
					graph.propagate(Task::LinkedPredict);
					MatrixPair output = graph.layer(-1)->latent_output;
					mean.noalias() += output.first;
					variance.noalias() += (square(output.first.array()).matrix() + output.second);
					if ((mean.array().isNaN()).any()) { nanflag = true;  break; }
					TVector tmp_mu = mean.array() / double(i + 1);
					double nrmse = metrics::rmse(Yref, tmp_mu, true);
					if (i > 2 && nrmse > 0.5) { nanflag = true;  break; }
					double r2 = metrics::r2_score(Yref, tmp_mu);
					pred_prog->write((double(i) / double(n_predict)), nrmse, r2);
				}
				delete pred_prog;

				auto pred_end = std::chrono::system_clock::now();
				std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
				std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
				std::cout << std::endl;
				mean.array() /= double(n_predict);
				variance.array() /= double(n_predict);
				variance.array() -= square(mean.array());

				return std::make_pair(mean, variance);
			}
			MatrixPair predict(const TMatrix& X, TMatrix& Yref, std::string mcs_path, bool& nanflag, unsigned int n_predict = 50, unsigned int n_thread = 1) {
				sample(50);
				TMatrix mean = TMatrix::Zero(X.rows(), 1);
				TMatrix variance = TMatrix::Zero(X.rows(), 1);
				std::vector<MatrixPair> predictions;

				auto pred_start = std::chrono::system_clock::now();
				std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
				std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
				ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "");
				graph.n_thread = n_thread;
				graph.check_connected(X);
				for (int i = 0; i < n_predict; ++i) {
					sample();
					graph.layer(0)->predict(X);
					graph.propagate(Task::LinkedPredict);
					MatrixPair output = graph.layer(-1)->latent_output;
					mean.noalias() += output.first;
					variance.noalias() += (square(output.first.array()).matrix() + output.second);
					if ((mean.array().isNaN()).any()) { nanflag = true;  break; }

					TVector nrmse_mu = mean.array() / double(i + 1);
					double nrmse = metrics::rmse(Yref, nrmse_mu, true);
					// if (i > 2 && nrmse > 0.5) { nanflag = true;  break; }
					double r2 = metrics::r2_score(Yref, nrmse_mu);
					pred_prog->write((double(i) / double(n_predict)), nrmse, r2);

					if (i == 0 || i == 99 || i == 199 || i == 299 || i == 399 || i == 499) {
						std::string mu_path = mcs_path + "-" + std::to_string(i) + "-M-MCS.dat";
						std::string var_path = mcs_path + "-" + std::to_string(i) + "-V-MCS.dat";
						TMatrix tmp_mu = mean.array() / double(i + 1);
						TMatrix tmp_var = variance.array() / double(i + 1);
						tmp_var.array() -= square(tmp_mu.array());
						write_data(mu_path, tmp_mu);
						write_data(var_path, tmp_var);
					}
				}
				delete pred_prog;

				auto pred_end = std::chrono::system_clock::now();
				std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
				std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
				std::cout << std::endl;
				mean.array() /= double(n_predict);
				variance.array() /= double(n_predict);
				variance.array() -= square(mean.array());

				return std::make_pair(mean, variance);
			}
			//

			// COMPUTATIONAL COST
			void train(const std::string& time_path, int n_iter = 50, int ess_burn = 10, Eigen::Index n_burn = 0) {
				train_iter += n_iter;
				auto train_start = std::chrono::system_clock::now();
				std::time_t train_start_t = std::chrono::system_clock::to_time_t(train_start);
				std::cout << "START: " << std::put_time(std::localtime(&train_start_t), "%F %T") << std::endl;
				ProgressBar* train_prog = new ProgressBar(std::clog, 70u, "[TRAIN]");
				for (int i = 0; i < n_iter; ++i) {
					//double progress = double(i) * 100.0 / double(n_iter);
					train_prog->write((double(i) / double(n_iter)));
					// I-step
					sample(ess_burn);
					// M-step
					graph.propagate(Task::Train);
				}
				delete train_prog;
				auto train_end = std::chrono::system_clock::now();
				std::time_t train_end_t = std::chrono::system_clock::to_time_t(train_end);
				std::cout << "END: " << std::put_time(std::localtime(&train_end_t), "%F %T") << std::endl;
				std::cout << std::endl;

				std::string time_diff = "0\t" + std::to_string(std::difftime(train_end_t, train_start_t));
				write_to_file(time_path, time_diff);

				// Estimate Parameters
				if (n_burn == 0) n_burn = std::size_t(0.75 * train_iter);
				else if (n_burn > train_iter) throw std::runtime_error("n_burn > train_iter");
				for (std::vector<Layer>::iterator layer = graph.m_layers.begin(); layer != graph.m_layers.end(); ++layer) {
					layer->estimate_parameters(n_burn);
				}
			}
			MatrixPair predict(const std::string& time_path, const TMatrix& X, unsigned int n_predict = 50, unsigned int n_thread = 1) {
				sample(50);
				TMatrix mean = TMatrix::Zero(X.rows(), 1);
				TMatrix variance = TMatrix::Zero(X.rows(), 1);
				std::vector<MatrixPair> predictions;

				auto pred_start = std::chrono::system_clock::now();
				std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
				std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
				ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "[PREDICT]");
				graph.n_thread = n_thread;
				for (int i = 0; i < n_predict; ++i) {
					sample();
					graph.layer(0)->predict(X);
					graph.propagate(Task::LinkedPredict);
					MatrixPair output = graph.layer(-1)->latent_output;
					mean.noalias() += output.first;
					variance.noalias() += (square(output.first.array()).matrix() + output.second);
					pred_prog->write((double(i) / double(n_predict)));
				}
				delete pred_prog;

				auto pred_end = std::chrono::system_clock::now();
				std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
				std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
				std::cout << std::endl;
				mean.array() /= double(n_predict);
				variance.array() /= double(n_predict);
				variance.array() -= square(mean.array());

				std::string time_diff = "1\t" + std::to_string(std::difftime(pred_end_t, pred_start_t));
				write_to_file(time_path, time_diff);

				return std::make_pair(mean, variance);
			}
			MatrixPair predict(const std::string& time_path, const TMatrix& X, TMatrix& Yref, bool& nanflag, unsigned int n_predict = 50, unsigned int n_thread = 1) {
				sample(50);
				TMatrix mean = TMatrix::Zero(X.rows(), 1);
				TMatrix variance = TMatrix::Zero(X.rows(), 1);
				std::vector<MatrixPair> predictions;

				auto pred_start = std::chrono::system_clock::now();
				std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
				std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
				ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "");
				graph.n_thread = n_thread;
				graph.check_connected(X);
				for (int i = 0; i < n_predict; ++i) {
					sample();
					graph.layer(0)->predict(X);
					graph.propagate(Task::LinkedPredict);
					MatrixPair output = graph.layer(-1)->latent_output;
					mean.noalias() += output.first;
					variance.noalias() += (square(output.first.array()).matrix() + output.second);
					if ((mean.array().isNaN()).any()) { nanflag = true;  break; }
					TVector tmp_mu = mean.array() / double(i + 1);
					double nrmse = metrics::rmse(Yref, tmp_mu, true);
					if (i > 2 && nrmse > 0.5) { nanflag = true;  break; }
					double r2 = metrics::r2_score(Yref, tmp_mu);
					pred_prog->write((double(i) / double(n_predict)), nrmse, r2);
				}
				delete pred_prog;

				auto pred_end = std::chrono::system_clock::now();
				std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
				std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
				std::cout << std::endl;
				mean.array() /= double(n_predict);
				variance.array() /= double(n_predict);
				variance.array() -= square(mean.array());
				std::string time_diff = "1\t" + std::to_string(std::difftime(pred_end_t, pred_start_t));
				write_to_file(time_path, time_diff);
				return std::make_pair(mean, variance);
			}
			MatrixPair predict(const std::string& time_path, const TMatrix& X, TMatrix& Yref, std::string mcs_path, bool& nanflag, unsigned int n_predict = 50, unsigned int n_thread = 1) {
				sample(50);
				TMatrix mean = TMatrix::Zero(X.rows(), 1);
				TMatrix variance = TMatrix::Zero(X.rows(), 1);
				std::vector<MatrixPair> predictions;

				auto pred_start = std::chrono::system_clock::now();
				std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
				std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
				ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "");
				graph.n_thread = n_thread;
				graph.check_connected(X);
				for (int i = 0; i < n_predict; ++i) {
					sample();
					graph.layer(0)->predict(X);
					graph.propagate(Task::LinkedPredict);
					MatrixPair output = graph.layer(-1)->latent_output;
					mean.noalias() += output.first;
					variance.noalias() += (square(output.first.array()).matrix() + output.second);
					if ((mean.array().isNaN()).any()) { nanflag = true;  break; }

					TVector nrmse_mu = mean.array() / double(i + 1);
					double nrmse = metrics::rmse(Yref, nrmse_mu, true);
					// if (i > 2 && nrmse > 0.5) { nanflag = true;  break; }
					double r2 = metrics::r2_score(Yref, nrmse_mu);
					pred_prog->write((double(i) / double(n_predict)), nrmse, r2);

					if (i == 0 || i == 99 || i == 199 || i == 299 || i == 399 || i == 499) {
						std::string mu_path = mcs_path + "-" + std::to_string(i) + "-M-MCS.dat";
						std::string var_path = mcs_path + "-" + std::to_string(i) + "-V-MCS.dat";
						TMatrix tmp_mu = mean.array() / double(i + 1);
						TMatrix tmp_var = variance.array() / double(i + 1);
						tmp_var.array() -= square(tmp_mu.array());
						write_data(mu_path, tmp_mu);
						write_data(var_path, tmp_var);
					}
				}
				delete pred_prog;

				auto pred_end = std::chrono::system_clock::now();
				std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
				std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
				std::cout << std::endl;
				mean.array() /= double(n_predict);
				variance.array() /= double(n_predict);
				variance.array() -= square(mean.array());
				std::string time_diff = "1\t" + std::to_string(std::difftime(pred_end_t, pred_start_t));
				write_to_file(time_path, time_diff);
				return std::make_pair(mean, variance);
			}
			//


		public:
			Graph graph;
			unsigned int train_iter = 0;
			unsigned int verbosity = 1;
		};

	}

}
#endif

/* OLD CODE */
/* NODE */
//class Node : public GP {
//public:
//	Node(shared_ptr<Kernel> kernel) : GP(kernel) {
//		if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
//		if (!kernel->variance.fixed()) { kernel->variance.fix(); }
//	}
//	Node(shared_ptr<Kernel> kernel, shared_ptr<Solver> solver) : GP(kernel, solver) {
//		if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
//		if (!kernel->variance.fixed()) { kernel->variance.fix(); }
//	}
//	Node(shared_ptr<Kernel> kernel, const double& likelihood_variance, const double& scale_) :
//		GP(kernel, likelihood_variance), scale("scale", scale_) {
//		if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
//		if (!kernel->variance.fixed()) { kernel->variance.fix(); }
//	}
//	Node(shared_ptr<Kernel> kernel, const double& likelihood_variance, const double& scale_, shared_ptr<Solver> solver) :
//		GP(kernel, solver, likelihood_variance), scale("scale", scale_) {
//		if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
//		if (!kernel->variance.fixed()) { kernel->variance.fix(); }
//	}
//	Node(shared_ptr<Kernel> kernel, const Parameter<double>& likelihood_variance, const Parameter<double>& scale) :
//		GP(kernel, likelihood_variance), scale(scale) {
//		if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
//		if (!kernel->variance.fixed()) { kernel->variance.fix(); }
//	}
//	Node(shared_ptr<Kernel> kernel, const Parameter<double>& likelihood_variance, const Parameter<double>& scale_, shared_ptr<Solver> solver) :
//		GP(kernel, solver, likelihood_variance) {
//		scale = scale_;
//		if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
//		if (!kernel->variance.fixed()) { kernel->variance.fix(); }
//	}
//
//	double log_marginal_likelihood() override {
//		// Compute Log Likelihood [Rasmussen, Eq 2.30]
//		double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
//		double YKinvY = (outputs.transpose() * alpha)(0);
//		double NLL = 0.0;
//		if (*scale.is_fixed) { NLL = 0.5 * (logdet + YKinvY); }
//		else { NLL = 0.5 * (logdet + (inputs.rows() * log(scale.value()))); }
//		NLL -= log_prior();
//		return -NLL;
//	}
//	double log_likelihood() {
//		update_cholesky();
//		TMatrix _K = K.array() * scale.value();
//		TLLT _chol(_K);
//		double logdet = 2 * _chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
//		double quad = (outputs.array() * (_chol.solve(outputs)).array()).sum();
//		double lml = -0.5 * (logdet + quad);
//		return lml;
//	}
//	double log_prior() {
//		// Gamma Distribution
//		// self.g = lambda x : (self.prior_coef[0] - 1) * np.log(x) - self.prior_coef[1] * x			
//		const double shape = 1.6;
//		const double rate = 0.3;
//		double lp = 0.0;
//		if (!(*kernel->length_scale.is_fixed)) {
//			lp += (((shape - 1.0) * log(kernel->length_scale.value().array())) - (rate * kernel->length_scale.value().array())).sum();
//		}
//		if (!(*likelihood_variance.is_fixed)) {
//			lp += ((shape - 1.0) * log(likelihood_variance.value())) - (rate * likelihood_variance.value());
//		}
//		return lp;
//	}
//	TVector log_prior_gradient() {
//		// Gamma Distribution
//		// self.gfod = lambda x : (self.prior_coef[0] - 1) - self.prior_coef[1] * x
//		const double shape = 1.6;
//		const double rate = 0.3;
//		TVector lpg;
//		if (!(*kernel->length_scale.is_fixed)) {
//			lpg = (shape - 1.0) - (rate * kernel->length_scale.value().array()).array();
//		}
//		if (!(*likelihood_variance.is_fixed)) {
//			lpg.conservativeResize(lpg.size() + 1);
//			lpg.tail(1)(0) = (shape - 1.0) - (rate * likelihood_variance.value());
//		}
//		return lpg;
//	}
//
//	void train() override {
//		TVector lower_bound, upper_bound, theta;
//		get_bounds(lower_bound, upper_bound, false);
//		theta = get_params(false);
//
//		if (solver->from_optim) {
//			auto objective = [this](const TVector& x, TVector* grad, void* opt_data)
//			{return objective_(x, grad, nullptr, opt_data); };
//			opt::OptimData optdata;
//			solver->solve(theta, objective, optdata);
//		}
//		else {
//			// LBFGSB
//			Objective objective(this, static_cast<int>(lower_bound.size()));
//			objective.set_bounds(lower_bound, upper_bound);
//			solver->solve(theta, objective);
//		}
//	}
//
//	TVector gradients() override {
//		// dNLL = alpha*alpha^T - K^-1 [Rasmussen, Eq 5.9]
//		if (alpha.size() == 0) { update_cholesky(); }
//		TMatrix aaT = alpha * alpha.transpose().eval();
//		TMatrix Kinv = chol.solve(TMatrix::Identity(inputs.rows(), inputs.rows()));
//		TMatrix dNLL = 0.5 * (aaT - Kinv); // dL_dK
//
//		std::vector<double> grad;
//		// Get dK/dlengthscale and dK/dvariance -> {dK/dlengthscale, dK/dvariance}
//		kernel->gradients(inputs, dNLL, D, K, grad);
//		if (!(*likelihood_variance.is_fixed)) { grad.push_back(dNLL.diagonal().sum()); }
//		TVector _grad = Eigen::Map<TVector>(grad.data(), grad.size());
//		if (!(*scale.is_fixed)) { _grad.array() /= scale.value(); }
//		// gamma log_prior derivative
//		TVector lpg = log_prior_gradient();
//		if (kernel->ARD) { _grad -= lpg; }
//		else { _grad.array() -= lpg.coeff(0); }
//		return _grad;
//	}
//	MatrixVariant predict(const TMatrix& X, bool return_var = false)
//	{
//		update_cholesky();
//		TMatrix Ks(inputs.rows(), X.rows());
//		Ks.noalias() = kernel->K(inputs, X);
//		TMatrix mu = Ks.transpose() * alpha;
//		if (return_var) {
//			TMatrix Kss = kernel->diag(X);
//			TMatrix V = chol.solve(Ks);
//			TMatrix var = abs((scale.value() * (Kss - (Ks.transpose() * V).diagonal()).array()));
//			return std::make_pair(mu, var);
//		}
//		else { return mu; }
//	}
//	void linked_prediction(TVector& latent_mu, TVector& latent_var, const TMatrix& linked_mu, const TMatrix& linked_var, const int& n_thread) {
//
//		update_cholesky();
//		kernel->expectations(linked_mu, linked_var);
//
//		if (n_thread == 0 || n_thread == 1) {
//			for (Eigen::Index i = 0; i < linked_mu.rows(); ++i) {
//				TMatrix I = TMatrix::Ones(inputs.rows(), 1);
//				TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
//				kernel->IJ(I, J, linked_mu.row(i), linked_var.row(i), inputs, i);
//				double trace = (K.llt().solve(J)).trace();
//				double Ialpha = (I.cwiseProduct(alpha)).array().sum();
//				latent_mu[i] = (Ialpha);
//				latent_var[i] =
//					(abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum()
//						- (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
//			}
//		}
//		else {
//			Eigen::initParallel(); // /openmp (MSVC) or -fopenmp (GCC) flag
//			thread_pool pool;
//			int split = int(linked_mu.rows() / n_thread);
//			const int remainder = int(linked_mu.rows()) % n_thread;
//			auto task = [=, &latent_mu, &latent_var](int begin, int end)
//			{
//				for (Eigen::Index i = begin; i < end; ++i) {
//					TMatrix I = TMatrix::Ones(inputs.rows(), 1);
//					TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
//					kernel->IJ(I, J, linked_mu.row(i), linked_var.row(i), inputs, i);
//					double trace = (K.llt().solve(J)).trace();
//					double Ialpha = (I.cwiseProduct(alpha)).array().sum();
//					latent_mu[i] = (Ialpha);
//					latent_var[i] =
//						(abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum()
//							- (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
//				}
//			};
//			for (int s = 0; s < n_thread; ++s) {
//				pool.push_task(task, int(s * split), int(s * split) + split);
//			}
//			pool.wait_for_tasks();
//			if (remainder > 0) {
//				task(linked_mu.rows() - remainder, linked_mu.rows());
//			}
//			pool.reset();
//		}
//	}
//	void set_params(const TVector& new_params) override
//	{
//		// Explicitly mention order? order = {StationaryKernel_lengthscale, StationaryKernel_variance, likelihood_variance}
//		kernel->set_params(new_params);
//		if (!(*likelihood_variance.is_fixed)) { likelihood_variance.transform_value(new_params.tail(1)(0)); }
//		update_cholesky();
//	}
//	Eigen::Index params_size() {
//		TVector param = get_params();
//		return param.size();
//	}
//	TMatrix get_parameter_history() {
//		if (history.size() == 0)
//		{
//			throw std::runtime_error("No Parameters Saved, set store_parameters = true");
//		}
//		Eigen::Index param_size = params_size();
//		TMatrix _history(history.size(), param_size);
//		for (std::vector<TVector>::size_type i = 0; i != history.size(); ++i) {
//			_history.row(i) = history[i];
//		}
//		return _history;
//	}
//
//	// Setters (Python Interface)
//	void set_inputs(const TMatrix& input) { inputs = input; }
//	void set_outputs(const TMatrix& output) { outputs = output; }
//	// Getters (Python Interface)
//	const TMatrix get_inputs() { return inputs; }
//	const TMatrix get_outputs() { return outputs; }
//
//private:
//	double objective_(const TVector& x, TVector* grad, TVector* hess, void* opt_data) {
//		set_params(x);
//		if (grad) { (*grad) = gradients() * -1.0; }
//		return -log_marginal_likelihood();
//	}
//
//protected:
//	void update_cholesky() {
//		TMatrix noise = TMatrix::Identity(inputs.rows(), outputs.rows());
//		K = kernel->K(inputs, inputs, D);
//		K += (noise * likelihood_variance.value());
//		chol = K.llt();
//		alpha = chol.solve(outputs);
//		// scale is not considered a variable in optimization, it is directly linked to chol
//		if (!(*scale.is_fixed)) {
//			scale = (outputs.transpose() * alpha)(0) / outputs.rows();
//		}
//	}
//	void get_bounds(TVector& lower, TVector& upper, bool transformed = false) {
//		kernel->get_bounds(lower, upper, transformed);
//
//		if (!(*likelihood_variance.is_fixed)) {
//			if (transformed) { likelihood_variance.transform_bounds(); }
//			lower.conservativeResize(lower.rows() + 1);
//			upper.conservativeResize(upper.rows() + 1);
//			lower.tail(1)(0) = likelihood_variance.get_bounds().first;
//			upper.tail(1)(0) = likelihood_variance.get_bounds().second;
//		}
//	}
//	TVector get_params(bool inverse_transform = true) override {
//		TVector params;
//		params = kernel->get_params(inverse_transform);
//		if (!(*likelihood_variance.is_fixed)) {
//			likelihood_variance.transform_value(inverse_transform);
//			params.conservativeResize(params.rows() + 1);
//			params.tail(1)(0) = likelihood_variance.value();
//		}
//		return params;
//	}
//
//public:
//	Parameter<double> scale = { "scale", 1.0, "none" };
//	BoolVector missing;
//	double objective_value = 0.0;
//	bool store_parameters = false;
//	bool connected = false;
//	std::vector<TVector> history;
//
//protected:
//	TVector  alpha;
//	TLLT	 chol;
//	TMatrix	 K;
//	TMatrix	 D;
//};

/* LAYER */
//class Layer {
//private:
//	void check_nodes() {
//		for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
//		{
//			if (node->inputs.size() == 0)
//			{
//				throw std::runtime_error("A Node in the Layer has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
//			}
//			if (node->outputs.size() == 0)
//			{
//				throw std::runtime_error("A Node in the Layer has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
//			}
//		}
//	}
//	void estimate_parameters(const Eigen::Index& n_burn) {
//		for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//			TMatrix history = node->get_parameter_history();
//			TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
//			node->set_params(theta);
//		}
//	}
//public:
//
//	Layer(const std::vector<Node>& nodes_, bool initialize = true) : nodes(nodes_) {
//		if (initialize) {
//			// Checks
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				if (node->inputs.size() > 0)
//				{
//					throw std::runtime_error("Node has inputs, required empty");
//				}
//				if (node->outputs.size() > 0)
//				{
//					throw std::runtime_error("Node has outputs, required empty");
//				}
//				if (node->kernel == nullptr)
//				{
//					throw std::runtime_error("Node has no kernel, kernel required");
//				}
//			}
//		}
//	}
//	Layer& operator()(Layer& layer) {
//		// Initialize			[ CurrentLayer(NextLayer) ]
//		if (state == 0) {
//			layer.index = index + 1;
//			if (layer.nodes.size() == nodes.size())
//			{
//				layer.set_inputs(o_output);
//				if (layer.o_output.size() == 0 || layer.o_output.rows() != o_output.rows())
//				{
//					layer.set_outputs(o_output, true);
//				}
//			}
//			else if (layer.nodes.size() < nodes.size())
//			{
//				if (layer.last_layer) { layer.set_inputs(o_output); }
//				else {
//					// Apply Dimensionality Reduction (Kernel PCA)
//					layer.set_inputs(o_output);
//					kernelpca::KernelPCA pca(layer.nodes.size(), "sigmoid");
//					TMatrix input_transformed = pca.transform(o_output);
//					if (layer.o_output.size() == 0 || layer.o_output.rows() != o_output.rows())
//					{
//						layer.set_outputs(input_transformed, true);
//					}
//				}
//			}
//			else
//			{
//				//layer.set_inputs(o_output);
//				//if (!layer.last_layer) {/* Dimension Expansion*/ }
//
//			}
//		}
//		// Linked Prediction	[ CurrentLayer(PreviousLayer) ]
//		if (state == 2) {
//			TMatrix linked_mu = layer.latent_output.first;
//			TMatrix linked_var = layer.latent_output.second;
//			TMatrix output_mean = TMatrix::Zero(linked_mu.rows(), nodes.size());
//			TMatrix output_variance = TMatrix::Zero(linked_var.rows(), nodes.size());
//			Eigen::Index column = 0;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				TVector latent_mu = TVector::Zero(linked_mu.rows());
//				TVector latent_var = TVector::Zero(linked_mu.rows());
//				node->linked_prediction(latent_mu, latent_var, linked_mu, linked_var, n_thread);
//				output_mean.block(0, column, linked_mu.rows(), 1) = latent_mu;
//				output_variance.block(0, column, linked_var.rows(), 1) = latent_var;
//				column++;
//			}
//			latent_output.first = output_mean;
//			latent_output.second = output_variance;
//		}
//		return *this;
//	}
//	std::vector<TMatrix> get_parameter_history() {
//		std::vector<TMatrix> history;
//		for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//			history.push_back(node->get_parameter_history());
//		}
//		return history;
//	}
//	void train() {
//		if (state == 0 || state == 2) { check_nodes(); state = 1; }
//		for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//			if (!node->store_parameters) { node->store_parameters = true; }
//			node->train();
//		}
//	}
//	void predict(const TMatrix& X) {
//		/*
//		* All nodes (GPR) predict X and output a pair of N X M matrices
//		* Where N = number of rows X ; M = number of nodes in layer
//		* The pair is the mean and variance.
//		*/
//		TMatrix node_mu(X.rows(), nodes.size());
//		TMatrix node_var(X.rows(), nodes.size());
//		for (std::vector<Node>::size_type i = 0; i != nodes.size(); ++i)
//		{
//			MatrixPair pred = std::get<MatrixPair>(nodes[i].predict(X, true));
//			node_mu.block(0, i, X.rows(), 1) = pred.first;
//			node_var.block(0, i, X.rows(), 1) = pred.second;
//		}
//
//		latent_output = std::make_pair(node_mu, node_var);
//		//latent_output = pred;
//	}
//	void connect_observed_inputs() {
//		connected = true;
//		for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//			node->connected = true;
//		}
//	}
//
//	// Setters
//	void set_inputs(const TMatrix& inputs) {
//		o_input = inputs;
//		for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//			node->inputs = inputs;
//		}
//	}
//	void set_outputs(const TMatrix& outputs, bool latent = false) {
//		BoolVector missing;
//		if (latent) { missing = BoolVector::Ones(outputs.rows()); }
//		else { missing = BoolVector::Zero(outputs.rows()); }
//		if ((outputs.array().isNaN()).any())
//		{
//			missing = operations::get_missing_index<BoolVector>(outputs);
//		}
//
//		if (nodes.size())
//			for (std::size_t c = 0; c < nodes.size(); ++c) {
//				nodes[c].outputs = outputs.col(c);
//				nodes[c].missing = missing;
//			}
//		o_output = outputs;
//
//	}
//	// Getters
//	TMatrix get_inputs() { return o_input; }
//	TMatrix get_outputs() { return o_output; }
//	void reconstruct_observed(const TMatrix& inputs, const TMatrix& outputs) {
//		o_input = inputs;
//		o_output = outputs;
//	}
//	friend class DGPSI;
//public:
//	std::vector<Node> nodes;
//	bool last_layer = false;
//	bool connected = false;
//	int n_thread = 0;
//	int state = 0;
//	int index = 0;
//private:
//	TMatrix o_input;
//	TMatrix o_output;
//	MatrixPair latent_output;
//
//};

/* DGPSI */
//class DGPSI : public Imputer {
//
//private:
//	void initialize_layers() {
//		if (layers.front().o_input.size() == 0) { throw std::runtime_error("First Layer Requires Observed Inputs"); }
//		if (layers.back().o_output.size() == 0) { throw std::runtime_error("Last Layer Requires Observed Outputs"); }
//		//if (layers.back().nodes.size() != 1) { throw std::runtime_error("Last Layer Must Only have 1 Node for a Single Output"); }
//		layers.front().index = 1;
//		layers.back().last_layer = true;
//		// Propagate First Layer
//		TMatrix X = layers.front().get_inputs();
//		layers.front().set_outputs(X, true);
//
//		for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
//			layer->state = 0;
//			(*layer)(*std::next(layer));
//		}
//	}
//	void sample(int n_burn = 10) {
//		for (int i = 0; i < n_burn; ++i) {
//			// DEBUG
//			//std::cout << "iter = " << i << std::endl;
//			//
//			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
//				for (std::size_t n = 0; n < layer->nodes.size(); ++n) {
//					ess_update(layer->nodes[n], *std::next(layer), n);
//				}
//			}
//		}
//	}
//public:
//	DGPSI(const std::vector<Layer>& layers, bool initialize = true) : layers(layers) {
//		if (initialize) {
//			initialize_layers();
//			sample(10);
//		}
//	}
//
//	void train(int n_iter = 50, int ess_burn = 10) {
//		auto train_start = std::chrono::system_clock::now();
//		std::time_t train_start_t = std::chrono::system_clock::to_time_t(train_start);
//		auto verbose = [this, &train_start_t, &n_iter](const int& n, const int& i, const double& p) {
//			if (verbosity == 1) {
//				if (n == 0 && i == 1) {
//					std::cout << "TRAIN START: " <<
//						std::put_time(std::localtime(&train_start_t), "%F %T") << std::endl;
//				}
//				std::cout << std::setw(3) << std::left <<
//					std::setprecision(1) << std::fixed << p <<
//					std::setw(5) << std::left << " % |";
//				std::cout << std::setw(7) <<
//					std::left << " LAYER " << std::setw(3) <<
//					std::left << i << "\r" << std::flush;
//				if (n == n_iter - 1 && i == layers.size()) {
//					auto train_end = std::chrono::system_clock::now();
//					std::time_t train_end_t = std::chrono::system_clock::to_time_t(train_end);
//					std::cout << "TRAIN END: " <<
//						std::put_time(std::localtime(&train_end_t), "%F %T") << std::endl;
//				}
//			}
//		};
//		n_iter_ = n_iter;
//		for (int i = 0; i < n_iter; ++i) {
//			double progress = double(i + 1) * 100.0 / double(n_iter);
//			// I-step
//			sample(ess_burn);
//			// M-step
//			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
//				layer->train();
//				verbose(i, layer->index, progress);
//			}
//		}
//		//std::system("cls");
//		std::cout << std::endl;
//	}
//	void estimate(Eigen::Index n_burn = 0) {
//		if (n_burn == 0) { n_burn = std::size_t(0.75 * n_iter_); }
//		for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
//			layer->estimate_parameters(n_burn);
//		}
//	}
//	MatrixPair predict(const TMatrix& X, int n_impute = 50, int n_thread = 1) {
//		sample(50);
//		const std::size_t n_layers = layers.size();
//		TMatrix mean = TMatrix::Zero(X.rows(), 1);
//		TMatrix variance = TMatrix::Zero(X.rows(), 1);
//		std::vector<MatrixPair> predictions;
//
//		auto pred_start = std::chrono::system_clock::now();
//		std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
//		auto verbose = [this, &pred_start_t, &n_impute](const int& i, const double& p) {
//			if (verbosity == 1) {
//				if (i == 0) {
//					std::cout << "PREDICTION START: " <<
//						std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
//				}
//				std::cout << std::setw(3) << std::left <<
//					std::setprecision(1) << std::fixed << p << std::setw(5) << std::left << " % |";
//				std::cout << std::setw(7) <<
//					std::left << "N_IMPUTE" << std::setw(1) << std::left << "" << i << "\r" << std::flush;
//				if (i == n_impute - 1) {
//					auto pred_end = std::chrono::system_clock::now();
//					std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
//					std::cout << "PREDICTION END: " <<
//						std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
//				}
//			}
//		};
//		for (int i = 0; i < n_impute; ++i) {
//			double progress = double(i + 1) * 100.0 / double(n_impute);
//			sample();
//			layers.front().predict(X);
//			std::size_t j = 1;
//			for (std::vector<Layer>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
//				if (layer->state != 2) { layer->state = 2; }
//				layer->n_thread = n_thread;
//				(*layer)(*std::prev(layer));
//				j++;
//			}
//			if (i == 0) {
//				mean = layers.back().latent_output.first;
//				variance = square(layers.back().latent_output.first.array()).matrix() + layers.back().latent_output.second;
//			}
//			else {
//				mean.noalias() += layers.back().latent_output.first;
//				variance.noalias() += (square(layers.back().latent_output.first.array()).matrix() + layers.back().latent_output.second);
//			}
//			verbose(i, progress);
//		}
//		std::cout << std::endl;
//		mean.array() /= double(n_impute);
//		variance.array() /= double(n_impute);
//		variance.array() -= square(mean.array());
//
//		return std::make_pair(mean, variance);
//	}
//
//	void set_observed(const TMatrix& X, const TMatrix& Z) {
//		if (X.size() != layers.front().o_input.size()) {
//			layers.front().set_inputs(X);
//			layers.back().set_outputs(Z);
//			initialize_layers();
//		}
//		else {
//			layers.front().set_inputs(X);
//			layers.back().set_outputs(Z);
//		}
//	}
//
//	const std::vector<Layer> get_layers() const { return layers; }
//
//	void set_n_iter(const int& n) { n_iter_ = n; }
//	const int n_iterations() const { return n_iter_; }
//	const std::string model_type() const { return "DGPSI"; }
//
//private:
//	int n_iter_ = 0;
//	std::vector<Layer> layers;
//public:
//	int verbosity = 1;
//};


