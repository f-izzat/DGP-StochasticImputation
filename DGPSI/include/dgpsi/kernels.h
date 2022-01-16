#ifndef KERNELS_H
#define KERNELS_H

#include <dgpsi/utilities.h>
#include <dgpsi/parameters.h>
#include <numeric>

namespace dgpsi::kernels {
	using namespace dgpsi::parameters;
	using namespace dgpsi::utilities::metrics;

	// Move to kernels.h
	enum TKernel { TSquaredExponential, TMatern52 };

	class Kernel {

	protected:
		virtual void dK_dlengthscale(std::vector<TMatrix>& dK, std::vector<double>& grad, const TMatrix& tmp, const TMatrix& dK_dR, const TMatrix& dNLL, const TMatrix& R) = 0;
	public:

		Kernel() {
			Parameter<TVector> ls("length_scale", TVector::Ones(1));
			length_scale = std::move(ls);

		}
		Kernel(const Kernel& kernel) {
			length_scale = kernel.length_scale;
		}
		Kernel(const double& length_scale) : length_scale("length_scale", TVector::Constant(1, length_scale)) {}
		Kernel(TVector& length_scale) : length_scale("lengthscale", length_scale) {
			if (this->length_scale.value().size() > 1) ARD = true;
		}
		Kernel(const Parameter<TVector>& length_scale) : length_scale(length_scale) {
			if (this->length_scale.value().size() > 1) ARD = true;
		}

		virtual const TMatrix K(const TMatrix& X1, const TMatrix& X2) { TMatrix KK; return KK; };
		virtual const TMatrix K(const TMatrix& X1, const TMatrix& X2, TMatrix& R) { TMatrix KK; return KK; };
		virtual const TMatrix K(const TMatrix& X1, const TMatrix& X2, const double& likelihood_variance) { TMatrix KK; return KK; };
		virtual const TMatrix K(const TMatrix& X1, const TMatrix& X2, const double& likelihood_variance, const Eigen::Index idx) { TMatrix KK; return KK; };
		virtual TMatrix diag(const TMatrix& X1) { TMatrix KK; return KK; };
		virtual void get_bounds(TVector& lower, TVector& upper) = 0;
		virtual void expectations(const TMatrix& mean, const TMatrix& variance) = 0;
		virtual void IJ(TMatrix& I, TMatrix& J, const TVector& mean, const TVector& variance, const TMatrix& X, const Eigen::Index& idx) = 0;

		virtual void set_params(const TVector& params) = 0;
		virtual TVector get_params() { TVector tmp; return tmp; }
		virtual void gradients(const TMatrix& X, std::vector<TMatrix>& grad) = 0;

		// Python Pickling
		virtual const Parameter<TVector> get_lengthscale() const { return length_scale; }

	public:
		Parameter<TVector> length_scale;
		bool ARD = false;

	};
	
	class Stationary : public Kernel {

	protected:
		void dK_dlengthscale(std::vector<TMatrix>& dK, std::vector<double>& grad, const TMatrix& tmp, const TMatrix& dK_dR, const TMatrix& dNLL, const TMatrix& R) override {
			if (ARD) {
				for (std::vector<TMatrix>::size_type i = 0; i != dK.size(); i++) {
					grad.push_back(-(dK[i].cwiseProduct(tmp).sum()) / pow(length_scale.value().array()[i], 3));
				}
			}
			else {
				grad.push_back(-(dK_dR.cwiseProduct(dNLL).sum()) / length_scale.value().coeff(0));
			}

		}
	public:
		Stationary() {}
		Stationary(const Stationary& kernel) : Kernel(kernel) {}
		Stationary(const double& length_scale) : Kernel(length_scale) {}
		Stationary(TVector& length_scale) : Kernel(length_scale) {}
		Stationary(const Parameter<TVector>& length_scale) : Kernel(length_scale) {}

		TMatrix diag(const TMatrix& X1) override {
			// Stationary Stationary
			return TMatrix::Ones(X1.rows(), 1);
		}
		void get_bounds(TVector& lower, TVector& upper) override {
			if (*length_scale.is_fixed) return;
			lower.conservativeResize(length_scale.size());
			upper.conservativeResize(length_scale.size());
			lower.head(length_scale.size()) = length_scale.get_bounds().first;
			upper.head(length_scale.size()) = length_scale.get_bounds().second;
		}
		void set_params(const TVector& params) override
		{
			if (*length_scale.is_fixed) return;
			length_scale.transform_value(params.head(length_scale.value().size()));
		}
		TVector get_params() override {
			std::vector<double> params;
			if (!(*length_scale.is_fixed)) {
				std::size_t n = length_scale.size();
				params.resize(n);
				length_scale.transform_value(true);
				TVector::Map(&params[0], n) = length_scale.value();
			}
			return Eigen::Map<TVector>(params.data(), params.size());
		}
		virtual void gradients(const TMatrix& X, std::vector<TMatrix>& grad) = 0;
		virtual void expectations(const TMatrix& mean, const TMatrix& variance) = 0;
		virtual void IJ(TMatrix& I, TMatrix& J, const TVector& mean, const TVector& variance, const TMatrix& X, const Eigen::Index& idx) = 0;
	};

	class SquaredExponential : public Stationary {

	private:
		void zeta(TMatrix& J, const TMatrix& Xz, const TMatrix& zeta0, const TMatrix& QD1, const TVector& QD2) {
			for (int j = 0; j < Xz.cols(); ++j) {
				TVector L = static_cast<TVector>(Xz.col(j).array().square().matrix()).array();
				TMatrix LR = L.replicate(1, Xz.rows()).transpose();
				TMatrix CL = (2 * (Xz.col(j) * Xz.col(j).transpose()).array()).matrix();
				J.array() *= zeta0(j) * exp(((-((LR + CL).colwise() + L) / QD1(j)) - (((LR - CL).colwise() + L) / QD2(j))).array()).array();
			}
		}

	public:

		SquaredExponential() : Stationary() {};
		SquaredExponential(const SquaredExponential& kernel) : Stationary(kernel) {}
		SquaredExponential(const double& length_scale) : Stationary(length_scale) {}
		SquaredExponential(TVector& length_scale) : Stationary(length_scale) {}
		SquaredExponential(const Parameter<TVector>& length_scale) : Stationary(length_scale) {}

		const TMatrix K(const TMatrix& X1, const TMatrix& X2) override {
			// Returns Stationary TMatrix
			TMatrix R2(X1.rows(), X2.rows());
			TVector tmp;
			if (ARD) tmp = length_scale.value();
			else tmp = TVector::Constant(X1.cols(), length_scale.value()(0));
			const TMatrix X1sc = X1.array().rowwise() / tmp.transpose().array();
			const TMatrix X2sc = X2.array().rowwise() / tmp.transpose().array();
			euclidean_distance(X1sc, X2sc, R2, true);
			//return variance.value() * exp(-0.5 * R2.array());
			return  exp(-R2.array());
		}
		const TMatrix K(const TMatrix& X1, const TMatrix& X2, const double& likelihood_variance) override {
			TMatrix R2(X1.rows(), X2.rows());
			TVector tmp;
			if (ARD) tmp = length_scale.value();
			else tmp = TVector::Constant(X1.cols(), length_scale.value()(0));
			const TMatrix X1sc = X1.array().rowwise() / tmp.transpose().array();
			const TMatrix X2sc = X2.array().rowwise() / tmp.transpose().array();
			euclidean_distance(X1sc, X2sc, R2, true);
			TMatrix noise = TMatrix::Identity(X1.rows(), X2.rows()).array() * likelihood_variance;
			//return ( exp(-0.5 * R2.array())).matrix() + noise;
			return ( exp(-R2.array())).matrix() + noise;
		}
		const TMatrix K(const TMatrix& X1, const TMatrix& X2, const double& likelihood_variance, const Eigen::Index idx) {
			TMatrix R2(X1.rows(), X2.rows());
			TVector tmp;
			if (ARD) tmp = length_scale.value();
			else tmp = TVector::Constant(X1.cols(), length_scale.value()(0));
			const TMatrix X1sc = X1.array() / tmp[idx];
			const TMatrix X2sc = X2.array() / tmp[idx];
			euclidean_distance(X1sc, X2sc, R2, true);
			TMatrix noise = TMatrix::Identity(X1.rows(), X2.rows()).array() * likelihood_variance;
			//return ( exp(-0.5 * R2.array())).matrix() + noise;			
			return ( exp(-R2.array())).matrix() + noise;
		};

		void gradients(const TMatrix& X, std::vector<TMatrix>& grad) override {
			if (!(*length_scale.is_fixed)) {
				std::vector<TMatrix> disi;
				TVector tmp;
				if (length_scale.size() < X.cols())
				{   // Expand lengthscale dimensions
					tmp = TVector::Constant(X.cols(), 1, length_scale.value()(0));
				}
				else {
					tmp = length_scale.value();
				}
				TMatrix Xsc = X.array().rowwise() / tmp.transpose().array();

				pdist(Xsc, Xsc, disi, true);
				TMatrix disi2 = TMatrix::Zero(X.rows(), X.rows());
				for (int i = 0; i < disi.size(); ++i) {
					disi2.array() += disi[i].array();
				}
				TMatrix K = exp(-disi2.array());
				if (ARD) {
					for (int i = 0; i < disi.size(); ++i) {
						grad.push_back(2 * disi[i].array() * K.array());
					}
				}
				else {
					grad.push_back(2 * disi2.array() * K.array());
				}
			}
		}
		void IJ(TMatrix& I, TMatrix& J, const TVector& mean, const TVector& variance, const TMatrix& X, const Eigen::Index& idx) override {
			TMatrix Xz = ((X.transpose().array().colwise() - mean.array())).transpose();
			// Compute I
			TMatrix xi = (exp(((-1 * square(Xz.array())).array().rowwise() / static_cast<TRVector>(xi_term2.row(idx)).array())).matrix()) * (xi_term1.row(idx).asDiagonal());
			I = xi.rowwise().prod();
			// Compute J
			zeta(J, Xz, zeta0.row(idx), QD1.row(idx), QD2);
		}
		void expectations(const TMatrix& mean, const TMatrix& variance) override {
			TRVector sqrd_ls = square(static_cast<TRVector>(length_scale.value()).array());
			xi_term1 = 1 / sqrt(1 + ((2 * variance.array()).rowwise() / sqrd_ls.array()));
			xi_term2 = (2 * variance.array()).rowwise() + sqrd_ls.array();
			zeta0 = 1 / sqrt(1 + ((4 * variance.array()).rowwise() / sqrd_ls.array()));
			QD1 = (8 * variance.array()).rowwise() + (2 * sqrd_ls.array());
			QD2 = 2 * square(length_scale.value().array()).array();
		}

	private:
		// SE Expectation Terms
		TMatrix xi_term1;
		TMatrix xi_term2;
		TMatrix zeta0;
		TMatrix QD1;
		TVector QD2;

	};

	class Matern52 : public Stationary {
	public:
		Matern52() {}
		Matern52(const Matern52& kernel) : Stationary(kernel) {}
		Matern52(const double& length_scale) : Stationary(length_scale) {}
		Matern52(TVector& length_scale) : Stationary(length_scale) {}
		Matern52(const Parameter<TVector>& length_scale) : Stationary(length_scale) {}
		const TMatrix K(const TMatrix& X1, const TMatrix& X2) override {
			TMatrix K1 = TMatrix::Ones(X1.rows(), X2.rows());
			TMatrix K2 = TMatrix::Zero(X1.rows(), X2.rows());
			TVector tmp;
			if (ARD) tmp = length_scale.value();
			else tmp = TVector::Constant(X1.cols(), length_scale.value()(0));
			const TMatrix X1sc = X1.array().rowwise() / tmp.transpose().array();
			const TMatrix X2sc = X2.array().rowwise() / tmp.transpose().array();
			std::vector<TMatrix> D;
			abspdist(X1sc, X2sc, D);
			for (unsigned int i = 0; i < D.size(); ++i) {
				K1.array() *= (1 + sqrt(5.0) * D[i].array() + (5.0 / 3.0) * square(D[i].array())).array();
				K2 += D[i];
			}
			return K1.array() * exp(-sqrt(5.0) * K2.array());
		}
		const TMatrix K(const TMatrix& X1, const TMatrix& X2, const double& likelihood_variance) override {
			TMatrix K1 = TMatrix::Ones(X1.rows(), X2.rows());
			TMatrix K2 = TMatrix::Zero(X1.rows(), X2.rows());
			TVector tmp;
			if (ARD) tmp = length_scale.value();
			else tmp = TVector::Constant(X1.cols(), length_scale.value()(0));
			const TMatrix X1sc = X1.array().rowwise() / tmp.transpose().array();
			const TMatrix X2sc = X2.array().rowwise() / tmp.transpose().array();
			std::vector<TMatrix> D;
			abspdist(X1sc, X2sc, D);
			for (unsigned int i = 0; i < D.size(); ++i) {
				K1.array() *= (1 + sqrt(5.0) * D[i].array() + (5.0 / 3.0) * square(D[i].array())).array();
				K2 += D[i];
			}
			TMatrix res = K1.array() * exp(-sqrt(5.0) * K2.array());
			res.diagonal().array() += likelihood_variance;
			return res;
		}
		const TMatrix K(const TMatrix& X1, const TMatrix& X2, const double& likelihood_variance, const Eigen::Index idx) {
			TMatrix K1 = TMatrix::Ones(X1.rows(), X2.rows());
			TMatrix K2 = TMatrix::Zero(X1.rows(), X2.rows());
			TMatrix R(X1.rows(), X2.rows());
			TVector tmp;
			if (ARD) tmp = length_scale.value();
			else tmp = TVector::Constant(X1.cols(), length_scale.value()(0));
			const TMatrix X1sc = X1.array() / tmp[idx];
			const TMatrix X2sc = X2.array() / tmp[idx];
			std::vector<TMatrix> D;
			abspdist(X1sc, X2sc, D);
			for (unsigned int i = 0; i < D.size(); ++i) {
				K1.array() *= (1 + sqrt(5.0) * D[i].array() + (5.0 / 3.0) * square(D[i].array())).array();
				K2 += D[i];
			}
			TMatrix res = K1.array() * exp(-sqrt(5.0) * K2.array());
			res.diagonal().array() += likelihood_variance;
			return res;
		}

		void gradients(const TMatrix& X, std::vector<TMatrix>& grad) override {
			if (*length_scale.is_fixed) return;
			std::vector<TMatrix> disi;
			TVector tmp;
			if (length_scale.size() < X.cols())
			{   // Expand lengthscale dimensions
				tmp = TVector::Constant(X.cols(), length_scale.value()(0));
			}
			else {
				tmp = length_scale.value();
			}
			TMatrix Xsc = X.array().rowwise() / tmp.transpose().array();

			pdist(Xsc, Xsc, disi, false);
			TMatrix K1 = TMatrix::Ones(X.rows(), X.rows());
			TMatrix K2 = TMatrix::Zero(X.rows(), X.rows());
			std::vector<TMatrix> coefi;

			for (int i = 0; i < disi.size(); ++i) {
				K1.array() *= (1 + sqrt(5.0) * abs(disi[i].array()) + (5.0 / 3.0) * disi[i].array().square());
				K2.array() += abs(disi[i].array());
				coefi.push_back(
					(disi[i].array().square() * (1.0 + sqrt(5.0) * abs(disi[i].array()))).array() /
					(1.0 + sqrt(5.0) * abs(disi[i].array()) + (5.0 / 3.0) * disi[i].array().square())
				);
			}
			K2.array() = exp(-sqrt(5.0) * K2.array());
			TMatrix K = K1.array() * K2.array();
			if (ARD) {
				for (int i = 0; i < coefi.size(); ++i) {
					grad.push_back((5.0 / 3.0) * coefi[i].array() * K.array());
				}
			}
			else {
				TMatrix coeff_sum = TMatrix::Zero(X.rows(), X.rows());
				for (int i = 0; i < coefi.size(); ++i) {
					coeff_sum.array() += coefi[i].array();
				}
				grad.push_back((5.0 / 3.0) * coeff_sum.array() * K.array());
			}
		}
		void IJ(TMatrix& I, TMatrix& J, const TVector& mean, const TVector& variance_, const TMatrix& X, const Eigen::Index& idx) override {
			// To avoid repetitive transpose, create tmp (RowMajor) length_scale and variance_
			TRVector ls;
			if (ARD) ls = static_cast<TRVector>(length_scale.value());
			else ls = TRVector::Constant(1, X.cols(), length_scale.value().coeff(0));

			TRVector mu = static_cast<TRVector>(mean);
			TRVector var = static_cast<TRVector>(variance_);

			// Find all variances that are zero
			std::vector<Eigen::Index> indices(variance_.size());
			std::iota(indices.begin(), indices.end(), 0);

			std::vector<Eigen::Index> zero_indices;
			std::vector<Eigen::Index> non_zero_indices;
			for (Eigen::Index i = 0; i < variance_.size(); ++i)
			{
				if (variance_[i] == 0.0) { zero_indices.push_back(i); }
			}
			std::set_difference(indices.begin(), indices.end(), zero_indices.begin(), zero_indices.end(),
				std::inserter(non_zero_indices, non_zero_indices.begin()));

			TMatrix  Xz = (-(X.array().rowwise() - mean.transpose().array()));
			TMatrix  muA = Xz.array().rowwise() - (sqrt(5) * var.array()) / ls.array();
			TMatrix  muB = (Xz.array().rowwise() + (sqrt(5) * var.array()) / ls.array());
			TRVector muC = mu.array() - ((2.0 * sqrt(5) * var.array()) / ls.array());
			TRVector muD = mu.array() + ((2.0 * sqrt(5) * var.array()) / ls.array());

			auto zero_variance = [&I, &J, &Xz, &ls, &zero_indices, this]()
			{
				TMatrix  tmp_Xz = Xz(Eigen::all, zero_indices);
				TRVector tmp_ls;
				tmp_ls = ls(zero_indices);
				// Id = (1 + sqrt(5)*np.abs(zX[i])/length[i] + 5*zX[i]**2/(3*length[i]**2)) * np.exp(-sqrt(5)*np.abs(zX[i])/length[i])
				TMatrix Id = (1.0 + ((sqrt(5.0) * abs(tmp_Xz.array())).array().rowwise() / tmp_ls.array()) +
					(5.0 * square(tmp_Xz.array())).rowwise() / (3.0 * square(tmp_ls.array()))) *
					(exp((-sqrt(5.0) * abs(tmp_Xz.array()).array()).rowwise() / tmp_ls.array()));
				I.array() *= (Id.rowwise().prod()).array();
				J.array() *= (I * I.transpose()).array();
			};

			auto non_zero_variance = [&I, &J, &X, &mean, &ls, &var, &muA, &muB, &muC, &muD, &non_zero_indices, this]()
			{
				TVector J0 = TVector::Ones(X.rows() * X.rows());
				// Mask with non zero variance indicies
				TMatrix  tmp_X = X(Eigen::all, non_zero_indices);
				TRVector tmp_ls;
				tmp_ls = ls(non_zero_indices);
				TVector  tmp_mu = mean(non_zero_indices);
				TRVector tmp_var = var(non_zero_indices);
				TMatrix  tmp_muA = muA(Eigen::all, non_zero_indices);
				TMatrix  tmp_muB = muB(Eigen::all, non_zero_indices);
				TRVector tmp_muC = muC(non_zero_indices);
				TRVector tmp_muD = muD(non_zero_indices);

				/* ================================ COMPUTE I  ================================ */
				TMatrix Xz = (-(tmp_X.array().rowwise() - tmp_mu.transpose().array()));
				TMatrix mt1 = ((-(Xz.array().rowwise() * (2.0 * sqrt(5.0) * tmp_ls.array()).matrix().array()).array()).array().rowwise() +
					(5.0 * tmp_var.array()).matrix().array());
				TMatrix pt1 = (((Xz.array().rowwise() * (2.0 * sqrt(5.0) * tmp_ls.array()).matrix().array()).array()).array().rowwise() +
					(5.0 * tmp_var.array()).matrix().array());

				TMatrix t1 = exp(mt1.array().rowwise() / (2 * square(tmp_ls.array())).matrix().array()).array() *
					((((1.0 + ((sqrt(5.0) * tmp_muA.array()).rowwise() / tmp_ls.array()).array()) +
						(5.0 * ((square(tmp_muA.array()).rowwise() + tmp_var.array()).rowwise() / (3 * square(tmp_ls.array()))).array())).array() *
						(pnorm((tmp_muA.array().rowwise() / sqrt(tmp_var.array())).matrix())).array()) +
						((sqrt(5.0) + ((5.0 * tmp_muA.array()).rowwise() / (3.0 * tmp_ls.array())).array()) *
							((exp((-0.5 * square(tmp_muA.array())).rowwise() / tmp_var.array())).array().rowwise() *
								(sqrt(0.5 * (tmp_var.array() / PI)) / tmp_ls.array()).matrix().array())));

				TMatrix t2 = exp(pt1.array().rowwise() / (2 * square(tmp_ls.array())).matrix().array()) * (
					(((1.0 - ((sqrt(5.0) * tmp_muB.array()).rowwise() / tmp_ls.array()).array()) +
						(5.0 * ((square(tmp_muB.array()).rowwise() + tmp_var.array()).rowwise() / (3 * square(tmp_ls.array()))).array())).array() *
						pnorm(((-tmp_muB.array()).rowwise() / sqrt(tmp_var.array())).matrix()).array()) +
					((sqrt(5.0) - ((5.0 * tmp_muB.array()).rowwise() / (3.0 * tmp_ls.array())).array()).array() *
						((exp((-0.5 * square(tmp_muB.array())).rowwise() / tmp_var.array())).array().rowwise() *
							(sqrt(0.5 * (tmp_var.array() / PI)) / tmp_ls.array()).matrix().array()).array()));

				I.array() *= ((t1 + t2).rowwise().prod()).array();

				/* ================================ COMPUTE J  ================================ */
				// E3A31 Coefficients
				// [(muC**2 + z_v) * E32] 
				TRVector CE32 = (square(tmp_muC.array()) + tmp_var.array());
				// (muC**3 + 3*z_v*muC)
				TRVector CE33 = pow(tmp_muC.array(), 3) + (3.0 * tmp_var.array() * tmp_muC.array());
				// (muC**4 + 6*z_v*muC**2 + 3*z_v**2)
				TRVector CE34 = pow(tmp_muC.array(), 4) + (6.0 * tmp_var.array() * square(tmp_muC.array())) + (3.0 * square(tmp_var.array()));
				// E5A51 Coefficients
				// [(muD**2 + z_v) * E52]
				TRVector DE52 = (square(tmp_muD.array()) + tmp_var.array());
				// [(muD**3 + 3*z_v*muD) * E53]
				TRVector DE53 = pow(tmp_muD.array(), 3) + (3.0 * tmp_var.array() * tmp_muD.array());
				// [(muD**4 + 6*z_v*muD**2 + 3*z_v**2) * E54]
				TRVector DE54 = pow(tmp_muD.array(), 4) + (6.0 * tmp_var.array() * square(tmp_muD.array())) + (3.0 * square(tmp_var.array()));
				// Loop over each column
				for (Eigen::Index c = 0; c < tmp_X.cols(); ++c) {
					TMatrix XX(int(pow(tmp_X.rows(), 2)), 2);
					TMatrix X1 = tmp_X(Eigen::all, c).transpose().replicate(tmp_X.rows(), 1);
					TMatrix X2 = tmp_X(Eigen::all, c).replicate(1, tmp_X.rows());
					TVector V1 = Eigen::Map<TVector>(X1.data(), X1.size());
					TVector V2 = Eigen::Map<TVector>(X2.data(), X2.size());
					XX << (V1.array() > V2.array()).select(V2, V1), (V1.array() > V2.array()).select(V1, V2);
					// Define Repetitive operations
					// (x1 * x2)
					TMatrix op_prod = XX.rowwise().prod();
					// (x1 + x2)
					TMatrix op_sum = XX.rowwise().sum();
					// (x1**2 + x2**2)
					TMatrix op_sq_sum = square(XX.array()).rowwise().sum();
					// (x1**2 * x2**2)
					TMatrix op_sq_prod = square(XX.array()).rowwise().prod();
					// Define repetitive terms
					double ls_sqrd = pow(tmp_ls(c), 2);
					double denominator = (9.0 * pow(tmp_ls(c), 4));
					double sqrtf = sqrt(5.0);
					double EX4 = 25.0 / (denominator);
					/* ================================ COMPUTE E3  ================================ */
					TMatrix E30 = 1.0 + (((25.0 * op_sq_prod.array()) -
						(((op_prod.array() * (5.0 * tmp_ls(c))) + (3.0 * pow(tmp_ls(c), 3))) * ((3.0 * sqrtf) * op_sum.array())).array() +
						((op_sq_sum.array() + (3.0 * op_prod.array())) * (15.0 * ls_sqrd))) / (denominator));
					TMatrix E31 = (((op_sq_sum.array() * (15.0 * sqrtf * tmp_ls(c))) -
						(((50.0 * op_prod.array()) + (75.0 * ls_sqrd)) * op_sum.array()) +
						(op_prod.array() * (60.0 * sqrtf * tmp_ls(c)))) +
						(18.0 * sqrtf * pow(tmp_ls(c), 3))) / (denominator);
					TMatrix E32 = (5.0 * (((5.0 * op_sq_sum.array()) + (15.0 * ls_sqrd)) - (op_sum.array() * (9.0 * sqrtf * tmp_ls(c))) +
						(20.0 * op_prod.array()))) / (denominator);
					TMatrix E33 = (10.0 * ((-5.0 * op_sum.array()) + (3.0 * sqrtf * tmp_ls(c)))) / (denominator);
					TMatrix E3A31 = E30.array() + (tmp_muC(c) * E31.array()) + (CE32(c) * E32.array()) + (CE33(c) * E33.array()) + (CE34(c) * EX4);
					TMatrix E3A32 = E31.array() + (tmp_muC(c) + XX.col(1).array()) * E32.array() +
						(pow(tmp_muC(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(1).array()) + (tmp_muC(c) * XX.col(1).array())) * E33.array() +
						(pow(tmp_muC(c), 3) + pow(XX.col(1).array(), 3) + (pow(tmp_muC(c), 2) * XX.col(1).array()) + (tmp_muC(c) * square(XX.col(1).array())) +
							(3.0 * tmp_var(c) * XX.col(1).array()) + (5.0 * tmp_var(c) * tmp_muC(c))) * EX4;

					TMatrix P1 = (exp((10.0 * tmp_var(c) + (sqrtf * tmp_ls(c) * (op_sum.array() - (2.0 * tmp_mu(c))))) / ls_sqrd)) *
						((0.5 * E3A31.array() * (1.0 + erf((tmp_muC(c) - XX.col(1).array()) / sqrt(2.0 * tmp_var(c))))) +
							(E3A32.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(1).array() - tmp_muC(c)) / tmp_var(c))));
					/* ================================ COMPUTE E4  ================================ */
					TMatrix E40 = 1.0 + (((25.0 * op_sq_prod.array()) +
						(3.0 * sqrtf * ((3.0 * pow(tmp_ls(c), 3)) - (5.0 * tmp_ls(c) * op_prod.array())) * (XX.col(1).array() - XX.col(0).array())) +
						(15.0 * ls_sqrd * (op_sq_sum.array() - (3.0 * op_prod.array())))) / (denominator));
					TMatrix E41 = 5.0 * ((3.0 * sqrtf * tmp_ls(c) * (square(XX.col(1).array()) - square(XX.col(0).array()))) +
						(3.0 * ls_sqrd * op_sum.array()) - (10.0 * op_prod.array() * op_sum.array())) / (denominator);
					TMatrix E42 = 5.0 * ((5.0 * op_sq_sum.array()) - (3.0 * ls_sqrd) - (3.0 * sqrtf * tmp_ls(c) * (XX.col(1).array() - XX.col(0).array())) +
						(20.0 * op_prod.array())) / (denominator);
					TMatrix E43 = -50.0 * (V1.array() + V2.array()) / (denominator);
					TMatrix E4A41 = E40.array() +
						(tmp_mu(c) * E41.array()) + ((pow(tmp_mu(c), 2) + tmp_var(c)) * E42.array()) +
						((pow(tmp_mu(c), 3) + 3.0 * tmp_var(c) * tmp_mu(c)) * E43.array()) +
						((pow(tmp_mu(c), 4) + 6.0 * tmp_var(c) * pow(tmp_mu(c), 2) + 3.0 * pow(tmp_var(c), 2)) * EX4);
					TMatrix E4A42 = E41.array() +
						((tmp_mu(c) + XX.col(0).array()) * E42.array()) +
						((pow(tmp_mu(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(0).array()) + (tmp_mu(c) * XX.col(0).array())) * E43.array()) +
						((pow(tmp_mu(c), 3) + pow(XX.col(0).array(), 3) + (pow(tmp_mu(c), 2) * XX.col(0).array()) + (tmp_mu(c) * square(XX.col(0).array())) +
							(3.0 * tmp_var(c) * XX.col(0).array()) + (5.0 * tmp_mu(c) * tmp_var(c))) * EX4);
					TMatrix E4A43 = E41.array() +
						((tmp_mu(c) + XX.col(1).array()) * E42.array()) +
						((pow(tmp_mu(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(1).array()) + (tmp_mu(c) * XX.col(1).array())) * E43.array()) +
						((pow(tmp_mu(c), 3) + pow(XX.col(1).array(), 3) + (pow(tmp_mu(c), 2) * XX.col(1).array()) + (tmp_mu(c) * square(XX.col(1).array())) +
							(3.0 * tmp_var(c) * XX.col(1).array()) + (5.0 * tmp_mu(c) * tmp_var(c))) * EX4);

					TMatrix P2 = exp(-sqrtf * (XX.col(1).array() - XX.col(0).array()) / tmp_ls(c)) *
						((0.5 * E4A41.array() * (erf((XX.col(1).array() - tmp_mu(c)) / (sqrt(2.0 * tmp_var(c)))) -
							erf((XX.col(0).array() - tmp_mu(c)) / (sqrt(2.0 * tmp_var(c)))))) +
							(E4A42.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(0).array() - tmp_mu(c)) / tmp_var(c))) -
							(E4A43.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(1).array() - tmp_mu(c)) / tmp_var(c))));
					/* ================================ COMPUTE E5  ================================ */
					TMatrix E50 = 1.0 + (((25.0 * op_sq_prod.array()) +
						(3.0 * sqrtf * ((3.0 * pow(tmp_ls(c), 3)) + (5.0 * tmp_ls(c) * op_prod.array())) * op_sum.array()) +
						(15.0 * ls_sqrd * (op_sq_sum.array() + (3.0 * op_prod.array())))) / (denominator));
					TMatrix E51 = (((op_sq_sum.array() * (15.0 * sqrtf * tmp_ls(c))) +
						(((50.0 * op_prod.array()) + (75.0 * ls_sqrd)) * op_sum.array()) +
						(op_prod.array() * (60.0 * sqrtf * tmp_ls(c)))) +
						(18.0 * sqrtf * pow(tmp_ls(c), 3))) / (denominator);
					TMatrix E52 = (5.0 * (((5.0 * op_sq_sum.array()) + (15.0 * ls_sqrd)) + (op_sum.array() * (9.0 * sqrtf * tmp_ls(c))) + (20.0 * op_prod.array()))) / (denominator);
					TMatrix E53 = (10.0 * ((5.0 * op_sum.array()) + (3.0 * sqrtf * tmp_ls(c)))) / (denominator);
					TMatrix E5A51 = E50.array() - (tmp_muD(c) * E51.array()) + (DE52(c) * E52.array()) - (DE53(c) * E53.array()) + (DE54(c) * EX4);
					TMatrix E5A52 = E51.array() - (tmp_muD(c) + XX.col(0).array()) * E52.array() +
						(pow(tmp_muD(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(0).array()) + (tmp_muD(c) * XX.col(0).array())) * E53.array() -
						(pow(tmp_muD(c), 3) + pow(XX.col(0).array(), 3) + (pow(tmp_muD(c), 2) * XX.col(0).array()) + (tmp_muD(c) * square(XX.col(0).array())) +
							(3.0 * tmp_var(c) * XX.col(0).array()) + (5.0 * tmp_var(c) * tmp_muD(c))) * EX4;
					TMatrix P3 = (exp((10.0 * tmp_var(c) - (sqrtf * tmp_ls(c) * (op_sum.array() - (2.0 * tmp_mu(c))))) / ls_sqrd)) *
						((0.5 * E5A51.array() * (1.0 + erf((XX.col(0).array() - tmp_muD(c)) / sqrt(2.0 * tmp_var(c))))) +
							(E5A52.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(0).array() - tmp_muD(c)) / tmp_var(c))));
					J0.array() *= (P1.array() + P2.array() + P3.array());

				}
				J.array() *= Eigen::Map<TMatrix>(J0.data(), X.rows(), X.rows()).array();

			};
			if (zero_indices.size() > 0) { zero_variance(); }
			if (non_zero_indices.size() > 0) { non_zero_variance(); }
		}
		void expectations(const TMatrix& mean, const TMatrix& variance_) override { return; }
	};

}
#endif


//class Matern32 : public Stationary {
//
//public:
//	Matern32() : Stationary() {};
//	Matern32(const Matern32& kernel) : Stationary(kernel) {}
//	Matern32(const double& length_scale, const double& variance) : Stationary(length_scale, variance) {}
//	Matern32(TVector& length_scale, const double& variance) : Stationary(length_scale, variance) {}
//	Matern32(const Parameter<TVector>& length_scale, const Parameter<double>& variance) : Stationary(length_scale, variance) {}
//	//
//	Matern32(const TVector& length_scale, const double& variance) : Stationary(length_scale, variance) {}
//	Matern32(const TVector& length_scale, double& variance) : Stationary(length_scale, variance) {}
//	//
//
//	const TMatrix K(const TMatrix& X1, const TMatrix& X2) override {
//		TMatrix R(X1.rows(), X2.rows());
//		TVector tmp;
//		if (length_scale.size() != X1.cols() && length_scale.size() == 1)
//		{   // Expand lengthscale dimensions
//			tmp = TVector::Constant(X1.cols(), 1, length_scale.value()(0));
//		}
//		else {
//			ARD = true;
//			tmp = length_scale.value();
//		}
//		const TMatrix X1sc = X1.array().rowwise() / tmp.transpose().array();
//		const TMatrix X2sc = X2.array().rowwise() / tmp.transpose().array();
//		euclidean_distance(X1sc, X2sc, R, false);
//		R *= sqrt(3);
//		return ( (1 + R.array()) * exp(-R.array())).matrix();
//	}
//	const TMatrix K(const TMatrix& X1, const TMatrix& X2, TMatrix& R1) override {
//		// sqrt euclidean distance R1
//		TVector tmp;
//		if (length_scale.size() != X1.cols() && length_scale.size() == 1)
//		{   // Expand lengthscale dimensions
//			tmp = TVector::Constant(X1.cols(), 1, length_scale.value()(0));
//		}
//		else {
//			ARD = true;
//			tmp = length_scale.value();
//		}
//		const TMatrix X1sc = X1.array().rowwise() / tmp.transpose().array();
//		const TMatrix X2sc = X2.array().rowwise() / tmp.transpose().array();
//		euclidean_distance(X1sc, X2sc, R1, false);
//		TMatrix R = R1.array() * sqrt(3);
//		return ( (1 + R.array()) * exp(-R.array())).matrix();
//	}
//	const TMatrix K(const TMatrix& X1, const TMatrix& X2, const double& likelihood_variance) override {
//		TMatrix R(X1.rows(), X2.rows());
//		TMatrix noise = TMatrix::Identity(X1.rows(), X2.rows()).array() * likelihood_variance;
//		TVector tmp;
//		if (length_scale.size() != X1.cols() && length_scale.size() == 1)
//		{   // Expand lengthscale dimensions
//			tmp = TVector::Constant(X1.cols(), 1, length_scale.value()(0));
//		}
//		else {
//			ARD = true;
//			tmp = length_scale.value();
//		}
//		const TMatrix X1sc = X1.array().rowwise() / tmp.transpose().array();
//		const TMatrix X2sc = X2.array().rowwise() / tmp.transpose().array();
//		euclidean_distance(X1sc, X2sc, R, false);
//		R *= sqrt(3);
//		return (variance.value() * (1 + R.array()) * exp(-R.array())).matrix() + noise;
//	}
//	const TMatrix K(const TMatrix& X1, const TMatrix& X2, const double& likelihood_variance, const Eigen::Index idx) {
//		TMatrix R(X1.rows(), X2.rows());
//		TMatrix noise = TMatrix::Identity(X1.rows(), X2.rows()).array() * likelihood_variance;
//		const TMatrix X1sc = X1.array() / length_scale.value()[idx];
//		const TMatrix X2sc = X2.array() / length_scale.value()[idx];
//		euclidean_distance(X1sc, X2sc, R, false);
//		R *= sqrt(3);
//		return (variance.value() * (1 + R.array()) * exp(-R.array())).matrix() + noise;
//	}
//
//	void IJ(TMatrix& I, TMatrix& J, const TVector& mean, const TVector& variance, const TMatrix& X, const Eigen::Index& idx) override { return; }
//	void expectations(const TMatrix& mean, const TMatrix& variance) override { return; }
//
//	void fod(const TMatrix& X, std::vector<TMatrix>& grad) override {
//		return;
//	}
//
//	void gradients(const TMatrix& X, const TMatrix& dNLL, const TMatrix& R, const TMatrix& K, std::vector<double>& grad) override
//	{
//		if (!(*length_scale.is_fixed)) {
//			TMatrix dK_dR = (variance.value() * (-3 * R.array()) * exp(-(R * sqrt(3)).array())).matrix();
//			std::vector<TMatrix> dK;
//			pdist(X, X, dK);
//			TMatrix tmp = R.cwiseInverse().cwiseProduct(dK_dR.cwiseProduct(dNLL));
//			tmp.diagonal().array() = 0.0;
//			dK_dlengthscale(dK, grad, tmp, dK_dR, dNLL, R);
//		}
//		if (!(*variance.is_fixed)) { dK_dvariance(K, dNLL, grad); }
//
//	}
//};

