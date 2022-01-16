#ifndef UTILITIES_H
#define UTILITIES_H

#include <fstream>
#include <filesystem>
#include <numeric>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <dgpsi/base.h>

namespace dgpsi::utilities {

    namespace metrics {
        // Scaler
        struct StandardScaler {
            StandardScaler() = default;
            void scale(TMatrix& X) {
                nrows = X.rows(); ncols = X.cols();
                mean = X.colwise().mean();
                TMatrix Xmean = X.rowwise() - X.colwise().mean();
                std_dev = (((Xmean).array().square().colwise().sum()) / ((Xmean).rows())).sqrt();
                X = (Xmean).array().rowwise() / (((Xmean).array().square().colwise().sum()) / ((Xmean).rows())).sqrt();
            }
            TMatrix rescale(const TMatrix& X) {
                TMatrix Z(nrows, ncols);
                Z = X.array().rowwise() * std_dev.transpose().array();
                Z.array().rowwise() += mean.transpose().array();
                return Z;
            }
            Eigen::Index nrows;
            Eigen::Index ncols;
            TVector mean;
            TVector std_dev;
        };        
        void minmax(TMatrix& X) {
            TMatrix num = X.rowwise() - X.colwise().minCoeff();
            TVector den = X.colwise().maxCoeff() - X.colwise().minCoeff();
            X = num.array().rowwise() / den.array().transpose();
        }
        void scale_to_range(TMatrix& X, const TVector& lower, TVector& upper) {
            minmax(X);
            upper.array() -= lower.array();
            for (Eigen::Index c = 0; c < X.cols(); ++c) {
                X.col(c).array() *= upper(c);
                X.col(c).array() += lower(c);
            }
        }
        void standardize(TMatrix& X) {
            TMatrix Xmean = X.rowwise() - X.colwise().mean();
            X = (Xmean).array().rowwise() / (((Xmean).array().square().colwise().sum()) / ((Xmean).rows())).sqrt();
        }
        // Error
        double rmse(const TMatrix& Ytrue, const TMatrix& Ypred, bool normalized = false) {
            double res = sqrt((Ypred - Ytrue).array().square().sum() / Ytrue.rows());
            if (normalized) return res / (Ytrue.maxCoeff() - Ytrue.minCoeff());
            else return res;
        }
        double r2_score(const Eigen::Ref<TVector> ytrue, const Eigen::Ref<TVector> ypred) {
            double res = (ytrue - ypred).array().square().sum();
            double tss = (ytrue.array() - ytrue.array().mean()).square().sum();
            return 1.0 - (res / tss);
        }
        // Distance
        void euclidean_distance(const TMatrix& X1, const TMatrix& X2, TMatrix& D, bool squared = false) {
            // Compute squared euclidean distance |X1(i) - X2(j)^T|^2 = |X1(i)|^2 + |X2(j)|^2 - (2 * X1(i)^T X2(j)) by default
            TMatrix D_ = ((-2.0 * (X1 * X2.transpose())).colwise()
                + X1.rowwise().squaredNorm()).rowwise()
                + X2.rowwise().squaredNorm().transpose();

            if (!squared) {
                if ((D_.array() < 0.0).any()) {
                    D_ = abs(D_.array());
                }
                D = (D_.array().sqrt()).matrix();
            }
            else {
                D = D_;
            }
        }
        void pdist(const TVector& X1, const TVector& X2, TMatrix& D) {
            // Pairwise distance between two Vectors
            D.resize(X1.rows(), X2.rows());
            for (int i = 0; i < X1.rows(); i++) {
                for (int j = 0; j < X2.rows(); j++) {
                    D(i, j) = X1(i) - X2(j);
                }
            }
        }
        void pdist(const TMatrix& X1, const TMatrix& X2, std::vector<TMatrix>& D, bool squared = true) {
            // Pairwise distance between two each column/dimension of 2 Matrices
            for (int i = 0; i < X1.cols(); i++) {
                TMatrix tmp(X1.rows(), X2.rows());
                for (int j = 0; j < X1.rows(); j++) {
                    for (int k = 0; k < X2.rows(); k++) {
                        tmp(j, k) = X1.col(i)(j) - X2.col(i)(k);
                    }
                }
                if (squared) { D.push_back(pow(tmp.array(), 2)); }
                else { D.push_back(tmp); }

            }
        }
        void abspdist(const TMatrix& X1, const TMatrix& X2, std::vector<TMatrix>& D) {
            // Pairwise distance between two each column/dimension of 2 Matrices
            for (int i = 0; i < X1.cols(); i++) {
                TMatrix tmp(X1.rows(), X2.rows());
                for (int j = 0; j < X1.rows(); j++) {
                    for (int k = 0; k < X2.rows(); k++) {
                        tmp(j, k) = abs(X1.col(i)(j) - X2.col(i)(k));
                    }
                }
                D.push_back(tmp);

            }
        }
        TMatrix pnorm(const TMatrix& X) {
            // 0.5*(1+erf(x/sqrt(2)))
            TMatrix P = Eigen::erf(X.array() / sqrt(2.0));
            return 0.5 * (1.0 + P.array());
        }
    }

    namespace operations {
        
        const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, "\t", "\n");
        template <typename Derived>
        void write_data(std::string name, const Eigen::MatrixBase<Derived>& matrix)
        {
            std::ofstream file(name.c_str());
            file << matrix.format(CSVFormat);
        }
        
        // Print
        void cursor_position(int x,int y)    
        {
            printf("%c[%d;%df",0x1B,y,x);
        }        

        // Eigen Indexing
        std::vector<int> arange(int a, int b) {
            if (b < a) { throw std::runtime_error("b < a"); }
            std::vector<int> range_;
            for (int i = a; i < b; ++i) {
                range_.push_back(i);
            }
            return range_;
        }
        template<typename Func>
        struct lambda_as_visitor_wrapper : Func {
            lambda_as_visitor_wrapper(const Func& f) : Func(f) {}
            template<typename S, typename I>
            void init(const S& v, I i, I j) { return Func::operator()(v, i, j); }
        };
        template<typename Mat, typename Func>
        void visit_lambda(const Mat& m, const Func& f)
        {
            lambda_as_visitor_wrapper<Func> visitor(f);
            m.visit(visitor);
        }
        template<typename TBool>
        TBool get_missing_index(const TMatrix& X1);
        template<>
        BoolVector get_missing_index(const TMatrix& X1) {
            BoolVector missing(X1.rows());
            visit_lambda(X1, [&missing, &X1](double v, int i, int j) {
                if (j == 0) {
                    if (X1.row(i).array().isNaN().any()) { missing(i) = true; return; }
                    else { missing(i) = false; }
                }
                else { return; }
                }
            );
            return missing;
        }
        template<>
        BoolMatrix get_missing_index(const TMatrix& X1) {
            BoolMatrix missing(X1.rows(), X1.cols());
            visit_lambda(X1, [&missing](double v, int i, int j)
                {
                    if (std::isnan(v)) { missing(i, j) = true; }
                    else { missing(i, j) = false; }
                });
            return missing;
        }
        void mask_matrix_in_place(TMatrix& X2, const TMatrix& X1, const BoolVector& mask, const bool& inverse, const int& axis) {
            /*
            * Masks a matrix X1 to get X2:
            *   -> X2 = X1[mask, :] / if (inverse) -> X2 = X1[~mask, :] (axis = 0 Row Wise)
            *   -> X2 = X1[:, mask] / if (inverse) -> X2 = X1[:, ~mask] (axis = 1 Col Wise)
            */
            int i, j;
            if (axis == 0) {
                // Row Wise
                if (inverse) {
                    X2.resize(X1.rows() - mask.count(), X1.cols());
                    for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                        if (mask(i)) { continue; }
                        else { X2.row(j) = X1.row(i); ++j; }
                    }
                }
                else {
                    X2.resize(mask.count(), X1.cols());
                    for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                        if (mask(i)) { X2.row(j) = X1.row(i); ++j; }
                        else { continue; }
                    }
                }
            }
            else if (axis == 1) {
                // Column Wise
                if (inverse) {
                    X2.resize(X1.rows(), X1.cols() - mask.count());
                    for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                        if (mask(i)) { continue; }
                        else { X2.col(j) = X1.col(i); ++j; }
                    }
                }
                else {
                    X2.resize(X1.rows(), mask.count());
                    for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                        if (mask(i)) { X2.col(j) = X1.col(i); ++j; }
                        else { continue; }
                    }
                }
            }
            else { throw std::runtime_error("Axis Out of Bounds"); }
        }
        TMatrix mask_matrix(const TMatrix& X, const BoolVector& mask, const bool& inverse, const int& axis) {
            /*
            * Masks a matrix X to get X2:
            *   -> X2 = X[mask, :] / if (inverse) -> X2 = X[~mask, :] (axis = 0 Row Wise)
            *   -> X2 = X[:, mask] / if (inverse) -> X2 = X[:, ~mask] (axis = 1 Col Wise)
            */
            int i, j;
            TMatrix XM;
            if (axis == 0) {
                // Row Wise
                if (inverse) {
                    XM.resize(X.rows() - mask.count(), X.cols());
                    for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                        if (mask(i)) { continue; }
                        else { XM.row(j) = X.row(i); ++j; }
                    }
                }
                else {
                    XM.resize(mask.count(), X.cols());
                    for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                        if (mask(i)) { XM.row(j) = X.row(i); ++j; }
                        else { continue; }
                    }
                }
            }
            else if (axis == 1) {
                // Column Wise
                if (inverse) {
                    XM.resize(X.rows(), X.cols() - mask.count());
                    for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                        if (mask(i)) { continue; }
                        else { XM.col(j) = X.col(i); ++j; }
                    }
                }
                else {
                    XM.resize(X.rows(), mask.count());
                    for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                        if (mask(i)) { XM.col(j) = X.col(i); ++j; }
                        else { continue; }
                    }
                }
            }
            else { throw std::runtime_error("Axis Out of Bounds"); }
            return XM;
        }
        // Eigen Tensor
        // namespace tensor {
        //     // Taken from : https://stackoverflow.com/questions/48795789/eigen-unsupported-tensor-to-eigen-matrix
        //     template<typename T>
        //     using  MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        //     template<typename T>
        //     using  VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        //     /* Convert Eigen::Tensor --> Eigen::Matrix */

        //      // Evaluates tensor expressions if needed
        //     template<typename T, typename Device = Eigen::DefaultDevice>
        //     auto asEval(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>& expr, // An Eigen::TensorBase object (Tensor, TensorMap, TensorExpr... )
        //         const Device& device = Device()                            // Override to evaluate on another device, e.g. thread pool or gpu.
        //     ) {
        //         using Evaluator = Eigen::TensorEvaluator<const Eigen::TensorForcedEvalOp<const T>, Device>;
        //         Eigen::TensorForcedEvalOp<const T> eval = expr.eval();
        //         Evaluator                          tensor(eval, device);
        //         tensor.evalSubExprsIfNeeded(nullptr);
        //         return tensor;
        //     }

        //     // Converts any Eigen::Tensor (or expression) to an Eigen::Matrix with shape rows/cols
        //     template<typename T, typename sizeType, typename Device = Eigen::DefaultDevice>
        //     auto MatrixCast(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>& expr, const sizeType rows, const sizeType cols, const Device& device = Device()) {
        //         auto tensor = asEval(expr, device);
        //         using Scalar = typename Eigen::internal::remove_const<typename decltype(tensor)::Scalar>::type;
        //         return static_cast<MatrixType<Scalar>>(Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols));
        //     }

        //     // Converts any Eigen::Tensor (or expression) to an Eigen::Vector with the same size
        //     template<typename T, typename Device = Eigen::DefaultDevice>
        //     auto VectorCast(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>& expr, const Device& device = Device()) {
        //         auto tensor = asEval(expr, device);
        //         auto size = Eigen::internal::array_prod(tensor.dimensions());
        //         using Scalar = typename Eigen::internal::remove_const<typename decltype(tensor)::Scalar>::type;
        //         return static_cast<VectorType<Scalar>>(Eigen::Map<const VectorType<Scalar>>(tensor.data(), size));
        //     }

        //     // View an existing Eigen::Tensor as an Eigen::Map<Eigen::Matrix>
        //     template<typename Scalar, auto rank, typename sizeType>
        //     auto MatrixMap(const Eigen::Tensor<Scalar, rank>& tensor, const sizeType rows, const sizeType cols) {
        //         return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
        //     }

        //     // View an existing Eigen::Tensor of rank 2 as an Eigen::Map<Eigen::Matrix>
        //     // Rows/Cols are determined from the matrix
        //     template<typename Scalar>
        //     auto MatrixMap(const Eigen::Tensor<Scalar, 2>& tensor) {
        //         return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
        //     }

        //     // View an existing Eigen::Tensor of rank 1 as an Eigen::Map<Eigen::Vector>
        //     // Rows is the same as the size of the tensor. 
        //     template<typename Scalar, auto rank>
        //     auto VectorMap(const Eigen::Tensor<Scalar, rank>& tensor) {
        //         return Eigen::Map<const VectorType<Scalar>>(tensor.data(), tensor.size());
        //     }

        //     /* Convert Eigen::Matrix--> Eigen::Tensor */

        //      // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
        //      // with dimensions specified in std::array
        //     template<typename Derived, typename T, auto rank>
        //     Eigen::Tensor<typename Derived::Scalar, rank>
        //         TensorCast(const Eigen::EigenBase<Derived>& matrix, const std::array<T, rank>& dims) {
        //         return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>
        //             (matrix.derived().eval().data(), dims);
        //     }

        //     // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
        //     // with dimensions specified in Eigen::DSizes
        //     template<typename Derived, typename T, auto rank>
        //     Eigen::Tensor<typename Derived::Scalar, rank>
        //         TensorCast(const Eigen::EigenBase<Derived>& matrix, const Eigen::DSizes<T, rank>& dims) {
        //         return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>
        //             (matrix.derived().eval().data(), dims);
        //     }

        //     // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
        //     // with dimensions as variadic arguments
        //     template<typename Derived, typename... Dims>
        //     auto TensorCast(const Eigen::EigenBase<Derived>& matrix, const Dims... dims) {
        //         static_assert(sizeof...(Dims) > 0, "TensorCast: sizeof... (Dims) must be larger than 0");
        //         return TensorCast(matrix, std::array<Eigen::Index, sizeof...(Dims)>{dims...});
        //     }

        //     // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
        //     // with dimensions directly as arguments in a variadic template
        //     template<typename Derived>
        //     auto TensorCast(const Eigen::EigenBase<Derived>& matrix) {
        //         if constexpr (Derived::ColsAtCompileTime == 1 or Derived::RowsAtCompileTime == 1) {
        //             return TensorCast(matrix, matrix.size());
        //         }
        //         else {
        //             return TensorCast(matrix, matrix.rows(), matrix.cols());
        //         }
        //     }

        //     // View an existing Eigen::Matrix as Eigen::TensorMap
        //     // with dimensions specified in std::array
        //     template<typename Derived, auto rank>
        //     auto TensorMap(const Eigen::PlainObjectBase<Derived>& matrix, const std::array<long, rank>& dims) {
        //         return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>(matrix.derived().data(), dims);
        //     }

        //     // View an existing Eigen::Matrix as Eigen::TensorMap
        //     // with dimensions as variadic arguments
        //     template<typename Derived, typename... Dims>
        //     auto TensorMap(const Eigen::PlainObjectBase<Derived>& matrix, const Dims... dims) {
        //         return TensorMap(matrix, std::array<long, static_cast<int>(sizeof...(Dims))>{dims...});
        //     }

        //     // View an existing Eigen::Matrix as Eigen::TensorMap
        //     // with dimensions determined automatically from the given matrix
        //     template<typename Derived>
        //     auto TensorMap(const Eigen::PlainObjectBase<Derived>& matrix) {
        //         if constexpr (Derived::ColsAtCompileTime == 1 or Derived::RowsAtCompileTime == 1) {
        //             return TensorMap(matrix, matrix.size());
        //         }
        //         else {
        //             return TensorMap(matrix, matrix.rows(), matrix.cols());
        //         }
        //     }

        // }
    }

    namespace statistics {
        // Random Utilities
        double random_uniform(double a, double b) {
            auto seed = std::random_device{}();
            std::mt19937 gen_primitive(seed);
            std::uniform_real_distribution<> uniform_dist(a, b);
            return uniform_dist(gen_primitive);
        }
        double random_uniform() {
            auto seed = std::random_device{}();
            std::mt19937 gen_primitive(seed);
            std::uniform_real_distribution<> uniform_dist;
            return uniform_dist(gen_primitive);
        }
        TMatrix gen_normal_matrix(Eigen::Index n_rows, Eigen::Index n_samples) {
            std::normal_distribution<double> normal_sampler{ 0,1 };
            TMatrix norm_matrix = TMatrix::Zero(n_rows, n_samples);
            auto seed = std::random_device{}();
            std::mt19937 gen_primitive(seed);
            operations::visit_lambda(norm_matrix, 
                [&norm_matrix, &normal_sampler, &gen_primitive](double v, int i, int j)                
                {norm_matrix(i, j) = normal_sampler(gen_primitive); });
            return norm_matrix;
        }
        // Taken from: https://stackoverflow.com/a/40245513
        struct MVN
        {
            MVN(TMatrix const& covar) : MVN(TVector::Zero(covar.rows()), covar) {}
            MVN(TVector const& mean, TMatrix const& covar) : mean(mean)
            {
                Eigen::SelfAdjointEigenSolver<TMatrix> eigenSolver(covar);
                transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
            }
            TVector operator()() const
            {
                // auto seed = std::random_device{}();
                // std::default_random_engine gen_primitive(seed);
                thread_local std::mt19937 gen_primitive(std::random_device{}());
                std::normal_distribution<> dist;
                // NumpyNormal generator;
                return mean + transform * TVector{ mean.size() }.unaryExpr([&](auto x) {
                    return dist(gen_primitive);
                    // return generator.getDouble();
                    });
            }
            TVector mean;
            TMatrix transform;
        };
    }

    namespace kernelpca {
        // Follows sklearn
        class KernelPCA {
        private:
            TMatrix compute_kernel(const TMatrix X0) {
                TMatrix _K(X0.rows(), X0.rows());
                gamma = 1.0 / X0.cols();
                // Sigmoid: k(x, y) = \tanh( \gamma x^\top y + c_0)
                //if (kernel == "sigmoid") {
                _K = X0 * X0.transpose();
                _K.array() *= gamma;
                _K.array() += constant;
                return tanh(_K.array());
                //}
                // Add more kernels
            }
            void center_kernel(TMatrix& K) {
                //  https://github.com/scikit-learn/scikit-learn/blob/e7fb5b8c8dd2cd4d7ccd6f9f9ad6c1d206c43a33/sklearn/preprocessing/_data.py#L2132
                TRVector K_fit_rows = (K.colwise().sum()).array() / K.rows();
                double K_fit_all = K_fit_rows.sum() / K.rows();
                TVector K_pred_cols = (K.rowwise().sum()).array() / K_fit_rows.size();
                K.array().rowwise() -= K_fit_rows.array();
                K.array().colwise() -= K_pred_cols.array();
                K.array() += K_fit_all;
            }

        public:
            KernelPCA(Eigen::Index n_components, std::string kernel) : n_components(n_components), kernel(kernel) {}

            TMatrix transform(const TMatrix& X) {
                gamma = 1.0 / X.cols();
                TMatrix K = compute_kernel(X);
                center_kernel(K);
                // SelfAdjointEigenSolver outputs in ascending order
                Eigen::SelfAdjointEigenSolver<TMatrix> eigsolve(K);
                TVector eigenvalues = eigsolve.eigenvalues();
                TMatrix eigenvectors = eigsolve.eigenvectors();
                // Select eigenvalues/vectors and sort in descending order
                std::vector<Eigen::Index> indices(n_components);
                std::iota(indices.begin(), indices.end(), K.rows() - n_components);
                lambda = eigenvalues(indices).reverse();
                alpha = eigenvectors(Eigen::all, indices).rowwise().reverse();
                // Remove zero eigenvalues
                BoolVector zero_index = lambda.array() == 0.0;
                if (zero_index.array().any()) {
                    TVector tmp = operations::mask_matrix(lambda, zero_index, true, 0);
                    lambda = tmp;
                    TMatrix tmp_a = operations::mask_matrix(alpha, zero_index, true, 0);
                    alpha = tmp_a;
                }

                // Multiply alpha by -1?
                return alpha.array().rowwise() * sqrt(lambda.array()).transpose();
            }


        public:
            Eigen::Index n_components;
            std::string kernel;
            TVector lambda;
            TMatrix alpha;
            double gamma = 0.001;
            double constant = 1.0;

        };

    }

}


#endif