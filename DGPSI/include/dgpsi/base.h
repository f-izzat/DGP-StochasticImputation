#ifndef BASE_H
#define BASE_H
#include <iostream>
#include <string>
#include <vector>
#include <variant>
#include <random>
#include <unsupported/Eigen/SpecialFunctions>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <ThreadPool/thread_pool.hpp>

#include <pcg/pcg_random.hpp>
pcg_extras::seed_seq_from<std::random_device> seed_source;
static pcg64 rng(seed_source);

static const double PI = std::atan(1) * 4;
static const double NaN = std::nan("1");

using std::tuple;
using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;
using std::make_unique;

// Generator for Random Matrix


/*
* Types Doc:
*	TMatrix			  : Matrix Type							   : Usage - [Global] i.e. Model inputs
*	TVector/TRVector  : Vector Type							   : Usage - [Global] i.e. Kernel length scale
*	BoolVector		  : Boolean Vector						   : Usage - [Global] i.e. Missing indices in Deep Model Layer outputs
*	TLLT			  : Cholesky L*L^T						   : Usage - [Base] i.e. Gaussian Process Model train/predict
*	MatrixPair		  : (TMatrix, TMatrix)					   : Usage - [Base] i.e. Output of GP Model prediction
*	VectorPair		  : (TVector, TVector)					   : Usage - 
*	MatrixPairVariant : MatrixPair or (MatrixPair, MatrixPair) : Usage - [Deep] i.e. Layer(Layer)
*	
*/


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> TMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TRMatrix;
typedef Eigen::VectorXd TVector;
typedef Eigen::RowVectorXd TRVector;
typedef Eigen::Vector<bool, Eigen::Dynamic> BoolVector;
typedef Eigen::LLT<TMatrix> TLLT;
typedef Eigen::LDLT<TMatrix> TLDLT;
typedef Eigen::FullPivLU<TMatrix> TFPLU;

typedef std::pair<TMatrix, TMatrix> MatrixPair;
typedef std::pair<TVector, TVector> VectorPair;
typedef std::variant<MatrixPair, std::pair<MatrixPair, MatrixPair>> MatrixPairVariant;


typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> BoolMatrix;
typedef std::tuple<TMatrix> MatrixTuple;
typedef std::tuple<MatrixPair> MatrixPairTuple;
typedef std::variant<TMatrix, MatrixPair> MatrixVariant;
#endif
