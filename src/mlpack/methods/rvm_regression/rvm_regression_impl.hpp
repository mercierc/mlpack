/**
 * @file rvmr.cpp
 * @author Clement Mercier
 *
 * Implementation of the Relevance Vector Machine.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RVM_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_RVM_REGRESSION_IMPL_HPP

#include "rvm_regression.hpp"

using namespace mlpack;

template<typename KernelType>
RVMRegression<KernelType>::RVMRegression(const KernelType& kernel,
                                         const bool centerData,
                                         const bool scaleData) :

  kernel(kernel),
  centerData(centerData),
  scaleData(scaleData),
  ardRegression(false) 
  {/*Nothing to do */}

  template <typename KernelType>
  RVMRegression<KernelType>::RVMRegression(const bool centerData,
                                           const bool scaleData) :
    centerData(centerData),
    scaleData(scaleData),
    kernel(kernel::LinearKernel()),
    ardRegression(true) 
  {/*Nothing to do*/}

template<typename KernelType>
void RVMRegression<KernelType>::Train(const arma::mat& data,
                                      const arma::rowvec& responses)
{
  arma::mat phi;
  arma::rowvec t;

  // Manage the kernel.
  if (ardRegression == false)
  {
    // We must keep the original training data for future predictions.
    phi = data;
    applyKernel(data, data, phi);

    //Preprocess the data. Center and scaleData.
    preprocess_data(phi,
		    responses,
		    centerData,
		    scaleData,
		    phi,
		    t,
		    data_offset,
		    data_scale,
		    responses_offset);
  }

  else
  {
    //Preprocess the data. Center and scaleData.
    preprocess_data(data,
		    responses,
		    centerData,
		    scaleData,
		    phi,
		    t,
		    data_offset,
		    data_scale,
		    responses_offset);
  }

  unsigned short p = phi.n_rows, n = phi.n_cols;
  // Initialize the hyperparameters and
  // begin with an infinitely broad prior.
  alpha_threshold = 1e4;
  alpha = arma::ones<arma::rowvec>(p) * 1e-6;
  beta =  1 / (arma::var(t) * 0.1);

  // Loop variables.
  double tol = 1e-5;
  double L = 1.0;
  double crit = 1.0;
  unsigned short nIterMax = 50;
  unsigned short i = 0;
  unsigned short ind_act;

  arma::rowvec gammai = arma::zeros<arma::rowvec>(p);
  arma::mat matA;
  arma::rowvec temp(n);
  arma::mat subPhi;
  // Initiaze a vector of all the indices from the first
  // to the last point.
  arma::uvec allCols(n);
  for (size_t i=0; i < n; i++) {allCols(i) = i;}

  while ((crit > tol) && (i < nIterMax))
    {
      crit = -L;
      activeSet = find(alpha < alpha_threshold);
      // Prune out the inactive basis function. This procedure speeds up
      // the algorithm.
      subPhi = phi.submat(activeSet, allCols);

      // Compute the posterior statistics.
      matA = diagmat(alpha.elem(activeSet));
      matCovariance = inv(matA
				+ (subPhi
				   * subPhi.t())
				* beta);

      omega = (matCovariance * subPhi * t.t()) * beta;

      // Update the alpha_i.
      for (size_t k = 0; k < activeSet.size(); k++)
	{
	  ind_act = activeSet[k];
	  gammai(ind_act) = 1 - matCovariance(k, k) * alpha(ind_act);

	  alpha(ind_act) = gammai(ind_act)
	    / (omega(k) * omega(k));
	}

      // Update beta.
      temp = t -  omega.t() * subPhi;
      beta = (n - sum(gammai.elem(activeSet))) / dot(temp, temp);

      // Comptute the stopping criterion.
      L = norm(omega);
      crit = abs(crit + L) / L;
      i++;
    }
}

template<typename KernelType>
void RVMRegression<KernelType>::Predict(const arma::mat& points,
                                        arma::rowvec& predictions) const
{
  arma::mat X;
  // Manage the kernel.
  if (ardRegression == false)
    applyKernel(phi, points, X);
  else
    X = points;

  arma::uvec allCols(X.n_cols);
  for (size_t i=0; i < X.n_cols; i++) {allCols[i] = i;}

  // Center and scaleData the points before applying the model.
  X.each_col() -= data_offset;
  X.each_col() /= data_scale;
  predictions = omega.t() * X.submat(activeSet, allCols)
                + responses_offset;
}

template<typename KernelType>
void RVMRegression<KernelType>::Predict(const arma::mat& points,
                                        arma::rowvec& predictions,
                                        arma::rowvec& std) const
{
  arma::mat X;
  // Manage the kernel.
  if (ardRegression == false)
    applyKernel(phi, points, X);
  else
    X = points;

  arma::uvec allCols(X.n_cols);
  for (size_t i=0; i < X.n_cols; i++) {allCols[i] = i;}
  
  // Center and scaleData the points before applying the model.
  X.each_col() -= data_offset;
  X.each_col() /= data_scale;
  predictions = omega.t() * X.submat(activeSet, allCols)
                + responses_offset;

  // Comptute the standard deviations
  arma::mat O(X.n_cols, X.n_cols);
  O = X.submat(activeSet, allCols).t()
    * matCovariance
    * X.submat(activeSet, allCols);
  std = sqrt(diagvec(1 / beta + O).t());
}

template<typename KernelType>
double RVMRegression<KernelType>::Rmse(const arma::mat& data,
                                       const arma::rowvec& responses) const
{
  arma::rowvec predictions;
  Predict(data, predictions);
  return sqrt(
	      mean(
		  square(responses - predictions)));
}

template<typename KernelType>
arma::vec RVMRegression<KernelType>::getCoefs() const
{
  // Get the size of the full solution with the offset.
  arma::colvec coefs = arma::zeros<arma::colvec>(data_offset.size());
  // omega[i] = 0 for the inactive basis functions

  // Now reconstruct the full solution.
  for (size_t i=0; i < activeSet.size(); i++)
    {
      coefs[activeSet[i]] = omega[i];
    }
  return coefs;
}

template<typename KernelType>
void RVMRegression<KernelType>::applyKernel(const arma::mat& X,
                                            const arma::mat& Y,
                                            arma::mat& gramMatrix) const {

  // Check if the dimensions are consistent.
  if (X.n_rows != Y.n_rows)
    {
      std::cout << "Error gramm" << std::endl;
      throw std::invalid_argument("Number of features not consistent");
    }

  gramMatrix = arma::zeros<arma::mat>(X.n_cols, Y.n_cols);
  arma::colvec xi = arma::zeros<arma::colvec>(X.n_rows);
  arma::colvec yj = arma::zeros<arma::colvec>(X.n_rows);

  for (size_t i = 0; i < X.n_cols; i++)
    {
      xi = X.col(i);
      for (size_t j = 0; j < Y.n_cols; j++)
	{
	  yj = Y.col(j);
	  gramMatrix(i, j) = kernel.Evaluate(xi, yj);
	}
    }
}

#endif
