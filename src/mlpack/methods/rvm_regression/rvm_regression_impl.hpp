/**
 * @file rvm_regression_impl.cpp
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
using namespace mlpack::regression;

template<typename KernelType>
RVMRegression<KernelType>::RVMRegression(const KernelType& kernel,
                                         const bool centerData,
                                         const bool scaleData,
                                         const bool ard) :

  kernel(kernel),
  centerData(centerData),
  scaleData(scaleData),
  ard(ard) 
  {/*Nothing to do */}

template <typename KernelType>
RVMRegression<KernelType>::RVMRegression(const bool centerData,
                                         const bool scaleData) :
    centerData(centerData),
    scaleData(scaleData),
    kernel(kernel::LinearKernel()),
    ard(false) 
  {/*Nothing to do*/}

template<typename KernelType>
void RVMRegression<KernelType>::Train(const arma::mat& data,
                                      const arma::rowvec& responses)
{
  arma::mat phi = data;
  arma::rowvec t;
  
  // Preprocess the data. Center and scaleData.
  responsesOffset = CenterScaleData(phi, 
                                    responses, 
                                    centerData, 
                                    scaleData, 
                                    phi, 
                                    t,
		                            dataOffset, 
                                    dataScale);

  // When ard is set to true the kernel is ignored and we work in the original 
  // input space.
  if (!ard)
  {
    relevantVectors = phi;
    applyKernel(data, data, phi);
    std::cout << phi.n_rows << " - " << phi.n_cols << std::endl;
   }
  // Initialize the hyperparameters and begin with an infinitely broad prior.
  alpha_threshold = 1e4;
  alpha = arma::ones<arma::rowvec>(phi.n_rows) * 1e-6;
  beta =  1 / arma::var(t, 1);
  
  // Loop variables.
  double tol = 1e-5;
  double normOmega = 1.0;
  double crit = 1.0;
  unsigned short nIterMax = 50;
  unsigned short i = 0;
  unsigned short ind_act;

  arma::rowvec gammai = arma::zeros<arma::rowvec>(phi.n_rows);
  arma::mat subPhi;
  // Initiaze a vector of all the indices from the first
  // to the last point.
  arma::uvec allCols(phi.n_cols);

  for (size_t i = 0; i < phi.n_cols; ++i) { allCols(i) = i; }

  while ((crit > tol) && (i < nIterMax))
  {
    crit = -normOmega;
    activeSet = find(alpha < alpha_threshold);

    // Prune out the inactive basis functions. This procedure speeds up
    // the algorithm.
    subPhi = arma::mat(phi.submat(activeSet, allCols));
    // matCovariance = inv(diagmat(alpha(activeSet)) + subPhi * subPhi.t() * beta);
    matCovariance  = subPhi * subPhi.t() * beta;
    matCovariance.diag() += alpha(activeSet);
    matCovariance = inv(matCovariance);
    std::cout << matCovariance.submat(0, 0, 3, 3) << std::endl;
    // Compute the solution for al the active basis functions.
    // omega = solve(diagmat(alpha.elem(activeSet) / beta) + subPhi * subPhi.t(),
    //               subPhi * t.t());
    omega = matCovariance * subPhi * t.t() * beta;
    // FIX ME std::cout << omega.head(3) << std::endl;
    // Update alpha.
    gammai = 1 - matCovariance.diag().t() % alpha(activeSet).t();
    alpha(activeSet) = gammai / (omega % omega).t(); 
    
    // Update beta.
    const arma::rowvec temp = t -  omega.t() * subPhi;
    beta = (phi.n_cols - sum(gammai)) / dot(temp, temp);

    // Comptute the stopping criterion.
    normOmega = norm(omega);
    crit = std::abs(crit + normOmega) / normOmega;
    i++;
  }
  // Compute the covariance matrice for the uncertaities later.
  std::cout << "End of training " << std::endl;

  if (!ard)
  {
    arma::uvec allRows(relevantVectors.n_rows);
    for (size_t i = 0; i < relevantVectors.n_rows; ++i) { allRows(i) = i; }
    relevantVectors = relevantVectors.submat(allRows, activeSet);
  }
}

template<typename KernelType>
void RVMRegression<KernelType>::Predict(const arma::mat& points,
                                        arma::rowvec& predictions) const
{
  arma::mat matX;
  // Manage the kernel.
  if (!ard)
  {
    applyKernel(relevantVectors, points, matX);
  }
  else
  {
    arma::uvec allCols(points.n_cols);
    for ( size_t i = 0; i < allCols.n_elem; ++i) { allCols(i) = i; }
    matX = arma::mat(points.submat(activeSet, allCols));
  }
  // Center and scaleData the points before applying the model.
  predictions = omega.t() * matX + responsesOffset;
}

template<typename KernelType>
void RVMRegression<KernelType>::Predict(const arma::mat& points,
                                        arma::rowvec& predictions,
                                        arma::rowvec& std) const
{
  arma::mat matX;
  // Manage the kernel.
  if (!ard)
    applyKernel(relevantVectors, points, matX);
  else
    matX = points;

  arma::uvec allCols(matX.n_cols);
  for (size_t i = 0; i < matX.n_cols; i++) { allCols(i) = i; }
  
  // Center and scaleData the points before applying the model.
  predictions = omega.t() * matX.submat(activeSet, allCols)
                + responsesOffset;

  // Comptute the standard deviations
  arma::mat O(matX.n_cols, matX.n_cols);
  O = matX.submat(activeSet, allCols).t()
      * matCovariance
      * matX.submat(activeSet, allCols);
  std = sqrt(diagvec(1 / beta + O).t());
}

template<typename KernelType>
double RVMRegression<KernelType>::RMSE(const arma::mat& data,
                                       const arma::rowvec& responses) const
{
  arma::rowvec predictions;
  Predict(data, predictions);
  return sqrt(mean(square(responses - predictions)));
}

// template<typename KernelType>
// arma::colvec RVMRegression<KernelType>::Omega() const
// {
//   // Get the size of the full solution with the offset.
//   arma::colvec coefs = arma::zeros<arma::colvec>(dataOffset.size());
//   // omega[i] = 0 for the inactive basis functions

//   // Now reconstruct the full solution.
//   for (size_t i = 0; i < activeSet.size(); i++)
//     coefs[activeSet[i]] = omega(i);
  
//   return coefs;
// }

template<typename KernelType>
void RVMRegression<KernelType>::applyKernel(const arma::mat& matX,
                                            const arma::mat& matY,
                                            arma::mat& kernelMatrix) const {

  // Check if the dimensions are consistent.
  if (matX.n_rows != matY.n_rows)
  {
    std::cout << "Error gramm : " << matX.n_rows << "!=" 
              << matY.n_rows << std::endl;
    throw std::invalid_argument("Number of features not consistent");
  }

  kernelMatrix = arma::mat(matX.n_cols, matY.n_cols);
  // Note that we only need to calculate the upper triangular part of the
  // kernel matrix, since it is symmetric. This helps minimize the number of
  // kernel evaluations.
  for (size_t i = 0; i < matX.n_cols; ++i)
    for (size_t j = i; j < matY.n_cols; ++j)
      kernelMatrix(i, j) = kernel.Evaluate(matX.col(i), matY.col(j));
    
  // Copy to the lower triangular part of the matrix.
  for (size_t i = 1; i < matX.n_cols; ++i)
    for (size_t j = 0; j < i; ++j)
      kernelMatrix(i, j) = kernelMatrix(j, i);
}

template<typename KernelType>
double RVMRegression<KernelType>::CenterScaleData(const arma::mat& data,
                                                  const arma::rowvec& responses,
                                                  bool centerData,
				                  bool scaleData,
                                                  arma::mat& dataProc,
				                  arma::rowvec& responsesProc,
				                  arma::colvec& dataOffset,
				                  arma::colvec& dataScale)
{
  // Initialize the offsets to their neutral forms.
  dataOffset = arma::zeros<arma::colvec>(data.n_rows);
  dataScale = arma::ones<arma::colvec>(data.n_rows);
  responsesOffset = 0.0;

  if (centerData)
  {
    dataOffset = mean(data, 1);
    responsesOffset = mean(responses);
  }

  if (scaleData)
    dataScale = stddev(data, 0, 1);

  // Copy data and response before the processing.
  dataProc = data;
  // Center the data.
  dataProc.each_col() -= dataOffset;
  // Scale the data.
  dataProc.each_col() /= dataScale;
  // Center the responses.
  responsesProc = responses - responsesOffset;

  return responsesOffset;
}
#endif
