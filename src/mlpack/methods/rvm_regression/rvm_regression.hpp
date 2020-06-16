/**
 * @file rvm_regression.hpp
 * @ Clement Mercier
 *
 * Definition of the RVMRegression class, which performs the 
 * Relevance Vector Machine for regression
**/

#ifndef MLPACK_METHODS_RVM_REGRESSION_HPP
#define MLPACK_METHODS_RVM_REGRESSION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace regression{

template<typename KernelType>
class RVMRegression
{
public:
  /**
   * Set the parameters of the RVMRegression (Relevance Vector Machine for regression) 
   *    object for a given kernel. There are numerous available kernels in 
   *    the mlpack::kernel namespace.
   *    Regulariation parameters are automaticaly set to their optimal values by 
   *    maximizing the marginal likelihood. Optimization is done by Evidence
   *    Maximization. A REFORMULER
   * @param kernel Kernel to be used for computation.
   * @param centerData Whether or not center the data according to the *
   *    examples.
   * @param scaleData Whether or to scaleData the data according to the 
   *    standard deviation of each feature.
   **/
  RVMRegression(const KernelType& kernel,
                const bool centerData,
                const bool scaleData,
                const bool ard);

  /**
   * Set the parameters of the ARD regression (Automatic Relevance Determination) 
   *    object without any kernel. The class Performs a linear regression with an ARD prior promoting 
   *    sparsity in the final solution. 
   *    Regulariation parameters are automaticaly set to their optimal values by 
   *    the maximmization of the marginal likelihood. Optimization is done by 
   *    Evidence Maximization.
   *    ARD regression is computed whatever the kernel type given for the 
   *    initalization.
   *
   * @param centerData Whether or not center the data according to the 
   *    examples.
   * @param scaleData Whether or to scaleData the data according to the 
   *    standard deviation of each feature.
   **/
  RVMRegression(const bool centerData = true,
                const bool scaleData = false);
   
  /**
   * Run Relevance Vector Machine for regression. The input matrix 
   *    (like all mlpack matrices) should be
   *    column-major -- each column is an observation and each row is 
   *    a dimension.
   *    
   * @param data Column-major input data (or row-major input data if rowMajor =
   *     true).
   * @param responses Vector of targets.
   **/
  void Train(const arma::mat& data,
	     const arma::rowvec& responses);

  /**
   * Predict \f$\hat{y}_{i}\f$ for each data point in the given data matrix using the
   *    currently-trained RVM model. Only the coefficients of the active basis 
   *    funcions are used for prediction. This allows fast predictions.
   * @param points The data points to apply the model.
   * @param predictions y, which will contained calculated values on completion.
   */
  void Predict(const arma::mat& points,
               arma::rowvec& predictions) const;

  /**
   * Predict \f$\hat{y}_{i}\f$ and the standard deviation of the predictive posterior 
   *    distribution for each data point in the given data matrix using the
   *    currently-trained RVM model. Only the coefficients of the active basis 
   *    funcions are used for prediction. This allows fast predictions.
   * @param points The data points to apply the model.
   * @param predictions y, which will contained calculated values on completion.
   * @param std Standard deviations of the predictions.
   */
  void Predict(const arma::mat& points,
               arma::rowvec& predictions,
	       arma::rowvec& std) const;

  /**
   * Apply the kernel function between the column vectors of two matrices 
   *    X and Y. If X=Y this function comptutes the Gramian matrix.
   * @param X Matrix of dimension \f$ M \times N1 \f$.
   * @param Y Matrix of dimension \f$ M \times N2 \f$.
   * @param gramMatrix of dimension \f$N1 \times N2\f$. Elements are equal
   *    to kernel.Evaluate(\f$ x_{i} \f$,\f$ y_{j} \f$).
   **/
  void applyKernel(const arma::mat& X,
		   const arma::mat& Y,
		   arma::mat& gramMatrix) const;
  
  /**
   * Compute the Root Mean Square Error
   * between the predictions returned by the model
   * and the true repsonses.
   * @param Points Data points to predict.
   * @param responses A vector of targets.
   * @return RMSE
   **/
  double RMSE(const arma::mat& data,
              const arma::rowvec& responses) const;

  /**
   * Get the solution vector
   *
   * @return omega Solution vector.
   */
  const arma::colvec& Omega() const { return omega; };

  /**
   * Get the precesion (or inverse variance) beta of the model.
   * @return \f$ \beta \f$ 
   **/
  double Beta() const { return beta; }

   /**
   * Get the precesion (or inverse variance) of the coeffcients.
   * @return \f$ \alpha_{i} \f$ 
   **/
  const arma::rowvec& Alpha() const { return alpha; }

  /**
   * Get the estimated variance.
   * @return 1.0 / \f$ \beta \f$
   **/
  double Variance() const { return 1.0 / Beta(); }

  /**
   * Get the indices of the active basis functions.
   * 
   * @return activeSet 
   **/
  arma::uvec ActiveSet() const { return activeSet; }

  /**
   * Get the mean vector computed on the features over the training points.
   * Vector of 0 if centerData is false.
   *   
   * @return responsesOffset
   */
  const arma::colvec& DataOffset() const { return dataOffset; }

  /**
   * Get the vector of standard deviations computed on the features over the 
   * training points. Vector of 1 if scaleData is false.
   *  
   * return dataOffset
   */
  const arma::colvec& DataScale() const { return dataScale; }

  /**
   * Get the mean value of the train responses.
   * @return responsesOffset
   */
  double ResponsesOffset() const { return responsesOffset; }

private:
  //! Center the data if true.
  bool centerData;

  //! Scale the data by standard deviations if true.
  bool scaleData;

  //! Mean vector computed over the points.
  arma::colvec dataOffset;

  //! Std vector computed over the points.
  arma::colvec dataScale;

  //! Mean of the response vector computed over the points.
  double responsesOffset;

  //! alpha_threshold limit to prune the basis functions.
  float alpha_threshold;

  //! kernel Kernel used.
  KernelType kernel;

  //! Indicates that ARD mode is used.
  bool ard;

  //! Kernel length scale.
  double gamma;

  //! Relevant vectors.
  arma::mat relevantVectors;

  //! Precision of the prior pdfs (independant gaussian).
  arma::rowvec alpha;

  //! Noise inverse variance.
  double beta;

  //! Solution vector.
  arma::colvec omega;

  //! Coavriance matrix of the solution vector omega.
  arma::mat matCovariance;

  //! activeSetive Indices of active basis functions.
  arma::uvec activeSet;

  /**
   * Center and scaleData the data. The last four arguments
   * allow future modifation of new points.
   *
   * @param data Design matrix in column-major format, dim(P,N).
   * @param responses A vector of targets.
   * @param centerData If true data will be centred according to the points.
   * @param centerData If true data will be scales by the standard deviations
   *     of the features computed according to the points.
   * @param dataProc data processed, dim(N,P).
   * @param responsesProc responses processed, dim(N).
   * @param dataOffset Mean vector of the design matrix according to the 
   *     points, dim(P).
   * @param dataScale Vector containg the standard deviations of the features
   *     dim(P).
   * @return reponsesOffset Mean of responses.
   */
  double CenterScaleData(const arma::mat& data,
		  const arma::rowvec& responses,
		  bool centerData,
		  bool scaleData,
		  arma::mat& dataProc,
		  arma::rowvec& responsesProc,
		  arma::colvec& dataOffset,
		  arma::colvec& dataScale);

};
} // namespace regression
} // namespace mlpack

// Include implementation.
#include "rvm_regression_impl.hpp"

#endif
