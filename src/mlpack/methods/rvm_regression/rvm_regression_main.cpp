/**
 * @file bayesian_linear_regression_main.cpp
 * @author Clement Mercier
 *
 * Executable for BayesianLinearRegression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "rvm_regression.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;

// PROGRAM_INFO("RVMRegression",
//     // Short description.
//     "An implementation of the bayesian linear regression, also known "
//     "as the Bayesian linear regression. This can train a Bayesian linear "
//     "regression model and use that model or a pre-trained model to output "
//     "regression "
//     "predictions for a test set.",
//     // Long description.
//     "An implementation of the bayesian linear regression, also known"
//     "as the Bayesian linear regression.\n "
//     "This is a probabilistic view and implementation of the linear regression. "
//     "Final solution is obtained by comptuting a posterior distribution from "
//     "gaussian likelihood and a zero mean gaussian isotropic prior distribution "
//     "on the solution. "
//     "\n"
//     "Optimization is AUTOMATIC and does not require cross validation. "
//     "The optimization is performed by maximization of the evidence function. "
//     "Parameters are tunned during the maximization of the marginal likelihood. "
//     "This procedure includes the Ockham's razor that penalizes over complex "
//     "solutions. "
//     "\n\n"
//     "This program is able to train a Bayesian linear regression model or load "
//     "a model from file, output regression predictions for a test set, and save "
//     "the trained model to a file. The Bayesian linear regression algorithm is "
//     "described in more detail below:"
//     "\n\n"
//     "Let X be a matrix where each row is a point and each column is a "
//     "dimension, t is a vector of targets, alpha is the precision of the "
//     "gaussian prior distribtion of w, and w is solution to determine. "
//     "\n\n"
//     "The Bayesian linear regression comptutes the posterior distribution of "
//     "the parameters by the Bayes's rule : "
//     "\n\n"
//     " p(w|X) = p(X,t|w) * p(w|alpha) / p(X)"
//     "\n\n"
//     "To train a BayesianLinearRegression model, the " +
//     PRINT_PARAM_STRING("input") + " and " + PRINT_PARAM_STRING("responses") +
//     "parameters must be given. The " + PRINT_PARAM_STRING("center") +
//     "and " + PRINT_PARAM_STRING("scale") + " parameters control the "
//     "centering and the normalizing options. A trained model can be saved with "
//     "the " + PRINT_PARAM_STRING("output_model") + ". If no training is desired "
//     "at all, a model can be passed via the " +
//     PRINT_PARAM_STRING("input_model") + " parameter."
//     "\n\n"
//     "The program can also provide predictions for test data using either the "
//     "trained model or the given input model.  Test points can be specified "
//     "with the " + PRINT_PARAM_STRING("test") + " parameter.  Predicted "
//     "responses to the test points can be saved with the " +
//     PRINT_PARAM_STRING("output_predictions") + " output parameter. The "
//     "corresponding standard deviation can be save by precising the " +
//     PRINT_PARAM_STRING("output_std") + " parameter."
//     "\n\n"
//     "For example, the following command trains a model on the data " +
//     PRINT_DATASET("data") + " and responses " + PRINT_DATASET("responses") +
//     "with center set to true and scale set to false (so, Bayesian "
//     "linear regression is being solved, and then the model is saved to " +
//     PRINT_MODEL("bayesian_linear_regression_model") + ":"
//     "\n\n" +
//     PRINT_CALL("bayesian_linear_regression", "input", "data", "responses",
//                "responses", "center", 1, "scale", 0, "output_model",
//                "bayesian_linear_regression_model") +
//     "\n\n"
//     "The following command uses the " +
//     PRINT_MODEL("bayesian_linear_regression_model") + " to provide predicted " +
//     " responses for the data " + PRINT_DATASET("test") + " and save those " +
//     " responses to " + PRINT_DATASET("test_predictions") + ": "
//     "\n\n" +
//     PRINT_CALL("bayesian_linear_regression", "input_model",
//                "bayesian_linear_regression_model", "test", "test",
//                "output_predictions", "test_predictions"));

// PARAM_MATRIX_IN("input", "Matrix of covariates (X).", "i");

// PARAM_MATRIX_IN("responses", "Matrix of responses/observations (y).", "r");

// PARAM_MODEL_IN(RVMRegression, "input_model", "Trained "
//                "BayesianLinearRegression model to use.", "m");

// PARAM_MODEL_OUT(RVMRegression, "output_model", "Output "
//                 "BayesianLinearRegression model.", "M");

// PARAM_MATRIX_IN("test", "Matrix containing points to regress on (test "
//                 "points).", "t");

// PARAM_MATRIX_OUT("output_predictions", "If --test_file is specified, this "
//                   "file is where the predicted responses will be saved.", "o");

// PARAM_MATRIX_OUT("output_std", "If --std_file is specified, this file is where "
//                  "the standard deviations of the predictive distribution will "
//                  "be saved.", "u");

// PARAM_INT_IN("center", "Center the data and fit the intercept. Set to 0 to "
//             "disable",
//             "c",
//             1);

// PARAM_INT_IN("scale", "Scale each feature by their standard deviations. "
//              "set to 1 to scale.",
//              "s",
//              0);

static void mlpackMain()
{

}
