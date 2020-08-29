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
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "rvm_regression.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;

// Program Name.
BINDING_NAME("Relevance Vector Machine for regression");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the Relevance Vector Machine (RVM) that can also be "
    "used for ARD regression on a given dataset if the kernel is not specified.");

// Long description.
BINDING_LONG_DESC(
    "This program trains a RVM model for regression on the dataset provided "
    "with the specified kernel. RVM is a bayesian kernel based technique "
    "similar to the SVM whose the solution is much more sparse, making this "
    "model fast to apply on test data. The optimization procedure maximizes "
    "the log marginal likelihood leading to automatic determination of the "
    "automaticaly determines the best hyperparameters set associated to the "
    "relevant vectors."
    "\n\n"
    "To train a RVMRegression model, the " +
    PRINT_PARAM_STRING("input") + " and " + PRINT_PARAM_STRING("responses") +
    "parameters must be given. The " + PRINT_PARAM_STRING("center") +
    "and " + PRINT_PARAM_STRING("scale") + " parameters control the "
    "centering and the normalizing options. A trained model can be saved with "
    "the " + PRINT_PARAM_STRING("output_model") + ". If no training is desired "
    "at all, a model can be passed via the " +
    PRINT_PARAM_STRING("input_model") + " parameter."
    "\n\n"
    "The program can also provide predictions for test data using either the "
    "trained model or the given input model. Test points can be specified "
    "with the " + PRINT_PARAM_STRING("test") + " parameter.  Predicted "
    "responses to the test points can be saved with the " +
    PRINT_PARAM_STRING("predictions") + " output parameter. The "
    "corresponding standard deviation can be save by precising the " +
    PRINT_PARAM_STRING("stds") + " parameter."
    "If the " + PRINT_PARAM_STRING("kernel") + "is not specified the model "
    "optimized is a bayesian linear regression associated to an ARD prior "
    "leading sparse solution over the variable domain."
    "\n"
    "The supported kernel are lister below:"
    "\n\n"
    " * 'linear': the standard linear dot product (same as normal PCA):\n"
    "    K(x, y) = x^T y\n"
    "\n"
    " * 'gaussian': a Gaussian kernel; requires bandwidth:\n"
    "    K(x, y) = exp(-(|| x - y || ^ 2) / (2 * (bandwidth ^ 2)))\n"
    "\n"
    " * 'polynomial': polynomial kernel; requires offset and degree:\n"
    "    K(x, y) = (x^T y + offset) ^ degree\n"
    "\n"
    " * 'hyptan': hyperbolic tangent kernel; requires scale and offset:\n"
    "    K(x, y) = tanh(scale * (x^T y) + offset)\n"
    "\n"
    " * 'laplacian': Laplacian kernel; requires bandwidth:\n"
    "    K(x, y) = exp(-(|| x - y ||) / bandwidth)\n"
    "\n"
    " * 'epanechnikov': Epanechnikov kernel; requires bandwidth:\n"
    "    K(x, y) = max(0, 1 - || x - y ||^2 / bandwidth^2)\n"
    "\n"
    " * 'cosine': cosine distance:\n"
    "    K(x, y) = 1 - (x^T y) / (|| x || * || y ||)\n"
    "\n"
    "The parameters for each of the kernels should be specified with the "
    "options " + PRINT_PARAM_STRING("bandwidth") + ", " +
    PRINT_PARAM_STRING("kernel_scale") + ", " +
    PRINT_PARAM_STRING("offset") + ", or " + PRINT_PARAM_STRING("degree") +
    " (or a combination of those parameters).");

//Example
BINFING_EXAMPLE(
    "For example, the following command trains a model on the data " +
    PRINT_DATASET("data") + " and responses " + PRINT_DATASET("responses") +
    "with center and scale set to true and a gaussian kernel of "
    "bandwith 1.0. RVM is solved and the model is saved to " +
    PRINT_MODEL("rvm_regression") + ":"
    "\n\n" +
    PRINT_CALL("rvm_regression", "input", "data", "responses", "responses", 
               "center", 1, "scale", 1, "output_model", 
               "rvm_regression_model", "kernel", "gaussian", "bandwidth", 1.0) +
    "The following command uses the " + PRINT_MODEL("rvm_regression_model") + 
    "to provide predicted responses for the data " + PRINT_DATASET("test") + 
    "and save those responses to " + PRINT_DATASET("test_predictions") + ":"
    "\n\n" + 
    PRINT_CALL("rvm_regression", "input_model", "vm_regression_model", "test", 
               "test", "predictions", "test_predictions") 

;
);
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
