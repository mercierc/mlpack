/**
 * @file tests/ann_layer_test.cpp
 * @author Marcus Edel
 * @author Praveen Ch
 *
 * Tests the ann layer modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "ann_test_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ANNLayerTest);

/**
 * Simple add module test.
 */
BOOST_AUTO_TEST_CASE(SimpleAddLayerTest)
{
  arma::mat output, input, delta;
  Add<> module(10);
  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(module.Parameters()), arma::accu(output));

  // Test the Backward function.
  module.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(delta));

  // Test the forward function.
  input = arma::ones(10, 1);
  module.Forward(input, output);
  BOOST_REQUIRE_CLOSE(10 + arma::accu(module.Parameters()),
      arma::accu(output), 1e-3);

  // Test the backward function.
  module.Backward(input, output, delta);
  BOOST_REQUIRE_CLOSE(arma::accu(output), arma::accu(delta), 1e-3);
}

/**
 * Jacobian add module test.
 */
BOOST_AUTO_TEST_CASE(JacobianAddLayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t elements = math::RandInt(2, 1000);
    arma::mat input;
    input.set_size(elements, 1);

    Add<> module(elements);
    module.Parameters().randu();

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Add layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientAddLayerTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      model->Add<Add<> >(10);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Test that the function that can access the outSize parameter of
 * the Add layer works.
 */
BOOST_AUTO_TEST_CASE(AddLayerParametersTest)
{
  // Parameter : outSize.
  Add<> layer(7);

  // Make sure we can get the parameter successfully.
  BOOST_REQUIRE_EQUAL(layer.OutputSize(), 7);
}

/**
 * Simple constant module test.
 */
BOOST_AUTO_TEST_CASE(SimpleConstantLayerTest)
{
  arma::mat output, input, delta;
  Constant<> module(10, 3.0);

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(output), 30.0);

  // Test the Backward function.
  module.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);

  // Test the forward function.
  input = arma::ones(10, 1);
  module.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(output), 30.0);

  // Test the backward function.
  module.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Jacobian constant module test.
 */
BOOST_AUTO_TEST_CASE(JacobianConstantLayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t elements = math::RandInt(2, 1000);
    arma::mat input;
    input.set_size(elements, 1);

    Constant<> module(elements, 1.0);

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Test that the function that can access the outSize parameter of the
 * Constant layer works.
 */
BOOST_AUTO_TEST_CASE(ConstantLayerParametersTest)
{
  // Parameter : outSize.
  Constant<> layer(7);

  // Make sure we can get the parameter successfully.
  BOOST_REQUIRE_EQUAL(layer.OutSize(), 7);
}

/**
 * Simple dropout module test.
 */
BOOST_AUTO_TEST_CASE(SimpleDropoutLayerTest)
{
  // Initialize the probability of setting a value to zero.
  const double p = 0.2;

  // Initialize the input parameter.
  arma::mat input(1000, 1);
  input.fill(1 - p);

  Dropout<> module(p);
  module.Deterministic() = false;

  // Test the Forward function.
  arma::mat output;
  module.Forward(input, output);
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::mean(output) - (1 - p))), 0.05);

  // Test the Backward function.
  arma::mat delta;
  module.Backward(input, input, delta);
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::mean(delta) - (1 - p))), 0.05);

  // Test the Forward function.
  module.Deterministic() = true;
  module.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(input), arma::accu(output));
}

/**
 * Perform dropout x times using ones as input, sum the number of ones and
 * validate that the layer is producing approximately the correct number of
 * ones.
 */
BOOST_AUTO_TEST_CASE(DropoutProbabilityTest)
{
  arma::mat input = arma::ones(1500, 1);
  const size_t iterations = 10;

  double probability[5] = { 0.1, 0.3, 0.4, 0.7, 0.8 };
  for (size_t trial = 0; trial < 5; ++trial)
  {
    double nonzeroCount = 0;
    for (size_t i = 0; i < iterations; ++i)
    {
      Dropout<> module(probability[trial]);
      module.Deterministic() = false;

      arma::mat output;
      module.Forward(input, output);

      // Return a column vector containing the indices of elements of X that
      // are non-zero, we just need the number of non-zero values.
      arma::uvec nonzero = arma::find(output);
      nonzeroCount += nonzero.n_elem;
    }
    const double expected = input.n_elem * (1 - probability[trial]) *
        iterations;
    const double error = fabs(nonzeroCount - expected) / expected;

    BOOST_REQUIRE_LE(error, 0.15);
  }
}

/*
 * Perform dropout with probability 1 - p where p = 0, means no dropout.
 */
BOOST_AUTO_TEST_CASE(NoDropoutTest)
{
  arma::mat input = arma::ones(1500, 1);
  Dropout<> module(0);
  module.Deterministic() = false;

  arma::mat output;
  module.Forward(input, output);

  BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(input));
}

/*
 * Perform test to check whether mean and variance remain nearly same
 * after AlphaDropout.
 */
BOOST_AUTO_TEST_CASE(SimpleAlphaDropoutLayerTest)
{
  // Initialize the probability of setting a value to alphaDash.
  const double p = 0.2;

  // Initialize the input parameter having a mean nearabout 0
  // and variance nearabout 1.
  arma::mat input = arma::randn<arma::mat>(1000, 1);

  AlphaDropout<> module(p);
  module.Deterministic() = false;

  // Test the Forward function when training phase.
  arma::mat output;
  module.Forward(input, output);
  // Check whether mean remains nearly same.
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::mean(input) - arma::mean(output))), 0.1);

  // Check whether variance remains nearly same.
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::var(input) - arma::var(output))), 0.1);

  // Test the Backward function when training phase.
  arma::mat delta;
  module.Backward(input, input, delta);
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::mean(delta) - 0)), 0.05);

  // Test the Forward function when testing phase.
  module.Deterministic() = true;
  module.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(input), arma::accu(output));
}

/**
 * Perform AlphaDropout x times using ones as input, sum the number of ones
 * and validate that the layer is producing approximately the correct number
 * of ones.
 */
BOOST_AUTO_TEST_CASE(AlphaDropoutProbabilityTest)
{
  arma::mat input = arma::ones(1500, 1);
  const size_t iterations = 10;

  double probability[5] = { 0.1, 0.3, 0.4, 0.7, 0.8 };
  for (size_t trial = 0; trial < 5; ++trial)
  {
    double nonzeroCount = 0;
    for (size_t i = 0; i < iterations; ++i)
    {
      AlphaDropout<> module(probability[trial]);
      module.Deterministic() = false;

      arma::mat output;
      module.Forward(input, output);

      // Return a column vector containing the indices of elements of X
      // that are not alphaDash, we just need the number of
      // nonAlphaDash values.
      arma::uvec nonAlphaDash = arma::find(module.Mask());
      nonzeroCount += nonAlphaDash.n_elem;
    }

    const double expected = input.n_elem * (1-probability[trial]) * iterations;

    const double error = fabs(nonzeroCount - expected) / expected;

    BOOST_REQUIRE_LE(error, 0.15);
  }
}

/**
 * Perform AlphaDropout with probability 1 - p where p = 0,
 * means no AlphaDropout.
 */
BOOST_AUTO_TEST_CASE(NoAlphaDropoutTest)
{
  arma::mat input = arma::ones(1500, 1);
  AlphaDropout<> module(0);
  module.Deterministic() = false;

  arma::mat output;
  module.Forward(input, output);

  BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(input));
}

/**
 * Simple linear module test.
 */
BOOST_AUTO_TEST_CASE(SimpleLinearLayerTest)
{
  arma::mat output, input, delta;
  Linear<> module(10, 10);
  module.Parameters().randu();
  module.Reset();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(input, output);
  BOOST_REQUIRE_CLOSE(arma::accu(
      module.Parameters().submat(100, 0, module.Parameters().n_elem - 1, 0)),
      arma::accu(output), 1e-3);

  // Test the Backward function.
  module.Backward(input, input, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Jacobian linear module test.
 */
BOOST_AUTO_TEST_CASE(JacobianLinearLayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = math::RandInt(2, 1000);
    const size_t outputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    Linear<> module(inputElements, outputElements);
    module.Parameters().randu();

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Linear layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientLinearLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      model->Add<Linear<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple noisy linear module test.
 */
BOOST_AUTO_TEST_CASE(SimpleNoisyLinearLayerTest)
{
  arma::mat output, input, delta;
  NoisyLinear<> module(10, 10);
  module.Parameters().randu();
  module.Reset();

  // Test the Backward function.
  module.Backward(input, input, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Jacobian noisy linear module test.
 */
BOOST_AUTO_TEST_CASE(JacobianNoisyLinearLayerTest)
{
  const size_t inputElements = math::RandInt(2, 1000);
  const size_t outputElements = math::RandInt(2, 1000);

  arma::mat input;
  input.set_size(inputElements, 1);

  NoisyLinear<> module(inputElements, outputElements);
  module.Parameters().randu();

  double error = JacobianTest(module, input);
  BOOST_REQUIRE_LE(error, 1e-5);
}

/**
 * Noisy Linear layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientNoisyLinearLayerTest)
{
  // Noisy linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<NoisyLinear<> >(10, 10);
      model->Add<NoisyLinear<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple linear no bias module test.
 */
BOOST_AUTO_TEST_CASE(SimpleLinearNoBiasLayerTest)
{
  arma::mat output, input, delta;
  LinearNoBias<> module(10, 10);
  module.Parameters().randu();
  module.Reset();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(input, output);
  BOOST_REQUIRE_EQUAL(0, arma::accu(output));

  // Test the Backward function.
  module.Backward(input, input, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Simple padding layer test.
 */
BOOST_AUTO_TEST_CASE(SimplePaddingLayerTest)
{
  arma::mat output, input, delta;
  Padding<> module(1, 2, 3, 4);

  // Test the Forward function.
  input = arma::randu(10, 1);
  module.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(input), arma::accu(output));
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows + 3);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols + 7);

  // Test the Backward function.
  module.Backward(input, output, delta);
  CheckMatrices(delta, input);
}

/**
 * Jacobian linear no bias module test.
 */
BOOST_AUTO_TEST_CASE(JacobianLinearNoBiasLayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = math::RandInt(2, 1000);
    const size_t outputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    LinearNoBias<> module(inputElements, outputElements);
    module.Parameters().randu();

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * LinearNoBias layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientLinearNoBiasLayerTest)
{
  // LinearNoBias function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      model->Add<LinearNoBias<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Jacobian negative log likelihood module test.
 */
BOOST_AUTO_TEST_CASE(JacobianNegativeLogLikelihoodLayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    NegativeLogLikelihood<> module;
    const size_t inputElements = math::RandInt(5, 100);
    arma::mat input;
    RandomInitialization init(0, 1);
    init.Initialize(input, inputElements, 1);

    arma::mat target(1, 1);
    target(0) = math::RandInt(1, inputElements - 1);

    double error = JacobianPerformanceTest(module, input, target);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Jacobian LeakyReLU module test.
 */
BOOST_AUTO_TEST_CASE(JacobianLeakyReLULayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    LeakyReLU<> module;

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Jacobian FlexibleReLU module test.
 */
BOOST_AUTO_TEST_CASE(JacobianFlexibleReLULayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    FlexibleReLU<> module;

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Flexible ReLU layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientFlexibleReLULayerTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(2, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, RandomInitialization>(
          NegativeLogLikelihood<>(), RandomInitialization(0.1, 0.5));

      model->Predictors() = input;
      model->Responses() = target;
      model->Add<Linear<> >(2, 2);
      model->Add<LinearNoBias<> >(2, 5);
      model->Add<FlexibleReLU<> >(0.05);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, RandomInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Jacobian MultiplyConstant module test.
 */
BOOST_AUTO_TEST_CASE(JacobianMultiplyConstantLayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    MultiplyConstant<> module(3.0);

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Jacobian HardTanH module test.
 */
BOOST_AUTO_TEST_CASE(JacobianHardTanHLayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    HardTanH<> module;

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Simple select module test.
 */
BOOST_AUTO_TEST_CASE(SimpleSelectLayerTest)
{
  arma::mat outputA, outputB, input, delta;

  input = arma::ones(10, 5);
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    input.col(i) *= i;
  }

  // Test the Forward function.
  Select<> moduleA(3);
  moduleA.Forward(input, outputA);
  BOOST_REQUIRE_EQUAL(30, arma::accu(outputA));

  // Test the Forward function.
  Select<> moduleB(3, 5);
  moduleB.Forward(input, outputB);
  BOOST_REQUIRE_EQUAL(15, arma::accu(outputB));

  // Test the Backward function.
  moduleA.Backward(input, outputA, delta);
  BOOST_REQUIRE_EQUAL(30, arma::accu(delta));

  // Test the Backward function.
  moduleB.Backward(input, outputA, delta);
  BOOST_REQUIRE_EQUAL(15, arma::accu(delta));
}

/**
 * Test that the functions that can access the parameters of the
 * Select layer work.
 */
BOOST_AUTO_TEST_CASE(SelectLayerParametersTest)
{
  // Parameter order : index, elements.
  Select<> layer(3, 5);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer.Index(), 3);
  BOOST_REQUIRE_EQUAL(layer.NumElements(), 5);
}

/**
 * Simple join module test.
 */
BOOST_AUTO_TEST_CASE(SimpleJoinLayerTest)
{
  arma::mat output, input, delta;
  input = arma::ones(10, 5);

  // Test the Forward function.
  Join<> module;
  module.Forward(input, output);
  BOOST_REQUIRE_EQUAL(50, arma::accu(output));

  bool b = output.n_rows == 1 || output.n_cols == 1;
  BOOST_REQUIRE_EQUAL(b, true);

  // Test the Backward function.
  module.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(50, arma::accu(delta));

  b = delta.n_rows == input.n_rows && input.n_cols;
  BOOST_REQUIRE_EQUAL(b, true);
}

/**
 * Simple add merge module test.
 */
BOOST_AUTO_TEST_CASE(SimpleAddMergeLayerTest)
{
  arma::mat output, input, delta;
  input = arma::ones(10, 1);

  for (size_t i = 0; i < 5; ++i)
  {
    AddMerge<> module(false, false);
    const size_t numMergeModules = math::RandInt(2, 10);
    for (size_t m = 0; m < numMergeModules; ++m)
    {
      IdentityLayer<> identityLayer;
      identityLayer.Forward(input, identityLayer.OutputParameter());

      module.Add<IdentityLayer<> >(identityLayer);
    }

    // Test the Forward function.
    module.Forward(input, output);
    BOOST_REQUIRE_EQUAL(10 * numMergeModules, arma::accu(output));

    // Test the Backward function.
    module.Backward(input, output, delta);
    BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(delta));
  }
}

/**
 * Test the LSTM layer with a user defined rho parameter and without.
 */
BOOST_AUTO_TEST_CASE(LSTMRrhoTest)
{
  const size_t rho = 5;
  arma::cube input = arma::randu(1, 1, 5);
  arma::cube target = arma::ones(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood<>, RandomInitialization> modelA(
      rho, false, NegativeLogLikelihood<>(), init);
  modelA.Add<IdentityLayer<> >();
  modelA.Add<Linear<> >(1, 10);

  // Use LSTM layer with rho.
  modelA.Add<LSTM<> >(10, 3, rho);
  modelA.Add<LogSoftMax<> >();

  // Create model without user defined rho parameter.
  RNN<NegativeLogLikelihood<> > modelB(
      rho, false, NegativeLogLikelihood<>(), init);
  modelB.Add<IdentityLayer<> >();
  modelB.Add<Linear<> >(1, 10);

  // Use LSTM layer with rho = MAXSIZE.
  modelB.Add<LSTM<> >(10, 3);
  modelB.Add<LogSoftMax<> >();

  ens::StandardSGD opt(0.1, 1, 5, -100, false);
  modelA.Train(input, target, opt);
  modelB.Train(input, target, opt);

  CheckMatrices(modelB.Parameters(), modelA.Parameters());
}

/**
 * LSTM layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientLSTMLayerTest)
{
  // LSTM function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(1, 1, 5);
      target.ones(1, 1, 5);
      const size_t rho = 5;

      model = new RNN<NegativeLogLikelihood<> >(rho);
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(1, 10);
      model->Add<LSTM<> >(10, 3, rho);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    RNN<NegativeLogLikelihood<> >* model;
    arma::cube input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Test that the functions that can modify and access the parameters of the
 * LSTM layer work.
 */
BOOST_AUTO_TEST_CASE(LSTMLayerParametersTest)
{
  // Parameter order : inSize, outSize, rho.
  LSTM<> layer1(1, 2, 3);
  LSTM<> layer2(1, 2, 4);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer1.InSize(), 1);
  BOOST_REQUIRE_EQUAL(layer1.OutSize(), 2);
  BOOST_REQUIRE_EQUAL(layer1.Rho(), 3);

  // Now modify the parameters to match the second layer.
  layer1.Rho() = 4;

  // Now ensure all the results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InSize(), layer2.InSize());
  BOOST_REQUIRE_EQUAL(layer2.OutSize(), layer2.OutSize());
  BOOST_REQUIRE_EQUAL(layer1.Rho(), layer2.Rho());
}

/**
 * Test the FastLSTM layer with a user defined rho parameter and without.
 */
BOOST_AUTO_TEST_CASE(FastLSTMRrhoTest)
{
  const size_t rho = 5;
  arma::cube input = arma::randu(1, 1, 5);
  arma::cube target = arma::ones(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood<>, RandomInitialization> modelA(
      rho, false, NegativeLogLikelihood<>(), init);
  modelA.Add<IdentityLayer<> >();
  modelA.Add<Linear<> >(1, 10);

  // Use FastLSTM layer with rho.
  modelA.Add<FastLSTM<> >(10, 3, rho);
  modelA.Add<LogSoftMax<> >();

  // Create model without user defined rho parameter.
  RNN<NegativeLogLikelihood<> > modelB(
      rho, false, NegativeLogLikelihood<>(), init);
  modelB.Add<IdentityLayer<> >();
  modelB.Add<Linear<> >(1, 10);

  // Use FastLSTM layer with rho = MAXSIZE.
  modelB.Add<FastLSTM<> >(10, 3);
  modelB.Add<LogSoftMax<> >();

  ens::StandardSGD opt(0.1, 1, 5, -100, false);
  modelA.Train(input, target, opt);
  modelB.Train(input, target, opt);

  CheckMatrices(modelB.Parameters(), modelA.Parameters());
}

/**
 * FastLSTM layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientFastLSTMLayerTest)
{
  // Fast LSTM function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(1, 1, 5);
      target = arma::ones(1, 1, 5);
      const size_t rho = 5;

      model = new RNN<NegativeLogLikelihood<> >(rho);
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(1, 10);
      model->Add<FastLSTM<> >(10, 3, rho);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    RNN<NegativeLogLikelihood<> >* model;
    arma::cube input, target;
  } function;

  // The threshold should be << 0.1 but since the Fast LSTM layer uses an
  // approximation of the sigmoid function the estimated gradient is not
  // correct.
  BOOST_REQUIRE_LE(CheckGradient(function), 0.2);
}

/**
 * Test that the functions that can modify and access the parameters of the
 * Fast LSTM layer work.
 */
BOOST_AUTO_TEST_CASE(FastLSTMLayerParametersTest)
{
  // Parameter order : inSize, outSize, rho.
  FastLSTM<> layer1(1, 2, 3);
  FastLSTM<> layer2(1, 2, 4);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer1.InSize(), 1);
  BOOST_REQUIRE_EQUAL(layer1.OutSize(), 2);
  BOOST_REQUIRE_EQUAL(layer1.Rho(), 3);

  // Now modify the parameters to match the second layer.
  layer1.Rho() = 4;

  // Now ensure all the results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InSize(), layer2.InSize());
  BOOST_REQUIRE_EQUAL(layer2.OutSize(), layer2.OutSize());
  BOOST_REQUIRE_EQUAL(layer1.Rho(), layer2.Rho());
}

/**
 * Testing the overloaded Forward() of the LSTM layer, for retrieving the cell
 * state. Besides output, the overloaded function provides read access to cell
 * state of the LSTM layer.
 */
BOOST_AUTO_TEST_CASE(ReadCellStateParamLSTMLayerTest)
{
  const size_t rho = 5, inputSize = 3, outputSize = 2;

  // Provide input of all ones.
  arma::cube input = arma::ones(inputSize, outputSize, rho);

  arma::mat inputGate, forgetGate, outputGate, hidden;
  arma::mat outLstm, cellLstm;

  // LSTM layer.
  LSTM<> lstm(inputSize, outputSize, rho);
  lstm.Reset();
  lstm.ResetCell(rho);

  // Initialize the weights to all ones.
  lstm.Parameters().ones();

  arma::mat inputWeight = arma::ones(outputSize, inputSize);
  arma::mat outputWeight = arma::ones(outputSize, outputSize);
  arma::mat bias = arma::ones(outputSize, input.n_cols);
  arma::mat cellCalc = arma::zeros(outputSize, input.n_cols);
  arma::mat outCalc = arma::zeros(outputSize, input.n_cols);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
      // Wrap a matrix around our data to avoid a copy.
      arma::mat stepData(input.slice(seqNum).memptr(),
          input.n_rows, input.n_cols, false, true);

      // Apply Forward() on LSTM layer.
      lstm.Forward(stepData, // Input.
                   outLstm,  // Output.
                   cellLstm, // Cell state.
                   false); // Don't write into the cell state.

      // Compute the value of cell state and output.
      // i = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      inputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // f = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      forgetGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // z = tanh(W.dot(x) + W.dot(h) + b).
      hidden = arma::tanh(inputWeight * stepData +
                     outputWeight * outCalc + bias);

      // c = f * c + i * z.
      cellCalc = forgetGate % cellCalc + inputGate % hidden;

      // o = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      outputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // h = o * tanh(c).
      outCalc = outputGate % arma::tanh(cellCalc);

      CheckMatrices(outLstm, outCalc, 1e-12);
      CheckMatrices(cellLstm, cellCalc, 1e-12);
  }
}

/**
 * Testing the overloaded Forward() of the LSTM layer, for retrieving the cell
 * state. Besides output, the overloaded function provides write access to cell
 * state of the LSTM layer.
 */
BOOST_AUTO_TEST_CASE(WriteCellStateParamLSTMLayerTest)
{
  const size_t rho = 5, inputSize = 3, outputSize = 2;

  // Provide input of all ones.
  arma::cube input = arma::ones(inputSize, outputSize, rho);

  arma::mat inputGate, forgetGate, outputGate, hidden;
  arma::mat outLstm, cellLstm;
  arma::mat cellCalc;

  // LSTM layer.
  LSTM<> lstm(inputSize, outputSize, rho);
  lstm.Reset();
  lstm.ResetCell(rho);

  // Initialize the weights to all ones.
  lstm.Parameters().ones();

  arma::mat inputWeight = arma::ones(outputSize, inputSize);
  arma::mat outputWeight = arma::ones(outputSize, outputSize);
  arma::mat bias = arma::ones(outputSize, input.n_cols);
  arma::mat outCalc = arma::zeros(outputSize, input.n_cols);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
      // Wrap a matrix around our data to avoid a copy.
      arma::mat stepData(input.slice(seqNum).memptr(),
          input.n_rows, input.n_cols, false, true);

      if (cellLstm.is_empty())
      {
        // Set the cell state to zeros.
        cellLstm = arma::zeros(outputSize, input.n_cols);
        cellCalc = arma::zeros(outputSize, input.n_cols);
      }
      else
      {
        // Set the cell state to zeros.
        cellLstm = arma::zeros(cellLstm.n_rows, cellLstm.n_cols);
        cellCalc = arma::zeros(cellCalc.n_rows, cellCalc.n_cols);
      }

      // Apply Forward() on the LSTM layer.
      lstm.Forward(stepData, // Input.
                   outLstm,  // Output.
                   cellLstm, // Cell state.
                   true);  // Write into cell state.

      // Compute the value of cell state and output.
      // i = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      inputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // f = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      forgetGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // z = tanh(W.dot(x) + W.dot(h) + b).
      hidden = arma::tanh(inputWeight * stepData +
                     outputWeight * outCalc + bias);

      // c = f * c + i * z.
      cellCalc = forgetGate % cellCalc + inputGate % hidden;

      // o = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      outputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // h = o * tanh(c).
      outCalc = outputGate % arma::tanh(cellCalc);

      CheckMatrices(outLstm, outCalc, 1e-12);
      CheckMatrices(cellLstm, cellCalc, 1e-12);
  }

  // Attempting to write empty matrix into cell state.
  lstm.Reset();
  lstm.ResetCell(rho);
  arma::mat stepData(input.slice(0).memptr(),
      input.n_rows, input.n_cols, false, true);

  lstm.Forward(stepData, // Input.
               outLstm,  // Output.
               cellLstm, // Cell state.
               true); // Write into cell state.

  for (size_t seqNum = 1; seqNum < rho; ++seqNum)
  {
    arma::mat empty;
    // Should throw error.
    BOOST_REQUIRE_THROW(lstm.Forward(stepData, // Input.
                                     outLstm,  // Output.
                                     empty, // Cell state.
                                     true),  // Write into cell state.
                                     std::runtime_error);
  }
}

/**
 * Test that the functions that can modify and access the parameters of the
 * GRU layer work.
 */
BOOST_AUTO_TEST_CASE(GRULayerParametersTest)
{
  // Parameter order : inSize, outSize, rho.
  GRU<> layer1(1, 2, 3);
  GRU<> layer2(1, 2, 4);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer1.InSize(), 1);
  BOOST_REQUIRE_EQUAL(layer1.OutSize(), 2);
  BOOST_REQUIRE_EQUAL(layer1.Rho(), 3);

  // Now modify the parameters to match the second layer.
  layer1.Rho() = 4;

  // Now ensure all the results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InSize(), layer2.InSize());
  BOOST_REQUIRE_EQUAL(layer2.OutSize(), layer2.OutSize());
  BOOST_REQUIRE_EQUAL(layer1.Rho(), layer2.Rho());
}

/**
 * Check if the gradients computed by GRU cell are close enough to the
 * approximation of the gradients.
 */
BOOST_AUTO_TEST_CASE(GradientGRULayerTest)
{
  // GRU function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(1, 1, 5);
      target = arma::ones(1, 1, 5);
      const size_t rho = 5;

      model = new RNN<NegativeLogLikelihood<> >(rho);
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(1, 10);
      model->Add<GRU<> >(10, 3, rho);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      arma::mat output;
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    RNN<NegativeLogLikelihood<> >* model;
    arma::cube input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * GRU layer manual forward test.
 */
BOOST_AUTO_TEST_CASE(ForwardGRULayerTest)
{
  // This will make it easier to clean memory later.
  GRU<>* gruAlloc = new GRU<>(3, 3, 5);
  GRU<>& gru = *gruAlloc;

  // Initialize the weights to all ones.
  NetworkInitialization<ConstInitialization>
    networkInit(ConstInitialization(1));
  networkInit.Initialize(gru.Model(), gru.Parameters());

  // Provide input of all ones.
  arma::mat input = arma::ones(3, 1);
  arma::mat output;

  gru.Forward(input, output);

  // Compute the z_t gate output.
  arma::mat expectedOutput = arma::ones(3, 1);
  expectedOutput *= -4;
  expectedOutput = arma::exp(expectedOutput);
  expectedOutput = arma::ones(3, 1) / (arma::ones(3, 1) + expectedOutput);
  expectedOutput = (arma::ones(3, 1)  - expectedOutput) % expectedOutput;

  // For the first input the output should be equal to the output of
  // gate z_t as the previous output fed to the cell is all zeros.
  BOOST_REQUIRE_LE(arma::as_scalar(arma::trans(output) * expectedOutput), 1e-2);

  expectedOutput = output;

  gru.Forward(input, output);

  double s = arma::as_scalar(arma::sum(expectedOutput));

  // Compute the value of z_t gate for the second input.
  arma::mat z_t = arma::ones(3, 1);
  z_t *= -(s + 4);
  z_t = arma::exp(z_t);
  z_t = arma::ones(3, 1) / (arma::ones(3, 1) + z_t);

  // Compute the value of o_t gate for the second input.
  arma::mat o_t = arma::ones(3, 1);
  o_t *= -(arma::as_scalar(arma::sum(expectedOutput % z_t)) + 4);
  o_t = arma::exp(o_t);
  o_t = arma::ones(3, 1) / (arma::ones(3, 1) + o_t);

  // Expected output for the second input.
  expectedOutput = z_t % expectedOutput + (arma::ones(3, 1) - z_t) % o_t;

  BOOST_REQUIRE_LE(arma::as_scalar(arma::trans(output) * expectedOutput), 1e-2);

  LayerTypes<> layer(gruAlloc);
  boost::apply_visitor(DeleteVisitor(), layer);
}

/**
 * Simple concat module test.
 */
BOOST_AUTO_TEST_CASE(SimpleConcatLayerTest)
{
  arma::mat output, input, delta, error;

  Linear<>* moduleA = new Linear<>(10, 10);
  moduleA->Parameters().randu();
  moduleA->Reset();

  Linear<>* moduleB = new Linear<>(10, 10);
  moduleB->Parameters().randu();
  moduleB->Reset();

  Concat<> module;
  module.Add(moduleA);
  module.Add(moduleB);

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(input, output);

  const double sumModuleA = arma::accu(
      moduleA->Parameters().submat(
      100, 0, moduleA->Parameters().n_elem - 1, 0));
  const double sumModuleB = arma::accu(
      moduleB->Parameters().submat(
      100, 0, moduleB->Parameters().n_elem - 1, 0));
  BOOST_REQUIRE_CLOSE(sumModuleA + sumModuleB, arma::accu(output.col(0)), 1e-3);

  // Test the Backward function.
  error = arma::zeros(20, 1);
  module.Backward(input, error, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Test to check Concat layer along different axes.
 */
BOOST_AUTO_TEST_CASE(ConcatAlongAxisTest)
{
  arma::mat output, input, error, outputA, outputB;
  size_t inputWidth = 4, inputHeight = 4, inputChannel = 2;
  size_t outputWidth, outputHeight, outputChannel = 2;
  size_t kW = 3, kH = 3;
  size_t batch = 1;

  // Using Convolution<> layer as inout to Concat<> layer.
  // Compute the output shape of convolution layer.
  outputWidth  = (inputWidth - kW) + 1;
  outputHeight = (inputHeight - kH) + 1;

  input = arma::ones(inputWidth * inputHeight * inputChannel, batch);

  Convolution<>* moduleA = new Convolution<>(inputChannel, outputChannel,
      kW, kH, 1, 1, 0, 0, inputWidth, inputHeight);
  Convolution<>* moduleB = new Convolution<>(inputChannel, outputChannel,
      kW, kH, 1, 1, 0, 0, inputWidth, inputHeight);

  moduleA->Reset();
  moduleA->Parameters().randu();
  moduleB->Reset();
  moduleB->Parameters().randu();

  // Compute output of each layer.
  moduleA->Forward(input, outputA);
  moduleB->Forward(input, outputB);

  arma::cube A(outputA.memptr(), outputWidth, outputHeight, outputChannel);
  arma::cube B(outputB.memptr(), outputWidth, outputHeight, outputChannel);

  error = arma::ones(outputWidth * outputHeight * outputChannel * 2, 1);

  for (size_t axis = 0; axis < 3; ++axis)
  {
    size_t x = 1, y = 1, z = 1;
    arma::cube calculatedOut;
    if (axis == 0)
    {
      calculatedOut.set_size(2 * outputWidth, outputHeight, outputChannel);
      for (size_t i = 0; i < A.n_slices; ++i)
      {
          arma::mat aMat = A.slice(i);
          arma::mat bMat = B.slice(i);
          calculatedOut.slice(i) = arma::join_cols(aMat, bMat);
      }
      x = 2;
    }
    if (axis == 1)
    {
      calculatedOut.set_size(outputWidth, 2 * outputHeight, outputChannel);
      for (size_t i = 0; i < A.n_slices; ++i)
      {
          arma::mat aMat = A.slice(i);
          arma::mat bMat = B.slice(i);
          calculatedOut.slice(i) = arma::join_rows(aMat, bMat);
      }
      y = 2;
    }
    if (axis == 2)
    {
      calculatedOut = arma::join_slices(A, B);
      z = 2;
    }

    // Compute output of Concat<> layer.
    arma::Row<size_t> inputSize{outputWidth, outputHeight, outputChannel};
    Concat<> module(inputSize, axis, true);
    module.Add(moduleA);
    module.Add(moduleB);
    module.Forward(input, output);
    arma::cube concatOut(output.memptr(), x * outputWidth,
        y * outputHeight, z * outputChannel);

    // Verify if the output reshaped to cubes are similar.
    CheckMatrices(concatOut, calculatedOut, 1e-12);
  }
  delete moduleA;
  delete moduleB;
}

/**
 * Test that the function that can access the axis parameter of the
 * Concat layer works.
 */
BOOST_AUTO_TEST_CASE(ConcatLayerParametersTest)
{
  // Parameter order : inputSize{width, height, channels}, axis, model, run.
  arma::Row<size_t> inputSize{128, 128, 3};
  Concat<> layer(inputSize, 2, false, true);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer.ConcatAxis(), 2);
}

/**
 * Concat layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientConcatLayerTest)
{
  // Concat function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);

      concat = new Concat<>(true);
      concat->Add<Linear<> >(10, 2);
      model->Add(concat);

      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    Concat<>* concat;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple concatenate module test.
 */
BOOST_AUTO_TEST_CASE(SimpleConcatenateLayerTest)
{
  arma::mat input = arma::ones(5, 1);
  arma::mat output, delta;

  Concatenate<> module;
  module.Concat() = arma::ones(5, 1) * 0.5;

  // Test the Forward function.
  module.Forward(input, output);

  BOOST_REQUIRE_EQUAL(arma::accu(output), 7.5);

  // Test the Backward function.
  module.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 5);
}

/**
 * Concatenate layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientConcatenateLayerTest)
{
  // Concatenate function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 5);

      arma::mat concat = arma::ones(5, 1);
      concatenate = new Concatenate<>();
      concatenate->Concat() = concat;
      model->Add(concatenate);

      model->Add<Linear<> >(10, 5);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    Concatenate<>* concatenate;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple lookup module test.
 */
BOOST_AUTO_TEST_CASE(SimpleLookupLayerTest)
{
  arma::mat output, input, delta, gradient;
  Lookup<> module(10, 5);
  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(2, 1);
  input(0) = 1;
  input(1) = 3;

  module.Forward(input, output);

  // The Lookup module uses index - 1 for the cols.
  const double outputSum = arma::accu(module.Parameters().col(0)) +
      arma::accu(module.Parameters().col(2));

  BOOST_REQUIRE_CLOSE(outputSum, arma::accu(output), 1e-3);

  // Test the Backward function.
  module.Backward(input, input, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(input), arma::accu(input));

  // Test the Gradient function.
  arma::mat error = arma::ones(2, 5);
  error = error.t();
  error.col(1) *= 0.5;

  module.Gradient(input, error, gradient);

  // The Lookup module uses index - 1 for the cols.
  const double gradientSum = arma::accu(gradient.col(0)) +
      arma::accu(gradient.col(2));

  BOOST_REQUIRE_CLOSE(gradientSum, arma::accu(error), 1e-3);
  BOOST_REQUIRE_CLOSE(arma::accu(gradient), arma::accu(error), 1e-3);
}

/**
 * Test that the functions that can access the parameters of the
 * Lookup layer work.
 */
BOOST_AUTO_TEST_CASE(LookupLayerParametersTest)
{
  // Parameter order : inSize, outSize.
  Lookup<> layer(5, 7);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer.InSize(), 5);
  BOOST_REQUIRE_EQUAL(layer.OutSize(), 7);
}

/**
 * Simple LogSoftMax module test.
 */
BOOST_AUTO_TEST_CASE(SimpleLogSoftmaxLayerTest)
{
  arma::mat output, input, error, delta;
  LogSoftMax<> module;

  // Test the Forward function.
  input = arma::mat("0.5; 0.5");
  module.Forward(input, output);
  BOOST_REQUIRE_SMALL(arma::accu(arma::abs(
    arma::mat("-0.6931; -0.6931") - output)), 1e-3);

  // Test the Backward function.
  error = arma::zeros(input.n_rows, input.n_cols);
  // Assume LogSoftmax layer is always associated with NLL output layer.
  error(1, 0) = -1;
  module.Backward(input, error, delta);
  BOOST_REQUIRE_SMALL(arma::accu(arma::abs(
      arma::mat("1.6487; 0.6487") - delta)), 1e-3);
}

/**
 * Simple Softmax module test.
 */
BOOST_AUTO_TEST_CASE(SimpleSoftmaxLayerTest)
{
  arma::mat input, output, gy, g;
  Softmax<> module;

  // Test the forward function.
  input = arma::mat("1.7; 3.6");
  module.Forward(input, output);
  BOOST_REQUIRE_SMALL(arma::accu(arma::abs(
    arma::mat("0.130108; 0.869892") - output)), 1e-4);

  // Test the backward function.
  gy = arma::zeros(input.n_rows, input.n_cols);
  gy(0) = 1;
  module.Backward(output, gy, g);
  BOOST_REQUIRE_SMALL(arma::accu(arma::abs(
    arma::mat("0.11318; -0.11318") - g)), 1e-04);
}

/**
 * Softmax layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientSoftmaxTest)
{
  // Softmax function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1; 0");

      model = new FFN<MeanSquaredError<>, RandomInitialization>;
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<Linear<> >(10, 10);
      model->Add<ReLULayer<> >();
      model->Add<Linear<> >(10, 2);
      model->Add<Softmax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<MeanSquaredError<> >* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/*
 * Simple test for the BilinearInterpolation layer
 */
BOOST_AUTO_TEST_CASE(SimpleBilinearInterpolationLayerTest)
{
  // Tested output against tensorflow.image.resize_bilinear()
  arma::mat input, output, unzoomedOutput, expectedOutput;
  size_t inRowSize = 2;
  size_t inColSize = 2;
  size_t outRowSize = 5;
  size_t outColSize = 5;
  size_t depth = 1;
  input.zeros(inRowSize * inColSize * depth, 1);
  input[0] = 1.0;
  input[1] = input[2] = 2.0;
  input[3] = 3.0;
  BilinearInterpolation<> layer(inRowSize, inColSize, outRowSize, outColSize,
      depth);
  expectedOutput = arma::mat("1.0000 1.4000 1.8000 2.0000 2.0000 \
      1.4000 1.8000 2.2000 2.4000 2.4000 \
      1.8000 2.2000 2.6000 2.8000 2.8000 \
      2.0000 2.4000 2.8000 3.0000 3.0000 \
      2.0000 2.4000 2.8000 3.0000 3.0000");
  expectedOutput.reshape(25, 1);
  layer.Forward(input, output);
  CheckMatrices(output - expectedOutput, arma::zeros(output.n_rows), 1e-12);

  expectedOutput = arma::mat("1.0000 1.9000 1.9000 2.8000");
  expectedOutput.reshape(4, 1);
  layer.Backward(output, output, unzoomedOutput);
  CheckMatrices(unzoomedOutput - expectedOutput,
      arma::zeros(input.n_rows), 1e-12);
}

/**
 * Test that the functions that can modify and access the parameters of the
 * Bilinear Interpolation layer work.
 */
BOOST_AUTO_TEST_CASE(BilinearInterpolationLayerParametersTest)
{
  // Parameter order : inRowSize, inColSize, outRowSize, outColSize, depth.
  BilinearInterpolation<> layer1(1, 2, 3, 4, 5);
  BilinearInterpolation<> layer2(2, 3, 4, 5, 6);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer1.InRowSize(), 1);
  BOOST_REQUIRE_EQUAL(layer1.InColSize(), 2);
  BOOST_REQUIRE_EQUAL(layer1.OutRowSize(), 3);
  BOOST_REQUIRE_EQUAL(layer1.OutColSize(), 4);
  BOOST_REQUIRE_EQUAL(layer1.InDepth(), 5);

  // Now modify the parameters to match the second layer.
  layer1.InRowSize() = 2;
  layer1.InColSize() = 3;
  layer1.OutRowSize() = 4;
  layer1.OutColSize() = 5;
  layer1.InDepth() = 6;

  // Now ensure all results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InRowSize(), layer2.InRowSize());
  BOOST_REQUIRE_EQUAL(layer1.InColSize(), layer2.InColSize());
  BOOST_REQUIRE_EQUAL(layer1.OutRowSize(), layer2.OutRowSize());
  BOOST_REQUIRE_EQUAL(layer1.OutColSize(), layer2.OutColSize());
  BOOST_REQUIRE_EQUAL(layer1.InDepth(), layer2.InDepth());
}

/**
 * Tests the BatchNorm Layer, compares the layers parameters with
 * the values from another implementation.
 * Link to the implementation - http://cthorey.github.io./backpropagation/
 */
BOOST_AUTO_TEST_CASE(BatchNormTest)
{
  arma::mat input, output;
  input << 5.1 << 3.5 << 1.4 << arma::endr
        << 4.9 << 3.0 << 1.4 << arma::endr
        << 4.7 << 3.2 << 1.3 << arma::endr;

  // BatchNorm layer with average parameter set to true.
  BatchNorm<> model(input.n_rows);
  model.Reset();

  // BatchNorm layer with average parameter set to false.
  BatchNorm<> model2(input.n_rows, 1e-5, false);
  model2.Reset();

  // Non-Deteministic Forward Pass Test.
  model.Deterministic() = false;
  model.Forward(input, output);

  // Value calculates using torch.nn.BatchNorm2d(momentum = None).
  arma::mat result;
  result << 1.1658 << 0.1100 << -1.2758 << arma::endr
         << 1.2579 << -0.0699 << -1.1880 << arma::endr
         << 1.1737 << 0.0958 << -1.2695 << arma::endr;

  CheckMatrices(output, result, 1e-1);

  model2.Forward(input, output);
  CheckMatrices(output, result, 1e-1);
  result.clear();

  // Values calculated using torch.nn.BatchNorm2d(momentum = None).
  output = model.TrainingMean();
  result << 3.33333333 << arma::endr
         << 3.1 << arma::endr
         << 3.06666666 << arma::endr;

  CheckMatrices(output, result, 1e-1);

  // Values calculated using torch.nn.BatchNorm2d().
  output = model2.TrainingMean();
  result << 0.3333 << arma::endr
         << 0.3100 << arma::endr
         << 0.3067 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  // Values calculated using torch.nn.BatchNorm2d(momentum = None).
  output = model.TrainingVariance();
  result << 3.4433 << arma::endr
         << 3.0700 << arma::endr
         << 2.9033 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  // Values calculated using torch.nn.BatchNorm2d().
  output = model2.TrainingVariance();
  result << 1.2443 << arma::endr
         << 1.2070 << arma::endr
         << 1.1903 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  // Deterministic Forward Pass test.
  model.Deterministic() = true;
  model.Forward(input, output);

  // Values calculated using torch.nn.BatchNorm2d(momentum = None).
  result << 0.9521 << 0.0898 << -1.0419 << arma::endr
         << 1.0273 << -0.0571 << -0.9702 << arma::endr
         << 0.9586 << 0.0783 << -1.0368 << arma::endr;

  CheckMatrices(output, result, 1e-1);

  // Values calculated using torch.nn.BatchNorm2d().
  model2.Deterministic() = true;
  model2.Forward(input, output);

  result << 4.2731 << 2.8388 << 0.9562 << arma::endr
         << 4.1779 << 2.4485 << 0.9921 << arma::endr
         << 4.0268 << 2.6519 << 0.9105 << arma::endr;
  CheckMatrices(output, result, 1e-1);
}

/**
 * BatchNorm layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientBatchNormTest)
{
  bool pass = false;
  for (size_t trial = 0; trial < 10; trial++)
  {
    // Add function gradient instantiation.
    struct GradientFunction
    {
      GradientFunction()
      {
        input = arma::randn(32, 2048);
        arma::mat target;
        target.ones(1, 2048);

        model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
        model->Predictors() = input;
        model->Responses() = target;
        model->Add<IdentityLayer<> >();
        model->Add<Linear<> >(32, 4);
        model->Add<BatchNorm<> >(4);
        model->Add<Linear<>>(4, 2);
        model->Add<LogSoftMax<> >();
      }

      ~GradientFunction()
      {
        delete model;
      }

      double Gradient(arma::mat& gradient) const
      {
        double error = model->Evaluate(model->Parameters(), 0, 2048, false);
        model->Gradient(model->Parameters(), 0, gradient, 2048);
        return error;
      }

      arma::mat& Parameters() { return model->Parameters(); }

      FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
      arma::mat input, target;
    } function;

    double gradient = CheckGradient(function);
    if (gradient < 2e-1)
    {
      pass = true;
      break;
    }
  }

  BOOST_REQUIRE(pass);
}

/**
 * Test that the functions that can access the parameters of the
 * Batch Norm layer work.
 */
BOOST_AUTO_TEST_CASE(BatchNormLayerParametersTest)
{
  // Parameter order : size, eps.
  BatchNorm<> layer(7, 1e-3);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer.InputSize(), 7);
  BOOST_REQUIRE_EQUAL(layer.Epsilon(), 1e-3);

  arma::mat runningMean(7, 1, arma::fill::randn);
  arma::mat runningVariance(7, 1, arma::fill::randn);

  layer.TrainingVariance() = runningVariance;
  layer.TrainingMean() = runningMean;
  CheckMatrices(layer.TrainingVariance(), runningVariance);
  CheckMatrices(layer.TrainingMean(), runningMean);
}

/**
 * VirtualBatchNorm layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientVirtualBatchNormTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randn(5, 256);
      arma::mat referenceBatch = arma::mat(input.memptr(), input.n_rows, 16);
      arma::mat target;
      target.ones(1, 256);

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(5, 5);
      model->Add<VirtualBatchNorm<> >(referenceBatch, 5);
      model->Add<Linear<> >(5, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 256, false);
      model->Gradient(model->Parameters(), 0, gradient, 256);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Test that the functions that can modify and access the parameters of the
 * Virtual Batch Norm layer work.
 */
BOOST_AUTO_TEST_CASE(VirtualBatchNormLayerParametersTest)
{
  arma::mat input = arma::randn(5, 256);
  arma::mat referenceBatch = arma::mat(input.memptr(), input.n_rows, 16);

  // Parameter order : referenceBatch, size, eps.
  VirtualBatchNorm<> layer(referenceBatch, 5, 1e-3);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer.InSize(), 5);
  BOOST_REQUIRE_EQUAL(layer.Epsilon(), 1e-3);
}

/**
 * MiniBatchDiscrimination layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(MiniBatchDiscriminationTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randn(5, 4);
      arma::mat target;
      target.ones(1, 4);

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(5, 5);
      model->Add<MiniBatchDiscrimination<> >(5, 10, 16);
      model->Add<Linear<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      return model->EvaluateWithGradient(model->Parameters(), 0, gradient, 4);
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple Transposed Convolution layer test.
 */
BOOST_AUTO_TEST_CASE(SimpleTransposedConvolutionLayerTest)
{
  arma::mat output, input, delta;

  TransposedConvolution<> module1(1, 1, 3, 3, 1, 1, 0, 0, 4, 4, 6, 6);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 15, 16);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Parameters()(0) = 1.0;
  module1.Parameters()(8) = 2.0;
  module1.Reset();
  module1.Forward(input, output);
  // Value calculated using tensorflow.nn.conv2d_transpose()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 360.0);

  // Test the backward function.
  module1.Backward(input, output, delta);
  // Value calculated using tensorflow.nn.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 720.0);

  TransposedConvolution<> module2(1, 1, 4, 4, 1, 1, 1, 1, 5, 5, 6, 6);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module2.Parameters() = arma::mat(16 + 1, 1, arma::fill::zeros);
  module2.Parameters()(0) = 1.0;
  module2.Parameters()(3) = 1.0;
  module2.Parameters()(6) = 1.0;
  module2.Parameters()(9) = 1.0;
  module2.Parameters()(12) = 1.0;
  module2.Parameters()(15) = 2.0;
  module2.Reset();
  module2.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 1512.0);

  // Test the backward function.
  module2.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 6504.0);

  TransposedConvolution<> module3(1, 1, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module3.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module3.Parameters()(1) = 2.0;
  module3.Parameters()(2) = 4.0;
  module3.Parameters()(3) = 3.0;
  module3.Parameters()(8) = 1.0;
  module3.Reset();
  module3.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 2370.0);

  // Test the backward function.
  module3.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 19154.0);

  TransposedConvolution<> module4(1, 1, 3, 3, 1, 1, 0, 0, 5, 5, 7, 7);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module4.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module4.Parameters()(2) = 2.0;
  module4.Parameters()(4) = 4.0;
  module4.Parameters()(6) = 6.0;
  module4.Parameters()(8) = 8.0;
  module4.Reset();
  module4.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 6000.0);

  // Test the backward function.
  module4.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 86208.0);

  TransposedConvolution<> module5(1, 1, 3, 3, 2, 2, 0, 0, 2, 2, 5, 5);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module5.Parameters() = arma::mat(25 + 1, 1, arma::fill::zeros);
  module5.Parameters()(2) = 8.0;
  module5.Parameters()(4) = 6.0;
  module5.Parameters()(6) = 4.0;
  module5.Parameters()(8) = 2.0;
  module5.Reset();
  module5.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 120.0);

  // Test the backward function.
  module5.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 960.0);

  TransposedConvolution<> module6(1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 5, 5);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module6.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module6.Parameters()(0) = 8.0;
  module6.Parameters()(3) = 6.0;
  module6.Parameters()(6) = 2.0;
  module6.Parameters()(8) = 4.0;
  module6.Reset();
  module6.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 410.0);

  // Test the backward function.
  module6.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 4444.0);

  TransposedConvolution<> module7(1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 6, 6);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module7.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module7.Parameters()(0) = 8.0;
  module7.Parameters()(2) = 6.0;
  module7.Parameters()(4) = 2.0;
  module7.Parameters()(8) = 4.0;
  module7.Reset();
  module7.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 606.0);

  module7.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 7732.0);
}

/**
 * Transposed Convolution layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientTransposedConvolutionLayerTest)
{
  // Add function gradient instantiation.
  // To make this test robust, check it five times.
  bool pass = false;
  for (size_t trial = 0; trial < 5; trial++)
  {
    struct GradientFunction
    {
      GradientFunction()
      {
        input = arma::linspace<arma::colvec>(0, 35, 36);
        target = arma::mat("1");

        model = new FFN<NegativeLogLikelihood<>, RandomInitialization>();
        model->Predictors() = input;
        model->Responses() = target;
        model->Add<TransposedConvolution<> >
            (1, 1, 3, 3, 2, 2, 1, 1, 6, 6, 12, 12);
        model->Add<LogSoftMax<> >();
      }

      ~GradientFunction()
      {
        delete model;
      }

      double Gradient(arma::mat& gradient) const
      {
        double error = model->Evaluate(model->Parameters(), 0, 1);
        model->Gradient(model->Parameters(), 0, gradient, 1);
        return error;
      }

      arma::mat& Parameters() { return model->Parameters(); }

      FFN<NegativeLogLikelihood<>, RandomInitialization>* model;
      arma::mat input, target;
    } function;

    if (CheckGradient(function) < 1e-3)
    {
      pass = true;
      break;
    }
  }
  BOOST_REQUIRE_EQUAL(pass, true);
}

/**
 * Simple MultiplyMerge module test.
 */
BOOST_AUTO_TEST_CASE(SimpleMultiplyMergeLayerTest)
{
  arma::mat output, input, delta;
  input = arma::ones(10, 1);

  for (size_t i = 0; i < 5; ++i)
  {
    MultiplyMerge<> module(false, false);
    const size_t numMergeModules = math::RandInt(2, 10);
    for (size_t m = 0; m < numMergeModules; ++m)
    {
      IdentityLayer<> identityLayer;
      identityLayer.Forward(input, identityLayer.OutputParameter());

      module.Add<IdentityLayer<> >(identityLayer);
    }

    // Test the Forward function.
    module.Forward(input, output);
    BOOST_REQUIRE_EQUAL(10, arma::accu(output));

    // Test the Backward function.
    module.Backward(input, output, delta);
    BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(delta));
  }
}

/**
 * Simple Atrous Convolution layer test.
 */
BOOST_AUTO_TEST_CASE(SimpleAtrousConvolutionLayerTest)
{
  arma::mat output, input, delta;

  AtrousConvolution<> module1(1, 1, 3, 3, 1, 1, 0, 0, 7, 7, 2, 2);
  // Test the Forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Parameters()(0) = 1.0;
  module1.Parameters()(8) = 2.0;
  module1.Reset();
  module1.Forward(input, output);
  // Value calculated using tensorflow.nn.atrous_conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 792.0);

  // Test the Backward function.
  module1.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 2376);

  AtrousConvolution<> module2(1, 1, 3, 3, 2, 2, 0, 0, 7, 7, 2, 2);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module2.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module2.Parameters()(0) = 1.0;
  module2.Parameters()(3) = 1.0;
  module2.Parameters()(6) = 1.0;
  module2.Reset();
  module2.Forward(input, output);
  // Value calculated using tensorflow.nn.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 264.0);

  // Test the backward function.
  module2.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 792.0);
}

/**
 * Atrous Convolution layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientAtrousConvolutionLayerTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::linspace<arma::colvec>(0, 35, 36);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, RandomInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<AtrousConvolution<> >(1, 1, 3, 3, 1, 1, 0, 0, 6, 6, 2, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, RandomInitialization>* model;
    arma::mat input, target;
  } function;

  // TODO: this tolerance seems far higher than necessary.  The implementation
  // should be checked.
  BOOST_REQUIRE_LE(CheckGradient(function), 0.2);
}

/**
 * Test the functions to access and modify the parameters of the
 * AtrousConvolution layer.
 */
BOOST_AUTO_TEST_CASE(AtrousConvolutionLayerParametersTest)
{
  // Parameter order for the constructor: inSize, outSize, kW, kH, dW, dH, padW,
  // padH, inputWidth, inputHeight, dilationW, dilationH, paddingType ("none").
  AtrousConvolution<> layer1(1, 2, 3, 4, 5, 6, std::make_tuple(7, 8),
      std::make_tuple(9, 10), 11, 12, 13, 14);
  AtrousConvolution<> layer2(2, 3, 4, 5, 6, 7, std::make_tuple(8, 9),
      std::make_tuple(10, 11), 12, 13, 14, 15);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), 11);
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), 12);
  BOOST_REQUIRE_EQUAL(layer1.KernelWidth(), 3);
  BOOST_REQUIRE_EQUAL(layer1.KernelHeight(), 4);
  BOOST_REQUIRE_EQUAL(layer1.StrideWidth(), 5);
  BOOST_REQUIRE_EQUAL(layer1.StrideHeight(), 6);
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadHTop(), 9);
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadHBottom(), 10);
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadWLeft(), 7);
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadWRight(), 8);
  BOOST_REQUIRE_EQUAL(layer1.DilationWidth(), 13);
  BOOST_REQUIRE_EQUAL(layer1.DilationHeight(), 14);

  // Now modify the parameters to match the second layer.
  layer1.InputWidth() = 12;
  layer1.InputHeight() = 13;
  layer1.KernelWidth() = 4;
  layer1.KernelHeight() = 5;
  layer1.StrideWidth() = 6;
  layer1.StrideHeight() = 7;
  layer1.Padding().PadHTop() = 10;
  layer1.Padding().PadHBottom() = 11;
  layer1.Padding().PadWLeft() = 8;
  layer1.Padding().PadWRight() = 9;
  layer1.DilationWidth() = 14;
  layer1.DilationHeight() = 15;

  // Now ensure all results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), layer2.InputWidth());
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), layer2.InputHeight());
  BOOST_REQUIRE_EQUAL(layer1.KernelWidth(), layer2.KernelWidth());
  BOOST_REQUIRE_EQUAL(layer1.KernelHeight(), layer2.KernelHeight());
  BOOST_REQUIRE_EQUAL(layer1.StrideWidth(), layer2.StrideWidth());
  BOOST_REQUIRE_EQUAL(layer1.StrideHeight(), layer2.StrideHeight());
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadHTop(), layer2.Padding().PadHTop());
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadHBottom(),
                      layer2.Padding().PadHBottom());
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadWLeft(),
                      layer2.Padding().PadWLeft());
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadWRight(),
                      layer2.Padding().PadWRight());
  BOOST_REQUIRE_EQUAL(layer1.DilationWidth(), layer2.DilationWidth());
  BOOST_REQUIRE_EQUAL(layer1.DilationHeight(), layer2.DilationHeight());
}

/**
 * Test that the padding options are working correctly in Atrous Convolution
 * layer.
 */
BOOST_AUTO_TEST_CASE(AtrousConvolutionLayerPaddingTest)
{
  arma::mat output, input, delta;

  // Check valid padding option.
  AtrousConvolution<> module1(1, 1, 3, 3, 1, 1,
      std::tuple<size_t, size_t>(1, 1), std::tuple<size_t, size_t>(1, 1), 7, 7,
      2, 2, "valid");

  // Test the Forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Reset();
  module1.Forward(input, output);

  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, 9);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // Test the Backward function.
  module1.Backward(input, output, delta);

  // Check same padding option.
  AtrousConvolution<> module2(1, 1, 3, 3, 1, 1,
      std::tuple<size_t, size_t>(0, 0), std::tuple<size_t, size_t>(0, 0), 7, 7,
      2, 2, "same");

  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module2.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module2.Reset();
  module2.Forward(input, output);

  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, 49);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // Test the backward function.
  module2.Backward(input, output, delta);
}

/**
 * Tests the LayerNorm layer.
 */
BOOST_AUTO_TEST_CASE(LayerNormTest)
{
  arma::mat input, output;
  input << 5.1 << 3.5 << arma::endr
        << 4.9 << 3.0 << arma::endr
        << 4.7 << 3.2 << arma::endr;

  LayerNorm<> model(input.n_rows);
  model.Reset();

  model.Forward(input, output);
  arma::mat result;
  result << 1.2247 << 1.2978 << arma::endr
         << 0 << -1.1355 << arma::endr
         << -1.2247 << -0.1622 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  output = model.Mean();
  result << 4.9000 << 3.2333 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  output = model.Variance();
  result << 0.0267 << 0.0422 << arma::endr;

  CheckMatrices(output, result, 1e-1);
}

/**
 * LayerNorm layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientLayerNormTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randn(10, 256);
      arma::mat target;
      target.ones(1, 256);

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      model->Add<LayerNorm<> >(10);
      model->Add<Linear<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 256, false);
      model->Gradient(model->Parameters(), 0, gradient, 256);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Test that the functions that can access the parameters of the
 * Layer Norm layer work.
 */
BOOST_AUTO_TEST_CASE(LayerNormLayerParametersTest)
{
  // Parameter order : size, eps.
  LayerNorm<> layer(5, 1e-3);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer.InSize(), 5);
  BOOST_REQUIRE_EQUAL(layer.Epsilon(), 1e-3);
}

/**
 * Test if the AddMerge layer is able to forward the
 * Forward/Backward/Gradient calls.
 */
BOOST_AUTO_TEST_CASE(AddMergeRunTest)
{
  arma::mat output, input, delta, error;

  AddMerge<> module(true, true);

  Linear<>* linear = new Linear<>(10, 10);
  module.Add(linear);

  linear->Parameters().randu();
  linear->Reset();

  input = arma::zeros(10, 1);
  module.Forward(input, output);

  double parameterSum = arma::accu(linear->Parameters().submat(
      100, 0, linear->Parameters().n_elem - 1, 0));

  // Test the Backward function.
  module.Backward(input, input, delta);

  // Clean up before we break,
  delete linear;

  BOOST_REQUIRE_CLOSE(parameterSum, arma::accu(output), 1e-3);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Test if the MultiplyMerge layer is able to forward the
 * Forward/Backward/Gradient calls.
 */
BOOST_AUTO_TEST_CASE(MultiplyMergeRunTest)
{
  arma::mat output, input, delta, error;

  MultiplyMerge<> module(true, true);

  Linear<>* linear = new Linear<>(10, 10);
  module.Add(linear);

  linear->Parameters().randu();
  linear->Reset();

  input = arma::zeros(10, 1);
  module.Forward(input, output);

  double parameterSum = arma::accu(linear->Parameters().submat(
      100, 0, linear->Parameters().n_elem - 1, 0));

  // Test the Backward function.
  module.Backward(input, input, delta);

  // Clean up before we break,
  delete linear;

  BOOST_REQUIRE_CLOSE(parameterSum, arma::accu(output), 1e-3);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Simple subview module test.
 */
BOOST_AUTO_TEST_CASE(SimpleSubviewLayerTest)
{
  arma::mat output, input, delta, outputMat;
  Subview<> moduleRow(1, 10, 19);

  // Test the Forward function for a vector.
  input = arma::ones(20, 1);
  moduleRow.Forward(input, output);
  BOOST_REQUIRE_EQUAL(output.n_rows, 10);

  Subview<> moduleMat(4, 3, 6, 0, 2);

  // Test the Forward function for a matrix.
  input = arma::ones(20, 8);
  moduleMat.Forward(input, outputMat);
  BOOST_REQUIRE_EQUAL(outputMat.n_rows, 12);
  BOOST_REQUIRE_EQUAL(outputMat.n_cols, 2);

  // Test the Backward function.
  moduleMat.Backward(input, input, delta);
  BOOST_REQUIRE_EQUAL(accu(delta), 160);
  BOOST_REQUIRE_EQUAL(delta.n_rows, 20);
}

/**
 * Subview index test.
 */
BOOST_AUTO_TEST_CASE(SubviewIndexTest)
{
  arma::mat outputEnd, outputMid, outputStart, input, delta;
  input = arma::linspace<arma::vec>(1, 20, 20);

  // Slicing from the initial indices.
  Subview<> moduleStart(1, 0, 9);
  arma::mat subStart = arma::linspace<arma::vec>(1, 10, 10);

  moduleStart.Forward(input, outputStart);
  CheckMatrices(outputStart, subStart);

  // Slicing from the mid indices.
  Subview<> moduleMid(1, 6, 15);
  arma::mat subMid = arma::linspace<arma::vec>(7, 16, 10);

  moduleMid.Forward(input, outputMid);
  CheckMatrices(outputMid, subMid);

  // Slicing from the end indices.
  Subview<> moduleEnd(1, 10, 19);
  arma::mat subEnd = arma::linspace<arma::vec>(11, 20, 10);

  moduleEnd.Forward(input, outputEnd);
  CheckMatrices(outputEnd, subEnd);
}

/**
 * Subview batch test.
 */
BOOST_AUTO_TEST_CASE(SubviewBatchTest)
{
  arma::mat output, input, outputCol, outputMat, outputDef;

  // All rows selected.
  Subview<> moduleCol(1, 0, 19);

  // Test with inSize 1.
  input = arma::ones(20, 8);
  moduleCol.Forward(input, outputCol);
  CheckMatrices(outputCol, input);

  // Few rows and columns selected.
  Subview<> moduleMat(4, 3, 6, 0, 2);

  // Test with inSize greater than 1.
  moduleMat.Forward(input, outputMat);
  output = arma::ones(12, 2);
  CheckMatrices(outputMat, output);

  // endCol changed to 3 by default.
  Subview<> moduleDef(4, 1, 6, 0, 4);

  // Test with inSize greater than 1 and endCol >= inSize.
  moduleDef.Forward(input, outputDef);
  output = arma::ones(24, 2);
  CheckMatrices(outputDef, output);
}

/**
 * Test that the functions that can modify and access the parameters of the
 * Subview layer work.
 */
BOOST_AUTO_TEST_CASE(SubviewLayerParametersTest)
{
  // Parameter order : inSize, beginRow, endRow, beginCol, endCol.
  Subview<> layer1(1, 2, 3, 4, 5);
  Subview<> layer2(1, 3, 4, 5, 6);

  // Make sure we can get the parameters correctly.
  BOOST_REQUIRE_EQUAL(layer1.InSize(), 1);
  BOOST_REQUIRE_EQUAL(layer1.BeginRow(), 2);
  BOOST_REQUIRE_EQUAL(layer1.EndRow(), 3);
  BOOST_REQUIRE_EQUAL(layer1.BeginCol(), 4);
  BOOST_REQUIRE_EQUAL(layer1.EndCol(), 5);

  // Now modify the parameters to match the second layer.
  layer1.BeginRow() = 3;
  layer1.EndRow() = 4;
  layer1.BeginCol() = 5;
  layer1.EndCol() = 6;

  // Now ensure all results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InSize(), layer2.InSize());
  BOOST_REQUIRE_EQUAL(layer1.BeginRow(), layer2.BeginRow());
  BOOST_REQUIRE_EQUAL(layer1.EndRow(), layer2.EndRow());
  BOOST_REQUIRE_EQUAL(layer1.BeginCol(), layer2.BeginCol());
  BOOST_REQUIRE_EQUAL(layer1.EndCol(), layer2.EndCol());
}

/*
 * Simple Reparametrization module test.
 */
BOOST_AUTO_TEST_CASE(SimpleReparametrizationLayerTest)
{
  arma::mat input, output, delta;
  Reparametrization<> module(5);

  // Test the Forward function. As the mean is zero and the standard
  // deviation is small, after multiplying the gaussian sample, the
  // output should be small enough.
  input = join_cols(arma::ones<arma::mat>(5, 1) * -15,
      arma::zeros<arma::mat>(5, 1));
  module.Forward(input, output);
  BOOST_REQUIRE_LE(arma::accu(output), 1e-5);

  // Test the Backward function.
  arma::mat gy = arma::zeros<arma::mat>(5, 1);
  module.Backward(input, gy, delta);
  BOOST_REQUIRE(arma::accu(delta) != 0); // klBackward will be added.
}

/**
 * Reparametrization module stochastic boolean test.
 */
BOOST_AUTO_TEST_CASE(ReparametrizationLayerStochasticTest)
{
  arma::mat input, outputA, outputB;
  Reparametrization<> module(5, false);

  input = join_cols(arma::ones<arma::mat>(5, 1),
      arma::zeros<arma::mat>(5, 1));

  // Test if two forward passes generate same output.
  module.Forward(input, outputA);
  module.Forward(input, outputB);

  CheckMatrices(outputA, outputB);
}

/**
 * Reparametrization module includeKl boolean test.
 */
BOOST_AUTO_TEST_CASE(ReparametrizationLayerIncludeKlTest)
{
  arma::mat input, output, gy, delta;
  Reparametrization<> module(5, true, false);

  input = join_cols(arma::ones<arma::mat>(5, 1),
      arma::zeros<arma::mat>(5, 1));
  module.Forward(input, output);

  // As KL divergence is not included, with the above inputs, the delta
  // matrix should be all zeros.
  gy = arma::zeros(output.n_rows, output.n_cols);
  module.Backward(output, gy, delta);

  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Jacobian Reparametrization module test.
 */
BOOST_AUTO_TEST_CASE(JacobianReparametrizationLayerTest)
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElementsHalf = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElementsHalf * 2, 1);

    Reparametrization<> module(inputElementsHalf, false, false);

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Reparametrization layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientReparametrizationLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 6);
      model->Add<Reparametrization<> >(3, false, true, 1);
      model->Add<Linear<> >(3, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Reparametrization layer beta numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientReparametrizationLayerBetaTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 2);
      target = arma::mat("1 1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 6);
      // Use a value of beta not equal to 1.
      model->Add<Reparametrization<> >(3, false, true, 2);
      model->Add<Linear<> >(3, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Test that the functions that can access the parameters of the
 * Reparametrization layer work.
 */
BOOST_AUTO_TEST_CASE(ReparametrizationLayerParametersTest)
{
  // Parameter order : latentSize, stochastic, includeKL, beta.
  Reparametrization<> layer(5, false, false, 2);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer.OutputSize(), 5);
  BOOST_REQUIRE_EQUAL(layer.Stochastic(), false);
  BOOST_REQUIRE_EQUAL(layer.IncludeKL(), false);
  BOOST_REQUIRE_EQUAL(layer.Beta(), 2);
}

/**
 * Simple residual module test.
 */
BOOST_AUTO_TEST_CASE(SimpleResidualLayerTest)
{
  arma::mat outputA, outputB, input, deltaA, deltaB;

  Sequential<>* sequential = new Sequential<>(true);
  Residual<>* residual = new Residual<>(true);

  Linear<>* linearA = new Linear<>(10, 10);
  linearA->Parameters().randu();
  linearA->Reset();
  Linear<>* linearB = new Linear<>(10, 10);
  linearB->Parameters().randu();
  linearB->Reset();

  // Add the same layers (with the same parameters) to both Sequential and
  // Residual object.
  sequential->Add(linearA);
  sequential->Add(linearB);

  residual->Add(linearA);
  residual->Add(linearB);

  // Test the Forward function (pass the same input to both).
  input = arma::randu(10, 1);
  sequential->Forward(input, outputA);
  residual->Forward(input, outputB);

  CheckMatrices(outputA, outputB - input);

  // Test the Backward function (pass the same error to both).
  sequential->Backward(input, input, deltaA);
  residual->Backward(input, input, deltaB);

  CheckMatrices(deltaA, deltaB - input);

  delete sequential;
  delete residual;
  delete linearA;
  delete linearB;
}

/**
 * Simple Highway module test.
 */
BOOST_AUTO_TEST_CASE(SimpleHighwayLayerTest)
{
  arma::mat outputA, outputB, input, deltaA, deltaB;
  Sequential<>* sequential = new Sequential<>(true);
  Highway<>* highway = new Highway<>(10, true);
  highway->Parameters().zeros();
  highway->Reset();

  Linear<>* linearA = new Linear<>(10, 10);
  linearA->Parameters().randu();
  linearA->Reset();
  Linear<>* linearB = new Linear<>(10, 10);
  linearB->Parameters().randu();
  linearB->Reset();

  // Add the same layers (with the same parameters) to both Sequential and
  // Highway object.
  highway->Add(linearA);
  highway->Add(linearB);
  sequential->Add(linearA);
  sequential->Add(linearB);

  // Test the Forward function (pass the same input to both).
  input = arma::randu(10, 1);
  sequential->Forward(input, outputA);
  highway->Forward(input, outputB);

  CheckMatrices(outputB, input * 0.5 + outputA * 0.5);

  delete sequential;
  delete highway;
  delete linearA;
  delete linearB;
}

/**
 * Test that the function that can access the inSize parameter of the
 * Highway layer works.
 */
BOOST_AUTO_TEST_CASE(HighwayLayerParametersTest)
{
  // Parameter order : inSize, model.
  Highway<> layer(1, true);

  // Make sure we can get the parameter successfully.
  BOOST_REQUIRE_EQUAL(layer.InSize(), 1);
}

/**
 * Sequential layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientHighwayLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(5, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(5, 10);

      highway = new Highway<>(10);
      highway->Add<Linear<> >(10, 10);
      highway->Add<ReLULayer<> >();
      highway->Add<Linear<> >(10, 10);
      highway->Add<ReLULayer<> >();

      model->Add(highway);
      model->Add<Linear<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    Highway<>* highway;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Sequential layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientSequentialLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      sequential = new Sequential<>();
      sequential->Add<Linear<> >(10, 10);
      sequential->Add<ReLULayer<> >();
      sequential->Add<Linear<> >(10, 5);
      sequential->Add<ReLULayer<> >();

      model->Add(sequential);
      model->Add<Linear<> >(5, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    Sequential<>* sequential;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * WeightNorm layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientWeightNormLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<Linear<> >(10, 10);

      Linear<>* linear = new Linear<>(10, 2);
      weightNorm = new WeightNorm<>(linear);

      model->Add(weightNorm);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    WeightNorm<>* weightNorm;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Test if the WeightNorm layer is able to forward the
 * Forward/Backward/Gradient calls.
 */
BOOST_AUTO_TEST_CASE(WeightNormRunTest)
{
  arma::mat output, input, delta, error;

  Linear<>* linear = new Linear<>(10, 10);

  WeightNorm<> module(linear);

  module.Parameters().randu();
  module.Reset();

  linear->Bias().zeros();

  input = arma::zeros(10, 1);
  module.Forward(input, output);

  // Test the Backward function.
  module.Backward(input, input, delta);

  BOOST_REQUIRE_EQUAL(0, arma::accu(output));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

// General ANN serialization test.
template<typename LayerType>
void ANNLayerSerializationTest(LayerType& layer)
{
  arma::mat input(5, 100, arma::fill::randu);
  arma::mat output(5, 100, arma::fill::randu);

  FFN<NegativeLogLikelihood<>, ann::RandomInitialization> model;
  model.Add<Linear<>>(input.n_rows, 10);
  model.Add<LayerType>(layer);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(10, output.n_rows);
  model.Add<LogSoftMax<>>();

  ens::StandardSGD opt(0.1, 1, 5, -100, false);
  model.Train(input, output, opt);

  arma::mat originalOutput;
  model.Predict(input, originalOutput);

  // Now serialize the model.
  FFN<NegativeLogLikelihood<>, ann::RandomInitialization> xmlModel, textModel,
      binaryModel;
  SerializeObjectAll(model, xmlModel, textModel, binaryModel);

  // Ensure that predictions are the same.
  arma::mat modelOutput, xmlOutput, textOutput, binaryOutput;
  model.Predict(input, modelOutput);
  xmlModel.Predict(input, xmlOutput);
  textModel.Predict(input, textOutput);
  binaryModel.Predict(input, binaryOutput);

  CheckMatrices(originalOutput, modelOutput, 1e-5);
  CheckMatrices(originalOutput, xmlOutput, 1e-5);
  CheckMatrices(originalOutput, textOutput, 1e-5);
  CheckMatrices(originalOutput, binaryOutput, 1e-5);
}

/**
 * Simple serialization test for batch normalization layer.
 */
BOOST_AUTO_TEST_CASE(BatchNormSerializationTest)
{
  BatchNorm<> layer(10);
  ANNLayerSerializationTest(layer);
}

/**
 * Simple serialization test for layer normalization layer.
 */
BOOST_AUTO_TEST_CASE(LayerNormSerializationTest)
{
  LayerNorm<> layer(10);
  ANNLayerSerializationTest(layer);
}

/**
 * Test that the functions that can modify and access the parameters of the
 * Convolution layer work.
 */
BOOST_AUTO_TEST_CASE(ConvolutionLayerParametersTest)
{
  // Parameter order: inSize, outSize, kW, kH, dW, dH, padW, padH, inputWidth,
  // inputHeight, paddingType.
  Convolution<> layer1(1, 2, 3, 4, 5, 6, std::tuple<size_t, size_t>(7, 8),
      std::tuple<size_t, size_t>(9, 10), 11, 12, "none");
  Convolution<> layer2(2, 3, 4, 5, 6, 7, std::tuple<size_t, size_t>(8, 9),
      std::tuple<size_t, size_t>(10, 11), 12, 13, "none");

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), 11);
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), 12);
  BOOST_REQUIRE_EQUAL(layer1.KernelWidth(), 3);
  BOOST_REQUIRE_EQUAL(layer1.KernelHeight(), 4);
  BOOST_REQUIRE_EQUAL(layer1.StrideWidth(), 5);
  BOOST_REQUIRE_EQUAL(layer1.StrideHeight(), 6);
  BOOST_REQUIRE_EQUAL(layer1.PadWLeft(), 7);
  BOOST_REQUIRE_EQUAL(layer1.PadWRight(), 8);
  BOOST_REQUIRE_EQUAL(layer1.PadHTop(), 9);
  BOOST_REQUIRE_EQUAL(layer1.PadHBottom(), 10);

  // Now modify the parameters to match the second layer.
  layer1.InputWidth() = 12;
  layer1.InputHeight() = 13;
  layer1.KernelWidth() = 4;
  layer1.KernelHeight() = 5;
  layer1.StrideWidth() = 6;
  layer1.StrideHeight() = 7;
  layer1.PadWLeft() = 8;
  layer1.PadWRight() = 9;
  layer1.PadHTop() = 10;
  layer1.PadHBottom() = 11;

  // Now ensure all results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), layer2.InputWidth());
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), layer2.InputHeight());
  BOOST_REQUIRE_EQUAL(layer1.KernelWidth(), layer2.KernelWidth());
  BOOST_REQUIRE_EQUAL(layer1.KernelHeight(), layer2.KernelHeight());
  BOOST_REQUIRE_EQUAL(layer1.StrideWidth(), layer2.StrideWidth());
  BOOST_REQUIRE_EQUAL(layer1.StrideHeight(), layer2.StrideHeight());
  BOOST_REQUIRE_EQUAL(layer1.PadWLeft(), layer2.PadWLeft());
  BOOST_REQUIRE_EQUAL(layer1.PadWRight(), layer2.PadWRight());
  BOOST_REQUIRE_EQUAL(layer1.PadHTop(), layer2.PadHTop());
  BOOST_REQUIRE_EQUAL(layer1.PadHBottom(), layer2.PadHBottom());
}

/**
 * Test that the padding options are working correctly in Convolution layer.
 */
BOOST_AUTO_TEST_CASE(ConvolutionLayerPaddingTest)
{
  arma::mat output, input, delta;

  // Check valid padding option.
  Convolution<> module1(1, 1, 3, 3, 1, 1, std::tuple<size_t, size_t>(1, 1),
      std::tuple<size_t, size_t>(1, 1), 7, 7, "valid");

  // Test the Forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Reset();
  module1.Forward(input, output);

  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, 25);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // Test the Backward function.
  module1.Backward(input, output, delta);

  // Check same padding option.
  Convolution<> module2(1, 1, 3, 3, 1, 1, std::tuple<size_t, size_t>(0, 0),
      std::tuple<size_t, size_t>(0, 0), 7, 7, "same");

  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module2.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module2.Reset();
  module2.Forward(input, output);

  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, 49);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // Test the backward function.
  module2.Backward(input, output, delta);
}

/**
 * Test that the padding options in Transposed Convolution layer.
 */
BOOST_AUTO_TEST_CASE(TransposedConvolutionLayerPaddingTest)
{
  arma::mat output, input, delta;

  TransposedConvolution<> module1(1, 1, 3, 3, 1, 1, 0, 0, 4, 4, 6, 6, "VALID");
  // Test the forward function.
  // Valid Should give the same result.
  input = arma::linspace<arma::colvec>(0, 15, 16);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Reset();
  module1.Forward(input, output);
  // Value calculated using tensorflow.nn.conv2d_transpose().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0.0);

  // Test the Backward Function.
  module1.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);

  // Test Valid for non zero padding.
  TransposedConvolution<> module2(1, 1, 3, 3, 2, 2,
      std::tuple<size_t, size_t>(0, 0), std::tuple<size_t, size_t>(0, 0),
      2, 2, 5, 5, "VALID");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module2.Parameters() = arma::mat(25 + 1, 1, arma::fill::zeros);
  module2.Parameters()(2) = 8.0;
  module2.Parameters()(4) = 6.0;
  module2.Parameters()(6) = 4.0;
  module2.Parameters()(8) = 2.0;
  module2.Reset();
  module2.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 120.0);

  // Test the Backward Function.
  module2.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 960.0);

  // Test for same padding type.
  TransposedConvolution<> module3(1, 1, 3, 3, 2, 2, 0, 0, 3, 3, 3, 3, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module3.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module3.Reset();
  module3.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the Backward Function.
  module3.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);

  // Output shape should equal input.
  TransposedConvolution<> module4(1, 1, 3, 3, 1, 1,
    std::tuple<size_t, size_t>(2, 2), std::tuple<size_t, size_t>(2, 2),
    5, 5, 5, 5, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module4.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module4.Reset();
  module4.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the Backward Function.
  module4.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);

  TransposedConvolution<> module5(1, 1, 3, 3, 2, 2, 0, 0, 2, 2, 2, 2, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module5.Parameters() = arma::mat(25 + 1, 1, arma::fill::zeros);
  module5.Reset();
  module5.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the Backward Function.
  module5.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);

  TransposedConvolution<> module6(1, 1, 4, 4, 1, 1, 1, 1, 5, 5, 5, 5, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module6.Parameters() = arma::mat(16 + 1, 1, arma::fill::zeros);
  module6.Reset();
  module6.Forward(input, output);
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the Backward Function.
  module6.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);
}

/**
 * Simple test for Max Pooling layer.
 */
BOOST_AUTO_TEST_CASE(MaxPoolingTestCase)
{
  // For rectangular input to pooling layers.
  arma::mat input = arma::mat(12, 1);
  arma::mat output;
  input.zeros();
  input(0) = 1;
  input(1) = 2;
  input(2) = 3;
  input(3) = input(8) = 7;
  input(4) = 4;
  input(5) = 5;
  input(6) = input(7) = 6;
  input(10) = 8;
  input(11) = 9;
  // Output-Size should be 2 x 2.
  // Square output.
  MaxPooling<> module1(2, 2, 2, 1);
  module1.InputHeight() = 3;
  module1.InputWidth() = 4;
  module1.Forward(input, output);
  // Calculated using torch.nn.MaxPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 28);
  BOOST_REQUIRE_EQUAL(output.n_elem, 4);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // For Square input.
  input = arma::mat(9, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(3) = 3;
  input(6) = 3;
  // Output-Size should be 1 x 2.
  // Rectangular output.
  MaxPooling<> module2(3, 2, 3, 1);
  module2.InputHeight() = 3;
  module2.InputWidth() = 3;
  module2.Forward(input, output);
  // Calculated using torch.nn.MaxPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 12.0);
  BOOST_REQUIRE_EQUAL(output.n_elem, 2);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // For Square input.
  input = arma::mat(16, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(4) = 3;
  input(8) = 3;
  // Output-Size should be 3 x 3.
  // Square output.
  MaxPooling<> module3(2, 2, 1, 1);
  module3.InputHeight() = 4;
  module3.InputWidth() = 4;
  module3.Forward(input, output);
  // Calculated using torch.nn.MaxPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 30.0);
  BOOST_REQUIRE_EQUAL(output.n_elem, 9);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // For Rectangular input.
  input = arma::mat(6, 1);
  input.zeros();
  input(0) = 1;
  input(1) = 1;
  input(3) = 1;
  // Output-Size should be 2 x 2.
  // Square output.
  MaxPooling<> module4(2, 1, 1, 1);
  module4.InputHeight() = 2;
  module4.InputWidth() = 3;
  module4.Forward(input, output);
  // Calculated using torch.nn.MaxPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 3);
  BOOST_REQUIRE_EQUAL(output.n_elem, 4);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
}

/**
 * Test that the functions that can modify and access the parameters of the
 * Glimpse layer work.
 */
BOOST_AUTO_TEST_CASE(GlimpseLayerParametersTest)
{
  // Parameter order : inSize, size, depth, scale, inputWidth, inputHeight.
  Glimpse<> layer1(1, 2, 3, 4, 5, 6);
  Glimpse<> layer2(1, 2, 3, 4, 6, 7);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), 6);
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), 5);
  BOOST_REQUIRE_EQUAL(layer1.Scale(), 4);
  BOOST_REQUIRE_EQUAL(layer1.Depth(), 3);
  BOOST_REQUIRE_EQUAL(layer1.GlimpseSize(), 2);
  BOOST_REQUIRE_EQUAL(layer1.InSize(), 1);

  // Now modify the parameters to match the second layer.
  layer1.InputHeight() = 7;
  layer1.InputWidth() = 6;

  // Now ensure that all the results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), layer2.InputHeight());
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), layer2.InputWidth());
  BOOST_REQUIRE_EQUAL(layer1.Scale(), layer2.Scale());
  BOOST_REQUIRE_EQUAL(layer1.Depth(), layer2.Depth());
  BOOST_REQUIRE_EQUAL(layer1.GlimpseSize(), layer2.GlimpseSize());
  BOOST_REQUIRE_EQUAL(layer1.InSize(), layer2.InSize());
}

/**
 * Test that the function that can access the stdev parameter of the
 * Reinforce Normal layer works.
 */
BOOST_AUTO_TEST_CASE(ReinforceNormalLayerParametersTest)
{
  // Parameter : stdev.
  ReinforceNormal<> layer(4.0);

  // Make sure we can get the parameter successfully.
  BOOST_REQUIRE_EQUAL(layer.StandardDeviation(), 4.0);
}

/**
 * Test that the function that can access the parameters of the
 * VR Class Reward layer works.
 */
BOOST_AUTO_TEST_CASE(VRClassRewardLayerParametersTest)
{
  // Parameter order : scale, sizeAverage.
  VRClassReward<> layer(2, false);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer.Scale(), 2);
  BOOST_REQUIRE_EQUAL(layer.SizeAverage(), false);
}

/**
 * Simple test for Adaptive pooling for Max Pooling layer.
 */
BOOST_AUTO_TEST_CASE(AdaptiveMaxPoolingTestCase)
{
  // For rectangular input.
  arma::mat input = arma::mat(12, 1);
  arma::mat output, delta;

  input.zeros();
  input(0) = 1;
  input(1) = 2;
  input(2) = 3;
  input(3) = input(8) = 7;
  input(4) = 4;
  input(5) = 5;
  input(6) = input(7) = 6;
  input(10) = 8;
  input(11) = 9;
  // Output-Size should be 2 x 2.
  // Square output.
  AdaptiveMaxPooling<> module1(2, 2);
  module1.InputHeight() = 3;
  module1.InputWidth() = 4;
  module1.Forward(input, output);
  // Calculated using torch.nn.AdaptiveMaxPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 28);
  BOOST_REQUIRE_EQUAL(output.n_elem, 4);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  // Test the Backward Function.
  module1.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 28.0);

  // For Square input.
  input = arma::mat(9, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(3) = 3;
  input(6) = 3;
  // Output-Size should be 1 x 2.
  // Rectangular output.
  AdaptiveMaxPooling<> module2(2, 1);
  module2.InputHeight() = 3;
  module2.InputWidth() = 3;
  module2.Forward(input, output);
  // Calculated using torch.nn.AdaptiveMaxPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 15.0);
  BOOST_REQUIRE_EQUAL(output.n_elem, 2);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  // Test the Backward Function.
  module2.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 15.0);

  // For Square input.
  input = arma::mat(16, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(4) = 3;
  input(8) = 3;
  // Output-Size should be 3 x 3.
  // Square output.
  AdaptiveMaxPooling<> module3(std::tuple<size_t, size_t>(3, 3));
  module3.InputHeight() = 4;
  module3.InputWidth() = 4;
  module3.Forward(input, output);
  // Calculated using torch.nn.AdaptiveMaxPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 30.0);
  BOOST_REQUIRE_EQUAL(output.n_elem, 9);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  // Test the Backward Function.
  module3.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 30.0);

  // For Rectangular input.
  input = arma::mat(20, 1);
  input.zeros();
  input(0) = 1;
  input(1) = 1;
  input(3) = 1;
  // Output-Size should be 2 x 2.
  // Square output.
  AdaptiveMaxPooling<> module4(std::tuple<size_t, size_t>(2, 2));
  module4.InputHeight() = 4;
  module4.InputWidth() = 5;
  module4.Forward(input, output);
  // Calculated using torch.nn.AdaptiveMaxPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 2);
  BOOST_REQUIRE_EQUAL(output.n_elem, 4);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  // Test the Backward Function.
  module4.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 2.0);
}

/**
 * Simple test for Adaptive pooling for Mean Pooling layer.
 */
BOOST_AUTO_TEST_CASE(AdaptiveMeanPoolingTestCase)
{
  // For rectangular input.
  arma::mat input = arma::mat(12, 1);
  arma::mat output, delta;

  input.zeros();
  input(0) = 1;
  input(1) = 2;
  input(2) = 3;
  input(3) = input(8) = 7;
  input(4) = 4;
  input(5) = 5;
  input(6) = input(7) = 6;
  input(10) = 8;
  input(11) = 9;
  // Output-Size should be 2 x 2.
  // Square output.
  AdaptiveMeanPooling<> module1(2, 2);
  module1.InputHeight() = 3;
  module1.InputWidth() = 4;
  module1.Forward(input, output);
  // Calculated using torch.nn.AdaptiveAvgPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 19.75);
  BOOST_REQUIRE_EQUAL(output.n_elem, 4);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  // Test the Backward Function.
  module1.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 7.0);

  // For Square input.
  input = arma::mat(9, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(3) = 3;
  input(6) = 3;
  // Output-Size should be 1 x 2.
  // Rectangular output.
  AdaptiveMeanPooling<> module2(1, 2);
  module2.InputHeight() = 3;
  module2.InputWidth() = 3;
  module2.Forward(input, output);
  // Calculated using torch.nn.AdaptiveAvgPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 4.5);
  BOOST_REQUIRE_EQUAL(output.n_elem, 2);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  // Test the Backward Function.
  module2.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);

  // For Square input.
  input = arma::mat(16, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(4) = 3;
  input(8) = 3;
  // Output-Size should be 3 x 3.
  // Square output.
  AdaptiveMeanPooling<> module3(std::tuple<size_t, size_t>(3, 3));
  module3.InputHeight() = 4;
  module3.InputWidth() = 4;
  module3.Forward(input, output);
  // Calculated using torch.nn.AdaptiveAvgPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 10.5);
  BOOST_REQUIRE_EQUAL(output.n_elem, 9);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  // Test the Backward Function.
  module3.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 10.5);

  // For Rectangular input.
  input = arma::mat(24, 1);
  input.zeros();
  input(0) = 3;
  input(1) = 3;
  input(4) = 3;
  // Output-Size should be 3 x 3.
  // Square output.
  AdaptiveMeanPooling<> module4(std::tuple<size_t, size_t>(3, 3));
  module4.InputHeight() = 4;
  module4.InputWidth() = 6;
  module4.Forward(input, output);
  // Calculated using torch.nn.AdaptiveAvgPool2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 2.25);
  BOOST_REQUIRE_EQUAL(output.n_elem, 9);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  // Test the Backward Function.
  module4.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 1.5);
}

BOOST_AUTO_TEST_CASE(TransposedConvolutionalLayerOptionalParameterTest)
{
  Sequential<>* decoder = new Sequential<>();

  // Check if we can create an object without specifying output.
  BOOST_REQUIRE_NO_THROW(decoder->Add<TransposedConvolution<>>(24, 16,
      5, 5, 1, 1, 0, 0, 10, 10));

  BOOST_REQUIRE_NO_THROW(decoder->Add<TransposedConvolution<>>(16, 1,
      15, 15, 1, 1, 1, 1, 14, 14));

    delete decoder;
}

BOOST_AUTO_TEST_CASE(BatchNormWithMinBatchesTest)
{
  arma::mat input, output, result, runningMean, runningVar, delta;

  // The input test matrix is of the form 3 x 2 x 4 x 1 where
  // number of images are 3 and number of feature maps are 2.
  input = arma::mat(8, 3);
  input << 1 << 446 << 42 << arma::endr
      << 2 << 16 << 63 << arma::endr
      << 3 << 13 << 63 << arma::endr
      << 4 << 21 << 21 << arma::endr
      << 1 << 13 << 11 << arma::endr
      << 32 << 45 << 42 << arma::endr
      << 22 << 16 << 63 << arma::endr
      << 32 << 13 << 42 << arma::endr;

  // Output calculated using torch.nn.BatchNorm2d().
  result = arma::mat(8, 3);
  result << -0.4786 << 3.2634 << -0.1338 << arma::endr
      << -0.4702 << -0.3525 << 0.0427 << arma::endr
      << -0.4618 << -0.3777 << 0.0427 << arma::endr
      << -0.4534 << -0.3104 << -0.3104 << arma::endr
      << -1.5429 << -0.8486 << -0.9643 << arma::endr
      << 0.2507 << 1.0029 << 0.8293 << arma::endr
      << -0.3279 << -0.675 << 2.0443 << arma::endr
      << 0.2507 << -0.8486 << 0.8293 << arma::endr;

  // Check correctness of batch normalization.
  BatchNorm<> module1(2, 1e-5, false, 0.1);
  module1.Reset();
  module1.Forward(input, output);
  CheckMatrices(output, result, 1e-1);

  // Check backward function.
  module1.Backward(input, output, delta);
  BOOST_REQUIRE_CLOSE(arma::accu(delta), 0.0102676, 1e-3);

  // Check values for running mean and running variance.
  // Calculated using torch.nn.BatchNorm2d().
  runningMean = arma::mat(2, 1);
  runningVar = arma::mat(2, 1);
  runningMean(0) = 5.7917;
  runningMean(1) = 2.76667;
  runningVar(0) = 1543.6545;
  runningVar(1) = 33.488;

  CheckMatrices(runningMean, module1.TrainingMean(), 1e-3);
  CheckMatrices(runningVar, module1.TrainingVariance(), 1e-2);

  // Check correctness of layer when running mean and variance
  // are updated using cumulative average.
  BatchNorm<> module2(2);
  module2.Reset();
  module2.Forward(input, output);
  CheckMatrices(output, result, 1e-1);

  // Check values for running mean and running variance.
  // Calculated using torch.nn.BatchNorm2d().
  runningMean(0) = 57.9167;
  runningMean(1) = 27.6667;
  runningVar(0) = 15427.5380;
  runningVar(1) = 325.8787;

  CheckMatrices(runningMean, module2.TrainingMean(), 1e-2);
  CheckMatrices(runningVar, module2.TrainingVariance(), 1e-2);

  // Check correctness when model is testing.
  arma::mat deterministicOutput;
  module1.Deterministic() = true;
  module1.Forward(input, deterministicOutput);

  result.clear();
  result = arma::mat(8, 3);
  result << -0.12195 << 11.20426 << 0.92158 << arma::endr
      << -0.0965 << 0.259824 << 1.4560 << arma::endr
      << -0.071054 << 0.183567 << 1.45607 << arma::endr
      << -0.045601<< 0.3870852 << 0.38708 << arma::endr
      << -0.305288 << 1.7683 << 1.4227 << arma::endr
      << 5.05166 << 7.29812<< 6.7797 << arma::endr
      << 3.323614 << 2.2867 << 10.4086 << arma::endr
      << 5.05166 << 1.7683 << 6.7797 << arma::endr;

  CheckMatrices(result, deterministicOutput, 1e-1);

  // Check correctness by updating the running mean and variance again.
  module1.Deterministic() = false;

  // Clean up.
  output.clear();
  input.clear();

  // The input test matrix is of the form 2 x 2 x 3 x 1 where
  // number of images are 2 and number of feature maps are 2.
  input = arma::mat(6, 2);
  input << 12 << 443 << arma::endr
      << 134 << 45 << arma::endr
      << 11 << 13 << arma::endr
      << 14 << 55 << arma::endr
      << 110 << 4 << arma::endr
      << 1 << 45 << arma::endr;

  result << -0.629337 << 2.14791 << arma::endr
      << 0.156797 << -0.416694 << arma::endr
      << -0.63578 << -0.622893 << arma::endr
      << -0.637481 << 0.4440386 << arma::endr
      << 1.894857 << -0.901267 << arma::endr
      << -0.980402 << 0.180253 << arma::endr;

  module1.Forward(input, output);
  CheckMatrices(result, output, 1e-3);

  // Check correctness for the second module as well.
  module2.Forward(input, output);
  CheckMatrices(result, output, 1e-3);

  // Calculated using torch.nn.BatchNorm2d().
  runningMean(0) = 16.1792;
  runningMean(1) = 6.30667;
  runningVar(0) = 4276.5849;
  runningVar(1) = 202.595;

  CheckMatrices(runningMean, module1.TrainingMean(), 1e-3);
  CheckMatrices(runningVar, module1.TrainingVariance(), 1e-1);

  // Check correctness of running mean and variance when their
  // values are updated using cumulative average.
  runningMean(0) = 83.79166;
  runningMean(1) = 32.9166;
  runningVar(0) = 22164.1035;
  runningVar(1) = 1025.2227;

  CheckMatrices(runningMean, module2.TrainingMean(), 1e-3);
  CheckMatrices(runningVar, module2.TrainingVariance(), 1e-3);

  // Check backward function.
  module1.Backward(input, output, delta);

  deterministicOutput.clear();
  module1.Deterministic() = true;
  module1.Forward(input, deterministicOutput);

  result.clear();
  result << -0.06388436 << 6.524754114 << arma::endr
      << 1.799655281 << 0.44047968 << arma::endr
      << -0.07913291 << -0.04784981 << arma::endr
      << 0.5405045 << 3.4210097 << arma::endr
      << 7.2851023 << -0.1620577 << arma::endr
      << -0.37282639 << 2.7184474 << arma::endr;

  // Calculated using torch.nn.BatchNorm2d().
  CheckMatrices(result, deterministicOutput, 1e-1);
}

/**
 * Batch Normalization layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientBatchNormWithMiniBatchesTest)
{
  // Add function gradient instantiation.
  // To make this test robust, check it ten times.
  bool pass = false;
  for (size_t trial = 0; trial < 10; trial++)
  {
    struct GradientFunction
    {
      GradientFunction()
      {
        input = arma::randn(16, 1024);
        arma::mat target;
        target.ones(1, 1024);

        model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
        model->Predictors() = input;
        model->Responses() = target;
        model->Add<IdentityLayer<>>();
        model->Add<Convolution<>>(1, 2, 3, 3, 1, 1, 0, 0, 4, 4);
        model->Add<BatchNorm<>>(2);
        model->Add<Linear<>>(2 * 2 * 2, 2);
        model->Add<LogSoftMax<>>();
      }

      ~GradientFunction()
      {
        delete model;
      }

      double Gradient(arma::mat& gradient) const
      {
        double error = model->Evaluate(model->Parameters(), 0, 1024, false);
        model->Gradient(model->Parameters(), 0, gradient, 1024);
        return error;
      }

      arma::mat& Parameters() { return model->Parameters(); }

      FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
      arma::mat input, target;
    } function;

    double gradient = CheckGradient(function);
    if (gradient < 1e-1)
    {
      pass = true;
      break;
    }
  }

  BOOST_REQUIRE(pass);
}

BOOST_AUTO_TEST_CASE(ConvolutionLayerTestCase)
{
  arma::mat input, output;

  // The input test matrix is of the form 3 x 2 x 4 x 1 where
  // number of images are 3 and number of feature maps are 2.
  input = arma::mat(8, 3);
  input << 1 << 446 << 42 << arma::endr
      << 2 << 16 << 63 << arma::endr
      << 3 << 13 << 63 << arma::endr
      << 4 << 21 << 21 << arma::endr
      << 1 << 13 << 11 << arma::endr
      << 32 << 45 << 42 << arma::endr
      << 22 << 16 << 63 << arma::endr
      << 32 << 13 << 42 << arma::endr;

  Convolution<> layer(2, 4, 1, 1, 1, 1, 0, 0, 4, 1);
  layer.Reset();

  // Set weights to 1.0 and bias to 0.0.
  layer.Parameters().zeros();
  arma::mat weight(2 * 4, 1);
  weight.fill(1.0);
  layer.Parameters().submat(arma::span(0, 2 * 4 - 1), arma::span()) = weight;
  layer.Forward(input, output);

  // Value calculated using torch.nn.Conv2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 4108);

  // Set bias to one.
  layer.Parameters().fill(1.0);
  layer.Forward(input, output);

  // Value calculated using torch.nn.Conv2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 4156);
}

BOOST_AUTO_TEST_SUITE_END();
