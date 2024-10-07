/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/acceleration_test_util.h"
#include "tensorflow/lite/kernels/acceleration_test_util_internal.h"
#include "tensorflow/lite/kernels/test_delegate_providers.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

class DelegateTestSuiteAccelerationTestParams {
 public:
  static const char* AccelerationTestConfig() {
    return acceleration_test_config_->c_str();
  }
  static void SetAccelerationTestConfig(std::string config) {
    *acceleration_test_config_ = config;
  }
  static void Destroy() { delete acceleration_test_config_; }
  static DelegateTestSuiteAccelerationTestParams ParseConfigurationLine(
      const std::string& conf_line) {
    // No test argument is supported at the moment.
    return {};
  }

 private:
  static std::string* acceleration_test_config_;
};

std::string*
    DelegateTestSuiteAccelerationTestParams::acceleration_test_config_ =
        new std::string(
            R"(
## Every Test can be allowlisted or denylisted using a regexp on its test_id

## Test_id
#
# The test_id is test_suite_name / test_name, this differs from the
# name used by the build because of the / separator instead of .
# Parameterized tests names are composed by the base test name / test / ordinal
# the ordinal is the position in the list of parameters generated by the
# cardinal product of all the different parameter sets

# Denylist/Allowlist
# To denylist an element simply add - before the test_id regex

## Rules evaluation
#
# Rules are checked in order, the first matching completes the browsing
# This can be useful to put more specific rules first and generic default
# ones below

## Test Arguments
#
# No test argument is supported at the moment.

# DTS checks all tests by default if no acceleraton test config file is
# provided.
.*

)");

void ValidateAcceleration(const SingleOpModel& model) {
  std::string test_id = GetCurrentTestId();
  const bool supported =
      GetAccelerationTestParam<DelegateTestSuiteAccelerationTestParams>(test_id)
          .has_value();
  if (!supported) {
    // Note that the error `kTfLiteApplicationError` is accepted here.
    // We only want to check the delegate is working properly, so an error due
    // to incompatibility between the model and the delegate is not considered a
    // failure here.
    EXPECT_THAT(model.GetDelegateApplicationStatus().value_or(kTfLiteOk),
                testing::AnyOf(kTfLiteOk, kTfLiteApplicationError));
    return;
  } else {
    EXPECT_EQ(model.GetDelegateApplicationStatus().value_or(kTfLiteOk),
              kTfLiteOk);
  }

  // If we have multiple delegates applied, we would skip this check at the
  // moment.
  int num_applied_delegates = model.GetNumberOfAppliedDelegates();
  if (num_applied_delegates > 1) {
    TFLITE_LOG(WARN) << "Skipping acceleration validation as "
                     << num_applied_delegates
                     << " delegates have been successfully applied.";
    return;
  }
  TFLITE_LOG(INFO) << "Validating acceleration with the stable delegate";
  EXPECT_EQ(model.CountNumberOfDelegatedPartitions(), 1)
      << "Expecting operation to be accelerated but cannot find a partition "
         "associated to the stable delegate";
  EXPECT_GT(num_applied_delegates, 0) << "No delegates were applied.";
}

bool InitKernelTest(int* argc, char** argv) {
  KernelTestDelegateProviders* const delegate_providers =
      KernelTestDelegateProviders::Get();
  if (!delegate_providers->InitFromCmdlineArgs(
          argc, const_cast<const char**>(argv))) {
    return false;
  }
  const auto& delegate_params = delegate_providers->ConstParams();
  if (delegate_params.HasParam("stable_delegate_settings_file") &&
      !delegate_params.Get<std::string>("stable_delegate_settings_file")
           .empty()) {
    AccelerationValidator::Get()->AddCallback(ValidateAcceleration);
  }
  if (delegate_params.HasParam(
          KernelTestDelegateProviders::kAccelerationTestConfigPath)) {
    std::string acceleration_test_config_path =
        delegate_params.Get<std::string>(
            KernelTestDelegateProviders::kAccelerationTestConfigPath);
    if (acceleration_test_config_path.empty()) {
      return true;
    }
    std::ifstream fp(acceleration_test_config_path);
    if (!fp.good()) {
      return false;
    }
    DelegateTestSuiteAccelerationTestParams::SetAccelerationTestConfig(
        std::string(std::istreambuf_iterator<char>(fp),
                    std::istreambuf_iterator<char>()));
  }
  return true;
}

void DestroyKernelTest() { DelegateTestSuiteAccelerationTestParams::Destroy(); }

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  tflite::LogToStderr();
  if (tflite::InitKernelTest(&argc, argv)) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    tflite::DestroyKernelTest();
    return ret;
  } else {
    return EXIT_FAILURE;
  }
}
