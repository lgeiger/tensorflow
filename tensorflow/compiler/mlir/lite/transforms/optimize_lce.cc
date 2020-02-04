/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"          // TF:llvm-project
#include "mlir/Pass/Pass.h"                // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Optimize LCE operations in functions.
struct OptimizeLCE : public FunctionPass<OptimizeLCE> {
  void runOnFunction() override;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize_lce.inc"

void OptimizeLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect OptimizeLCE pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateOptimizeLCEPass() {
  return std::make_unique<OptimizeLCE>();
}

static PassRegistration<OptimizeLCE> pass(
    "tfl-optimize-lce", "Optimize within the TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
