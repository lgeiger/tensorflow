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

include "tensorflow/compiler/mlir/lite/transforms/prepare_patterns.td"
include "tensorflow/compiler/mlir/lite/lce/ir/lce_ops.td"


// This relies on implementation details of larq.math.sign. We should make
// this more general in the future
multiclass FuseBSign<dag AddOp> {
  def : Pat<(TF_SignOp (AddOp (TF_SignOp $arg), $c)), (TF_LqceBsignOp $arg),
            [], (addBenefit 100)>;
  def : Pat<(TF_SignOp (AddOp $c, (TF_SignOp $arg))), (TF_LqceBsignOp $arg),
            [], (addBenefit 100)>;
}

foreach AddOp = [TF_AddOp, TF_AddV2Op] in
  defm : FuseBSign<AddOp>;

// Copied from legalize_patterns.td
class I32VectorElementsAttr<int len> : ElementsAttrBase<
  CPred<"$_self.isa<DenseIntElementsAttr>() &&"
      "$_self.cast<DenseIntElementsAttr>().getType()."
      "getElementType().isInteger(32)">,
  "32-bit int elements attribute of shape [" # len # "]"> {

  let storageType = [{ DenseIntElementsAttr }];
  let returnType = [{ DenseIntElementsAttr }];

  let constBuilderCall = "DenseElementsAttr::get("
    "RankedTensorType::get({" # len # "}, $_builder.getIntegerType(32)), $0)";
}

def GetBias : NativeCodeCall<"GetBias($0)">;
def GetMultiplier : NativeCodeCall<"GetMultiplier($0)">;
def BinaryFilter : Constraint<CPred<"IsBinaryFilter($0)">>;

def : Pat<(TF_Conv2DOp (TF_LqceBsignOp $input), $filter, $strides, $use_cudnn,
                       $padding, $explicit_padding, $data_format, $dilations),
          (TF_LqceBconv2d64Op $input,
            (TF_TransposeOp $filter, (ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{3, 0, 1, 2}">)),
            (TF_ConstOp (GetMultiplier $filter)),
            (TF_ConstOp (GetBias $filter)),
            $strides, $padding, $explicit_padding,
            $data_format, $dilations, ConstantAttr<StrAttr, "OHWI">),
          [(BinaryFilter $filter)], (addBenefit 90)>;
