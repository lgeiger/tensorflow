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

include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"


def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

// TODO: Check shapes before fusing
multiclass FuseAddOrSubWithBConv2D<dag binaryOp> {
  def : Pat<(binaryOp (TF_LqceBconv2d64Op:$output $input, $filter,
                          $fused_multiply, (ConstantOp F32ElementsAttr:$fused_add), $strides,
                          $padding, $explicit_paddings, $data_format,
                          $dilations, $filter_format),
                      (ConstantOp F32ElementsAttr:$value), TFL_AF_None),
            (TF_LqceBconv2d64Op $input, $filter, $fused_multiply,
                          (binaryOp (ConstantOp $fused_add), (ConstantOp $value), TFL_AF_None),
                          $strides, $padding, $explicit_paddings, $data_format,
                          $dilations, $filter_format),
            [(HasOneUse $output)]>;
}
foreach binaryOp = [TFL_AddOp, TFL_SubOp] in
  defm : FuseAddOrSubWithBConv2D<binaryOp>;

// TODO: Check shapes before fusing
multiclass FuseMulOrDivWithBConv2D<dag binaryOp> {
  def : Pat<(binaryOp (TF_LqceBconv2d64Op:$conv_output $input, $filter,
                        (ConstantOp F32ElementsAttr:$fused_multiply),
                        (ConstantOp F32ElementsAttr:$fused_add),
                        $strides, $padding, $explicit_paddings, $data_format,
                        $dilations, $filter_format),
                      (ConstantOp F32ElementsAttr:$value), $TFL_AF_None),
            (TF_LqceBconv2d64Op $input, $filter,
                        (binaryOp (ConstantOp $fused_multiply),
                                  (ConstantOp $value), TFL_AF_None),
                        (binaryOp (ConstantOp $fused_add),
                                  (ConstantOp $value), TFL_AF_None),
                        $strides, $padding, $explicit_paddings, $data_format,
                        $dilations, $filter_format),
         [(HasOneUse $conv_output)]>;
}

foreach binaryOp = [TFL_DivOp, TFL_MulOp] in
  defm : FuseMulOrDivWithBConv2D<binaryOp>;
