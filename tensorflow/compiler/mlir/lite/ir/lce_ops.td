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

// This is the operation definition file for TensorFlow.
//
// This file contains TensorFlow ops whose definitions are amended to fix
// issues or provide more information. In this file you have full control
// of the op definition; all changes will be retained with subsequent
// refreshes.
//
// This file includes another file, `tf_generated_ops.td`, which contains
// all ops whose definitions are generated from TensorFlow codebase.
// Changes made there are not respected.

#ifndef LCE_OPS
#define LCE_OPS

include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

def TF_LqceBsignOp : TF_Op<"LqceBsign", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns an element-wise indication of the binary sign of a number.";

  let description = [{
`y = sign(x) = -1` if `x < 0`; 1 if `x >= 0`.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LqceBconv2d64Op : TF_Op<"LqceBconv2d64", [NoSideEffect]> {
  let summary = [{
Computes a 2-D binary convolution by binarizing and bitpacking the input and filter.
  }];

  let description = [{
TODO
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32]>:$input,
    TensorOf<[BF16, F16, F32, F64, I32]>:$filter,
    TensorOf<[F32]>:$fused_multiply,
    TensorOf<[F32]>:$fused_add,

    I64ArrayAttr:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations,
    DefaultValuedAttr<TF_AnyStrAttrOf<["HWIO", "OHWI", "OHWI_PACKED"]>, "HWIO">:$filter_format
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  /* let verifier = [{
    return Verify(*this);
  }]; */
}

#endif // LCE_OPS
