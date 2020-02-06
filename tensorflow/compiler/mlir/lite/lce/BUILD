load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load(
    "//third_party/mlir:tblgen.bzl",
    "gentbl",
)

package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = ["//tensorflow/compiler/mlir/..."],
)

gentbl(
    name = "lce_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/lce_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/lce_ops.cc.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/lce_ops.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/lce_ops.td",
    td_srcs = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)


gentbl(
    name = "tensorflow_lite_prepare_lce_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_prepare.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns.td",
    td_srcs = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "transforms/prepare_patterns.td",
    ],
)

gentbl(
    name = "tensorflow_lite_optimize_lce_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns.td",
    td_srcs = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

# FIXME(arash): this needs structure
cc_library(
    name = "lce_passes",
    srcs = [
        "ir/lce_ops.cc.inc",
        "ir/lce_ops.cc",
        "transforms/prepare_tf.cc",
        "transforms/generated_prepare.inc",
        "transforms/optimize.cc",
        "transforms/generated_optimize.inc",
    ],
    hdrs = [
        "transforms/passes.h",
        "transforms/utils.h",
        "ir/lce_ops.h",
        "ir/lce_ops.h.inc",
    ],
    deps = [
        ":lce_ops_inc_gen",
        ":tensorflow_lite_prepare_lce_inc_gen",
        ":tensorflow_lite_optimize_lce_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_inc_gen",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_op_interfaces_inc_gen",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "@flatbuffers",
        "@llvm-project//mlir:QuantOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_tfl_passes",
    srcs = ["tf_tfl_passes.cc"],
    hdrs = [
        "tf_tfl_passes.h",
        "transforms/passes.h",
    ],
    deps = [
        ":lce_passes",
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",
    ],
)

# FIXME(arash): We probably don't need all of this
tf_cc_binary(
    name = "tf_lce_translate",
    srcs = ["tf_lce_translate.cc"],
    deps = [
        ":lce_passes",
        ":tf_tfl_passes",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_tfl_translate_cl_options",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
