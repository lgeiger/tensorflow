"""An object-local variable management scheme."""
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import abc
import collections
import enum
import weakref

import six

from tensorflow.python.checkpoint import checkpoint_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import registration
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export

# Key where the object graph proto is saved in a TensorBundle
OBJECT_GRAPH_PROTO_KEY = "_CHECKPOINTABLE_OBJECT_GRAPH"

# A key indicating a variable's value in an object's checkpointed Tensors
# (Trackable._gather_saveables_for_checkpoint). If this is the only key and
# the object has no dependencies, then its value may be restored on object
# creation (avoiding double assignment when executing eagerly).
VARIABLE_VALUE_KEY = "VARIABLE_VALUE"
OBJECT_CONFIG_JSON_KEY = "OBJECT_CONFIG_JSON"


@enum.unique
class SaveType(str, enum.Enum):
  SAVEDMODEL = "savedmodel"
  CHECKPOINT = "checkpoint"


@tf_export("__internal__.tracking.TrackableReference", v1=[])
class TrackableReference(object):
  """A named reference to a trackable object for use with the `Trackable` class.

  These references mark named `Trackable` dependencies of a `Trackable` object
  and should be created when overriding `Trackable._checkpoint_dependencies`.

  Attributes:
    name: The local name for this dependency.
    ref: The `Trackable` object being referenced.
  """

  __slots__ = ("_name", "_ref")

  def __init__(self, name, ref):
    self._name = name
    self._ref = ref

  @property
  def name(self):
    return self._name

  @property
  def ref(self):
    return self._ref

  def __iter__(self):
    yield self.name
    yield self.ref

  def __repr__(self):
    return f"{self.__class__.__name__}(name={self.name}, ref={self.ref})"

  def __eq__(self, o):
    if isinstance(o, tuple):
      return (self.name, self.ref) == o
    elif isinstance(o, TrackableReference):
      return self.name == o.name and self.ref == o.ref
    else:
      return False


class WeakTrackableReference(TrackableReference):
  """TrackableReference that stores weak references."""
  __slots__ = ()

  def __init__(self, name, ref):
    if not isinstance(self, weakref.ref):
      ref = weakref.ref(ref)
    super(WeakTrackableReference, self).__init__(name=name, ref=ref)

  @property
  def ref(self):
    return self._ref()


# TODO(bfontain):  Update once sharded initialization interface is finalized.
ShardInfo = collections.namedtuple(
    "CheckpointInitialValueShardInfo", ["shape", "offset"])


@tf_export("__internal__.tracking.CheckpointInitialValueCallable", v1=[])
class CheckpointInitialValueCallable(object):
  """A callable object that returns a CheckpointInitialValue.

  See CheckpointInitialValue for more information.
  """

  def __init__(self, checkpoint_position):
    self._checkpoint_position = checkpoint_position

  @property
  def checkpoint_position(self):
    return self._checkpoint_position

  def __call__(self, shape=None, dtype=None, shard_info=None):
    # Note that the signature here is for compatibility with normal callable
    # initializers which take shape and dtype. Although dtype isn't used, it
    # will get passed in by a functool.partial_wrapper in places like
    # base_layer_utils.py's make_variable.
    return CheckpointInitialValue(
        self._checkpoint_position, shape, shard_info=shard_info)

  @property
  def restore_uid(self):
    return self._checkpoint_position.restore_uid


@tf_export("__internal__.tracking.CheckpointInitialValue", v1=[])
class CheckpointInitialValue(ops.Tensor):
  """Tensor wrapper for managing update UIDs in `Variables`.

  When supplied as an initial value, objects of this type let a `Variable`
  (`Variable`, `ResourceVariable`, etc.) know the UID of the restore the initial
  value came from. This allows deferred restorations to be sequenced in the
  order the user specified them, and lets us fall back on assignment if an
  initial value is not set (e.g. due to a custom getter interfering).

  See comments in _add_variable_with_custom_getter for more information about
  how `CheckpointInitialValue` is used.
  """

  def __init__(self, checkpoint_position, shape=None, shard_info=None):
    if shard_info:
      full_shape_str = " ".join("%d" % d for d in shape) + " "
      slice_spec = ":".join(
          "%d,%d" % (o, s) for o, s in zip(shard_info.offset, shard_info.shape))
      shape_and_slice = full_shape_str + slice_spec
    else:
      shape_and_slice = ""
    self.wrapped_value = checkpoint_position.value_tensors(
        {VARIABLE_VALUE_KEY: shape_and_slice})[VARIABLE_VALUE_KEY]
    self._checkpoint_position = checkpoint_position

  def __getattr__(self, attr):
    try:
      return getattr(self.wrapped_value, attr)
    except AttributeError:
      return self.__getattribute__(attr)

  @property
  def checkpoint_position(self):
    return self._checkpoint_position


class NoRestoreSaveable(saveable_object.SaveableObject):
  """Embeds a tensor in a checkpoint with no restore ops."""

  def __init__(self, tensor, name, dtype=None, device=None):
    spec = saveable_object.SaveSpec(
        tensor, "", name, dtype=dtype, device=device)
    super(NoRestoreSaveable, self).__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    return control_flow_ops.no_op()


@six.add_metaclass(abc.ABCMeta)
class PythonStateSaveable(saveable_object.SaveableObject):
  """An interface for saving/restoring volatile Python state."""

  @abc.abstractmethod
  def feed_dict_additions(self):
    """When running a graph, indicates fresh state to feed.

    Returns:
      A dictionary mapping `Tensor`s to current Python state.
    """
    pass

  @abc.abstractmethod
  def freeze(self):
    """Create a new `SaveableObject` which freezes current state as a constant.

    Used when executing eagerly to embed the current state as a constant, or
    when creating a static tf.compat.v1.train.Saver with the frozen current
    Python state.

    Returns:
      A `SaveableObject` which is not a `PythonStateSaveable` instance (i.e. has
      no Python state associated with it).
    """
    pass


class PythonStringStateSaveable(PythonStateSaveable):
  """Saves Python state in a checkpoint."""

  def __init__(self, name, state_callback, restore_callback):
    """Configure saving.

    Args:
      name: The checkpoint key to write to.
      state_callback: A function taking no arguments which returns a string.
        This function is run every time a checkpoint is written.
      restore_callback: A function taking a Python string, used to restore
        state.
    """
    def _state_callback_wrapper():
      with ops.init_scope():
        return state_callback()

    self._state_callback = _state_callback_wrapper
    self._restore_callback = restore_callback
    with ops.device("/cpu:0"):
      self._save_string = constant_op.constant("", dtype=dtypes.string)
    spec = saveable_object.SaveSpec(
        self._save_string, "", name, dtype=dtypes.string)
    super(PythonStringStateSaveable, self).__init__(self._save_string, [spec],
                                                    name)

  def feed_dict_additions(self):
    """When running a graph, indicates fresh state to feed."""
    return {self._save_string: self._state_callback()}

  def freeze(self):
    """Create a frozen `SaveableObject` which saves the current state."""

    def _constant_state():
      return constant_op.constant(self._state_callback(), dtype=dtypes.string)

    return NoRestoreSaveable(
        tensor=_constant_state,
        dtype=dtypes.string,
        name=self.name,
        device="cpu:0")

  def python_restore(self, restored_strings):
    """Called to restore Python state."""
    if self._restore_callback:
      restored, = restored_strings
      self._restore_callback(restored)

  def restore(self, restored_tensors, restored_shapes):
    """Called to restore TensorFlow state (nothing to do)."""
    return control_flow_ops.no_op()


class CheckpointPosition(object):
  """Indicates a position within a `_CheckpointRestoreCoordinator`."""

  __slots__ = ["_checkpoint", "_proto_id"]

  def __init__(self, checkpoint, proto_id):
    """Specify an object within a checkpoint.

    Args:
      checkpoint: A _CheckpointRestoreCoordinator object.
      proto_id: The index of this object in TrackableObjectGraph.nodes.
    """
    self._checkpoint = checkpoint
    self._proto_id = proto_id

  def restore(self, trackable):
    """Restore this value into `trackable`."""
    with ops.init_scope():
      if self.bind_object(trackable):
        # This object's correspondence with a checkpointed object is new, so
        # process deferred restorations for it and its dependencies.
        restore_ops = trackable._restore_from_checkpoint_position(self)  # pylint: disable=protected-access
        if restore_ops:
          self._checkpoint.new_restore_ops(restore_ops)

  def bind_object(self, trackable):
    """Set a checkpoint<->object correspondence.

    Args:
      trackable: The object to record a correspondence for.

    Returns:
      True if this is a new assignment, False if this object has already been
      mapped to a checkpointed `Object` proto.
    Raises:
      AssertionError: If another object is already bound to the `Object` proto.
    """
    checkpoint = self.checkpoint
    checkpoint.all_python_objects.add(trackable)
    current_assignment = checkpoint.object_by_proto_id.get(self._proto_id, None)
    checkpoint.matched_proto_ids.add(self._proto_id)
    if current_assignment is None:
      checkpoint.object_by_proto_id[self._proto_id] = trackable
      return True  # New assignment
    else:
      # The object was already mapped for this checkpoint load, which means
      # we don't need to do anything besides check that the mapping is
      # consistent (if the dependency DAG is not a tree then there are
      # multiple paths to the same object).
      if current_assignment is not trackable:
        logging.warning(
            "Inconsistent references when loading the checkpoint into this "
            "object graph. For example, in the saved checkpoint object, "
            "`model.layer.weight` and `model.layer_copy.weight` reference the "
            "same variable, while in the current object these are two different"
            " variables. The referenced variables are:"
            f"({current_assignment} and {trackable}).")
      return False  # Not a new assignment

  def is_simple_variable(self):
    """Determine whether this value is restorable with a Tensor initializer."""
    attributes = self.object_proto.attributes
    return (len(attributes) == 1 and
            attributes[0].name == VARIABLE_VALUE_KEY and
            not self.object_proto.children)

  def value_tensors(self, shape_and_slices=None):
    """Create value `Tensor`s for this object's attributes.

    Does not require that the Python object has been created. Used for
    restore-on-create when executing eagerly.

    Args:
      shape_and_slices: A dict mapping from object attribute names to a shape
        and slice string that will be passed to a RestoreV2 op. If the dict is
        None or if an object attribute is not in the dict, the full tensor will
        be restored.

    Returns:
      A dictionary mapping from object attribute names to `Tensor`s.
    """
    value_tensors = {}
    for serialized_tensor in self.object_proto.attributes:
      checkpoint_key = serialized_tensor.checkpoint_key
      dtype = self._checkpoint.dtype_map[checkpoint_key]
      base_type = dtype.base_dtype
      io_device = self._checkpoint.options.experimental_io_device or "cpu:0"
      with ops.init_scope():
        with ops.device(io_device):
          # Run the restore itself on the io_device(CPU or specified).
          if (shape_and_slices is not None and
              serialized_tensor.name in shape_and_slices):
            shape_and_slice = shape_and_slices[serialized_tensor.name]
          else:
            shape_and_slice = ""
          value, = io_ops.restore_v2(
              prefix=self._checkpoint.save_path_tensor,
              tensor_names=[checkpoint_key],
              shape_and_slices=[shape_and_slice],
              dtypes=[base_type],
              name="%s_checkpoint_read" % (serialized_tensor.name,))
        # Copy the value to the current device if necessary.
        value_tensors[serialized_tensor.name] = array_ops.identity(value)
    return value_tensors

  def gather_ops_or_named_saveables(self):
    """Looks up or creates SaveableObjects which don't have cached ops."""
    # pylint:disable=g-import-not-at-top
    # There are circular dependencies between Trackable and SaveableObject,
    # so we must import it here.
    # TODO(b/224069573): Remove this code from Trackable.
    from tensorflow.python.training.saving import saveable_object_util
    # pylint:enable=g-import-not-at-top

    saveables = saveable_object_util.saveable_objects_from_trackable(
        self.trackable)
    # Name saveables based on the name this object had when it was checkpointed.
    named_saveables = {}
    python_saveables = []
    existing_restore_ops = []
    for serialized_tensor in self.object_proto.attributes:
      if context.executing_eagerly():
        existing_op = None
      else:
        existing_op = self._checkpoint.restore_ops_by_name.get(
            serialized_tensor.checkpoint_key, None)
      if existing_op is not None:
        existing_restore_ops.append(existing_op)
        continue

      # Only if we don't have cached ops for this SaveableObject, we'll see if
      # the SaveableObject itself has been cached. If not, we'll make it, and
      # either way we'll extract new ops from it (or if it has Python state to
      # restore, we'll run that).
      saveables_cache = self._checkpoint.graph_view.saveables_cache
      if saveables_cache is None:
        # No SaveableObject caching when executing eagerly.
        saveable = None
      else:
        # If we've already created and cached a SaveableObject for this
        # attribute, we can re-use it to avoid re-creating some ops when graph
        # building.
        saveable_list = saveables_cache.get(self.trackable,
                                            {}).get(serialized_tensor.name,
                                                    (None,))
        if len(saveable_list) == 1:
          # Almost every attribute will have exactly one SaveableObject.
          saveable, = saveable_list
        else:
          # Don't use cached SaveableObjects for partitioned variables, which is
          # the only case where we'd have a list of SaveableObjects. Op caching
          # will catch them.
          saveable = None
      if saveable is not None:
        # The name of this attribute has changed, so we need to re-generate
        # the SaveableObject.
        if serialized_tensor.checkpoint_key not in saveable.name:
          saveable = None
          del saveables_cache[self.trackable]
      if saveable is None:
        # If there was no cached SaveableObject, we should check if the Python
        # object has the attribute.
        saveable_factory = saveables.get(serialized_tensor.name, None)
        if saveable_factory is None:
          # Purposefully does not throw an exception if attributes have been
          # added or deleted. Stores unused attributes so an exception can be
          # raised if the user decides to check that everything in the
          # checkpoint was loaded.
          self._checkpoint.unused_attributes.setdefault(
              self._proto_id, []).append(serialized_tensor.name)
          continue
        if callable(saveable_factory):
          saveable = saveable_factory(name=serialized_tensor.checkpoint_key)
        else:
          saveable = saveable_factory
        if saveables_cache is not None:
          saveables_cache.setdefault(self.trackable,
                                     {})[serialized_tensor.name] = [saveable]
      if isinstance(saveable, PythonStateSaveable):
        python_saveables.append(saveable)
      else:
        named_saveables[serialized_tensor.checkpoint_key] = saveable
    return existing_restore_ops, named_saveables, python_saveables

  def restore_ops(self):
    """Create or fetch restore ops for this object's attributes.

    Requires that the `Trackable` Python object has been bound to an object
    ID in the checkpoint.

    Returns:
      A list of operations when graph building, or an empty list when executing
      eagerly.
    """
    if self._has_registered_saver():
      raise ValueError("Unable to run individual checkpoint restore for objects"
                       " with registered savers.")
    (restore_ops, tensor_saveables,
     python_saveables) = self.gather_ops_or_named_saveables()
    restore_ops.extend(
        self._checkpoint.restore_saveables(tensor_saveables, python_saveables))
    return restore_ops

  @property
  def checkpoint(self):
    return self._checkpoint

  @property
  def trackable(self):
    return self._checkpoint.object_by_proto_id[self._proto_id]

  @property
  def object_proto(self):
    return self._checkpoint.object_graph_proto.nodes[self._proto_id]

  @property
  def proto_id(self):
    return self._proto_id

  @property
  def restore_uid(self):
    return self._checkpoint.restore_uid

  def __repr__(self):
    return repr(self.object_proto)

  def value_shape(self):
    """The shape of the VARIABLE_VALUE tensor.

    Returns:
      If found a TensorShape object, otherwise None.
    """
    for serialized_tensor in self.object_proto.attributes:
      if serialized_tensor.name == VARIABLE_VALUE_KEY:
        return self._checkpoint.shape_map[serialized_tensor.checkpoint_key]
    return None

  def _has_registered_saver(self):
    return bool(self.object_proto.registered_saver.name)

  def get_registered_saver_name(self):
    if self._has_registered_saver():
      saver_name = self.object_proto.registered_saver.name
      registration.validate_restore_function(self.trackable, saver_name)
      return saver_name
    return None

  # TODO(kathywu): remove this method from CheckpointPosition once the class
  # has been copied into `checkpoint_util.py`.
  def create_slot_variable_position(self, optimizer_object, variable,
                                    slot_variable_id, slot_name):
    """Generates CheckpointPosition for a slot variable.

    Args:
      optimizer_object: Optimizer that owns the slot variable.
      variable: Variable associated with the slot variable.
      slot_variable_id: ID of the slot variable.
      slot_name: Name of the slot variable.

    Returns:
      If there is a slot variable in the `optimizer_object` that has not been
      bound to the checkpoint, this function returns a tuple of (
        new `CheckpointPosition` for the slot variable,
        the slot variable itself).
    """
    slot_variable_position = CheckpointPosition(
        checkpoint=self.checkpoint, proto_id=slot_variable_id)
    # pylint: disable=protected-access
    slot_variable = optimizer_object._create_or_restore_slot_variable(
        slot_variable_position=slot_variable_position,
        variable=variable,
        slot_name=slot_name)
    # pylint: enable=protected-access
    if (slot_variable is not None and
        slot_variable_position.bind_object(slot_variable)):
      return slot_variable_position, slot_variable
    else:
      return None, None


_SlotVariableRestoration = collections.namedtuple(
    "_SlotVariableRestoration",
    [
        # The checkpoint proto id of the optimizer object.
        "optimizer_id",
        # The checkpoint proto id of the slot variable.
        "slot_variable_id",
        "slot_name",
    ])


@tf_export("__internal__.tracking.no_automatic_dependency_tracking", v1=[])
def no_automatic_dependency_tracking(method):
  """Disables automatic dependency tracking on attribute assignment.

  Use to decorate any method of a Trackable object. Attribute assignment in
  that method will not add dependencies (also respected in Model). Harmless if
  used in a class which does not do automatic dependency tracking (which means
  it's safe to use in base classes which may have subclasses which also inherit
  from Trackable).

  Args:
    method: The method to decorate.

  Returns:
    A decorated method which sets and un-sets automatic dependency tracking for
    the object the method is called on (not thread safe).
  """

  def _method_wrapper(self, *args, **kwargs):
    previous_value = getattr(self, "_self_setattr_tracking", True)
    self._self_setattr_tracking = False  # pylint: disable=protected-access
    try:
      result = method(self, *args, **kwargs)
    finally:
      self._self_setattr_tracking = previous_value  # pylint: disable=protected-access
    return result

  return tf_decorator.make_decorator(
      target=method, decorator_func=_method_wrapper)


@tf_contextlib.contextmanager
def no_manual_dependency_tracking_scope(obj):
  """A context that disables manual dependency tracking for the given `obj`.

  Sometimes library methods might track objects on their own and we might want
  to disable that and do the tracking on our own. One can then use this context
  manager to disable the tracking the library method does and do your own
  tracking.

  For example:

  class TestLayer(tf.keras.Layer):
    def build():
      with no_manual_dependency_tracking_scope(self):
        var = self.add_variable("name1")  # Creates a var and doesn't track it
      self._track_trackable("name2", var)  # We track variable with name `name2`

  Args:
    obj: A trackable object.

  Yields:
    a scope in which the object doesn't track dependencies manually.
  """
  # pylint: disable=protected-access
  previous_value = getattr(obj, "_manual_tracking", True)
  obj._manual_tracking = False
  try:
    yield
  finally:
    obj._manual_tracking = previous_value


@tf_contextlib.contextmanager
def no_automatic_dependency_tracking_scope(obj):
  """A context that disables automatic dependency tracking when assigning attrs.

  Objects that inherit from Autotrackable automatically creates dependencies
  to trackable objects through attribute assignments, and wraps data structures
  (lists or dicts) with trackable classes. This scope may be used to temporarily
  disable this behavior. This works similar to the decorator
  `no_automatic_dependency_tracking`.

  Example usage:
  ```
  model = tf.keras.Model()
  model.arr1 = []  # Creates a ListWrapper object
  with no_automatic_dependency_tracking_scope(model):
    model.arr2 = []  # Creates a regular, untracked python list
  ```

  Args:
    obj: A trackable object.

  Yields:
    a scope in which the object doesn't track dependencies.
  """
  previous_value = getattr(obj, "_setattr_tracking", True)
  obj._setattr_tracking = False  # pylint: disable=protected-access
  try:
    yield
  finally:
    obj._setattr_tracking = previous_value  # pylint: disable=protected-access


@tf_export("__internal__.tracking.Trackable", v1=[])
class Trackable(object):
  """Base class for `Trackable` objects without automatic dependencies.

  This class has no __setattr__ override for performance reasons. Dependencies
  must be added explicitly. Unless attribute assignment is performance-critical,
  use `AutoTrackable` instead. Use `Trackable` for `isinstance`
  checks.
  """

  # For compatibility with wrapt.ObjectProxy, attributes are all prefixed with
  # _self_. We have some properties to forward semi-public attributes to their
  # _self_ equivalents.

  @property
  def _setattr_tracking(self):
    if not hasattr(self, "_self_setattr_tracking"):
      self._self_setattr_tracking = True
    return self._self_setattr_tracking

  @_setattr_tracking.setter
  def _setattr_tracking(self, value):
    self._self_setattr_tracking = value

  @property
  def _update_uid(self):
    return self._self_update_uid

  @_update_uid.setter
  def _update_uid(self, value):
    self._self_update_uid = value

  @property
  def _unconditional_checkpoint_dependencies(self):
    return self._self_unconditional_checkpoint_dependencies

  @property
  def _unconditional_dependency_names(self):
    return self._self_unconditional_dependency_names

  @property
  def _name_based_restores(self):
    return self._self_name_based_restores

  # Trackable does not do automatic dependency tracking, but uses the
  # no_automatic_dependency_tracking decorator so it can avoid adding
  # dependencies if a subclass is Trackable / inherits from Model (both of
  # which have __setattr__ overrides).
  @no_automatic_dependency_tracking
  def _maybe_initialize_trackable(self):
    """Initialize dependency management.

    Not __init__, since most objects will forget to call it.
    """
    if hasattr(self, "_self_unconditional_checkpoint_dependencies"):
      # __init__ already called. This check means that we don't need
      # Trackable.__init__() in the constructor of every TensorFlow object.
      return
    # A list of TrackableReference objects. Some classes implementing
    # `Trackable`, notably `Optimizer`s, may override the
    # _checkpoint_dependencies property with conditional dependencies
    # (e.g. based on the current graph when saving).
    self._self_unconditional_checkpoint_dependencies = []
    # Maps names -> Trackable objects
    self._self_unconditional_dependency_names = {}
    # Restorations for other Trackable objects on which this object may
    # eventually depend. Maps local name -> CheckpointPosition list. Optimizers
    # tack on conditional dependencies, and so need separate management of
    # deferred dependencies too.
    self._self_unconditional_deferred_dependencies = {}
    # The UID of the highest assignment to this object. Used to ensure that the
    # last requested assignment determines the final value of an object.
    if hasattr(self, "_self_update_uid"):
      raise AssertionError(
          "Internal error: the object had an update UID set before its "
          "initialization code was run.")
    self._self_update_uid = -1
    # When executing eagerly, holds a collection of _NameBasedRestoreCoordinator
    # instances, which should be checked when creating variables or other
    # saveables. These are passed on recursively to all dependencies, since
    # unlike object-based checkpoint restores we don't know which subgraph is
    # being restored in advance. This mechanism is only necessary for
    # restore-on-create when executing eagerly, and so is unused when graph
    # building.
    self._self_name_based_restores = set()

    # Dictionary of SaveableObjects factories. This dictionary is defined when
    # the object is loaded from the SavedModel. When writing a custom class,
    # prefer overriding "_gather_saveables_from_checkpoint" to using this
    # attribute.
    self._self_saveable_object_factories = {}

  @property
  def _object_identifier(self):
    """String used to identify this object in a SavedModel.

    THIS FIELD HAS BEEN DEPRECATED IN FAVOR OF THE NAME REGISTERED WITH
    `register_serializable`.

    Generally, the object identifier is constant across objects of the same
    class, while the metadata field is used for instance-specific data.

    Returns:
      String object identifier.
    """
    return "_generic_user_object"

  def _no_dependency(self, value):
    """If automatic dependency tracking is enabled, ignores `value`."""
    return value

  def _name_based_attribute_restore(self, checkpoint):
    """Restore the object's attributes from a name-based checkpoint."""
    self._self_name_based_restores.add(checkpoint)
    if self._self_update_uid < checkpoint.restore_uid:
      checkpoint.eager_restore(self)
      self._self_update_uid = checkpoint.restore_uid

  @property
  def _checkpoint_dependencies(self):
    """All dependencies of this object.

    May be overridden to include conditional dependencies.

    Returns:
      A list of `TrackableReference` objects indicating named
      `Trackable` dependencies which should be saved along with this
      object.
    """
    return self._self_unconditional_checkpoint_dependencies

  @property
  def _deferred_dependencies(self):
    """A dictionary with deferred dependencies.

    Stores restorations for other Trackable objects on which this object
    may eventually depend. May be overridden by sub-classes (e.g. Optimizers use
    conditional dependencies based the current graph, and so need separate
    management of deferred dependencies too).

    Returns:
      A dictionary mapping from local name to a list of CheckpointPosition
      objects.
    """
    return self._self_unconditional_deferred_dependencies

  def _lookup_dependency(self, name):
    """Look up a dependency by name.

    May be overridden to include conditional dependencies.

    Args:
      name: The local name of the dependency.

    Returns:
      A `Trackable` object, or `None` if no dependency by this name was
      found.
    """
    return self._self_unconditional_dependency_names.get(name, None)

  def _add_variable_with_custom_getter(self,
                                       name,
                                       shape=None,
                                       dtype=dtypes.float32,
                                       initializer=None,
                                       getter=None,
                                       overwrite=False,
                                       **kwargs_for_getter):
    """Restore-on-create for a variable be saved with this `Trackable`.

    If the user has requested that this object or another `Trackable` which
    depends on this object be restored from a checkpoint (deferred loading
    before variable object creation), `initializer` may be ignored and the value
    from the checkpoint used instead.

    Args:
      name: A name for the variable. Must be unique within this object.
      shape: The shape of the variable.
      dtype: The data type of the variable.
      initializer: The initializer to use. Ignored if there is a deferred
        restoration left over from a call to
        `_restore_from_checkpoint_position`.
      getter: The getter to wrap which actually fetches the variable.
      overwrite: If True, disables unique name and type checks.
      **kwargs_for_getter: Passed to the getter.

    Returns:
      The new variable object.

    Raises:
      ValueError: If the variable name is not unique.
    """
    self._maybe_initialize_trackable()
    with ops.init_scope():
      if context.executing_eagerly():
        # If this is a variable with a single Tensor stored in the checkpoint,
        # we can set that value as an initializer rather than initializing and
        # then assigning (when executing eagerly). This call returns None if
        # there is nothing to restore.
        checkpoint_initializer = self._preload_simple_restoration(
            name=name)
      else:
        checkpoint_initializer = None
      if (checkpoint_initializer is not None and
          not (isinstance(initializer, CheckpointInitialValueCallable) and
               (initializer.restore_uid > checkpoint_initializer.restore_uid))):
        # If multiple Trackable objects are "creating" the same variable
        # via the magic of custom getters, the one with the highest restore UID
        # (the one called last) has to make the final initializer. If another
        # custom getter interrupts this process by overwriting the initializer,
        # then we'll catch that when we call _track_trackable. So this is
        # "best effort" to set the initializer with the highest restore UID.
        initializer = checkpoint_initializer
    new_variable = getter(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        **kwargs_for_getter)

    # If we set an initializer and the variable processed it, tracking will not
    # assign again. It will add this variable to our dependencies, and if there
    # is a non-trivial restoration queued, it will handle that. This also
    # handles slot variables.
    if not overwrite or isinstance(new_variable, Trackable):
      return self._track_trackable(new_variable, name=name, overwrite=overwrite)
    else:
      # TODO(allenl): Some variable types are not yet supported. Remove this
      # fallback once all get_variable() return types are Trackable.
      return new_variable

  def _preload_simple_restoration(self, name):
    """Return a dependency's value for restore-on-create.

    Note the restoration is not deleted; if for some reason preload is called
    and then not assigned to the variable (for example because a custom getter
    overrides the initializer), the assignment will still happen once the
    variable is tracked (determined based on checkpoint.restore_uid).

    Args:
      name: The object-local name of the dependency holding the variable's
        value.

    Returns:
      An callable for use as a variable's initializer/initial_value, or None if
      one should not be set (either because there was no variable with this name
      in the checkpoint or because it needs more complex deserialization). Any
      non-trivial deserialization will happen when the variable object is
      tracked.
    """
    deferred_dependencies_list = self._deferred_dependencies.get(name, ())
    if not deferred_dependencies_list:
      # Nothing to do; we don't have a restore for this dependency queued up.
      return
    for checkpoint_position in deferred_dependencies_list:
      if not checkpoint_position.is_simple_variable():
        # If _any_ pending restoration is too complicated to fit in an
        # initializer (because it has dependencies, or because there are
        # multiple Tensors to restore), bail and let the general tracking code
        # handle it.
        return None
    checkpoint_position = max(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid)
    return CheckpointInitialValueCallable(
        checkpoint_position=checkpoint_position)

  def _track_trackable(self, trackable, name, overwrite=False):
    """Declare a dependency on another `Trackable` object.

    Indicates that checkpoints for this object should include variables from
    `trackable`.

    Variables in a checkpoint are mapped to `Trackable`s based on the names
    provided when the checkpoint was written. To avoid breaking existing
    checkpoints when modifying a class, neither variable names nor dependency
    names (the names passed to `_track_trackable`) may change.

    Args:
      trackable: A `Trackable` which this object depends on.
      name: A local name for `trackable`, used for loading checkpoints into the
        correct objects.
      overwrite: Boolean, whether silently replacing dependencies is OK. Used
        for __setattr__, where throwing an error on attribute reassignment would
        be inappropriate.

    Returns:
      `trackable`, for convenience when declaring a dependency and
      assigning to a member variable in one statement.

    Raises:
      TypeError: If `trackable` does not inherit from `Trackable`.
      ValueError: If another object is already tracked by this name.
    """
    self._maybe_initialize_trackable()
    if not isinstance(trackable, Trackable):
      raise TypeError(
          "Trackable._track_trackable() can only be used to track objects of "
          f"type Trackable. Got type {type(trackable)}.")
    if not getattr(self, "_manual_tracking", True):
      return trackable
    new_reference = TrackableReference(name=name, ref=trackable)
    current_object = self._lookup_dependency(name)
    if (current_object is not None and current_object is not trackable):
      if not overwrite:
        raise ValueError(
            f"Called Trackable._track_trackable() with name='{name}', "
            "but a Trackable with this name is already declared as a "
            "dependency. Names must be unique (or overwrite=True).")
      # This is a weird thing to do, but we're not going to stop people from
      # using __setattr__.
      for index, (old_name, _) in enumerate(
          self._self_unconditional_checkpoint_dependencies):
        if name == old_name:
          self._self_unconditional_checkpoint_dependencies[
              index] = new_reference
    elif current_object is None:
      self._self_unconditional_checkpoint_dependencies.append(new_reference)
      self._handle_deferred_dependencies(name=name, trackable=trackable)
    self._self_unconditional_dependency_names[name] = trackable
    return trackable

  def _handle_deferred_dependencies(self, name, trackable):
    """Pop and load any deferred checkpoint restores into `trackable`.

    This method does not add a new dependency on `trackable`, but it does
    check if any outstanding/deferred dependencies have been queued waiting for
    this dependency to be added (matched based on `name`). If so,
    `trackable` and its dependencies are restored. The restorations are
    considered fulfilled and so are deleted.

    `_track_trackable` is more appropriate for adding a
    normal/unconditional dependency, and includes handling for deferred
    restorations. This method allows objects such as `Optimizer` to use the same
    restoration logic while managing conditional dependencies themselves, by
    overriding `_checkpoint_dependencies` and `_lookup_dependency` to change the
    object's dependencies based on the context it is saved/restored in (a single
    optimizer instance can have state associated with multiple graphs).

    Args:
      name: The name of the dependency within this object (`self`), used to
        match `trackable` with values saved in a checkpoint.
      trackable: The Trackable object to restore (inheriting from `Trackable`).
    """
    self._maybe_initialize_trackable()
    trackable._maybe_initialize_trackable()  # pylint: disable=protected-access
    deferred_dependencies_list = self._deferred_dependencies.pop(name, ())
    for checkpoint_position in sorted(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid,
        reverse=True):
      checkpoint_position.restore(trackable)

    # Pass on any name-based restores queued in this object.
    for name_based_restore in sorted(
        self._self_name_based_restores,
        key=lambda checkpoint: checkpoint.restore_uid,
        reverse=True):
      trackable._name_based_attribute_restore(name_based_restore)  # pylint: disable=protected-access

  def _restore_from_checkpoint_position(self, checkpoint_position):
    """Restore this object and its dependencies (may be deferred)."""
    # Attempt a breadth-first traversal, since presumably the user has more
    # control over shorter paths. If we don't have all of the dependencies at
    # this point, the end result is not breadth-first (since other deferred
    # traversals will happen later).

    # You may be wondering why elements in the `visit_queue` are tuples that
    # contains both CheckpointPositions and their Trackable. The reason is that
    # Optimizers will not keep a strong reference to slot vars for
    # ShardedVariables. The slot variable must be kept in memory until the
    # restore saveables have been created.
    visit_queue = collections.deque([(checkpoint_position,
                                      checkpoint_position.trackable)])
    restore_ops = []
    tensor_saveables = {}
    python_saveables = []
    registered_savers = collections.defaultdict(dict)
    while visit_queue:
      current_position, trackable = visit_queue.popleft()

      # Restore using the ops defined in a Saveable or registered function.
      registered_saver = current_position.get_registered_saver_name()
      if registered_saver:
        object_name = (
            current_position.object_proto.registered_saver.object_name)
        registered_savers[registered_saver][object_name] = trackable
        trackable._self_update_uid = current_position.checkpoint.restore_uid  # pylint: disable=protected-access
      else:
        new_restore_ops, new_tensor_saveables, new_python_saveables = (
            trackable._single_restoration_from_checkpoint_position(  # pylint: disable=protected-access
                current_position))
        restore_ops.extend(new_restore_ops)
        tensor_saveables.update(new_tensor_saveables)
        python_saveables.extend(new_python_saveables)

      _queue_children_for_restoration(current_position, visit_queue)
      checkpoint_util.queue_slot_variables(current_position, visit_queue)

    restore_ops.extend(
        current_position.checkpoint.restore_saveables(tensor_saveables,
                                                      python_saveables,
                                                      registered_savers))
    return restore_ops

  def _single_restoration_from_checkpoint_position(self, checkpoint_position):
    """Restore this object, and either queue its dependencies or defer them."""
    self._maybe_initialize_trackable()
    checkpoint = checkpoint_position.checkpoint
    # If the UID of this restore is lower than our current update UID, we don't
    # need to actually restore the object. However, we should pass the
    # restoration on to our dependencies.
    if checkpoint.restore_uid > self._self_update_uid:
      restore_ops, tensor_saveables, python_saveables = (
          checkpoint_position.gather_ops_or_named_saveables())
      self._self_update_uid = checkpoint.restore_uid
    else:
      restore_ops = ()
      tensor_saveables = {}
      python_saveables = ()
    return restore_ops, tensor_saveables, python_saveables

  def _gather_saveables_for_checkpoint(self):
    """Returns a dictionary of values to checkpoint with this object.

    NOTE: This method is deprecated, please use `_serialize_to_tensors` and
    `_restore_from_tensors` instead.

    Keys in the returned dictionary are local to this object and in a separate
    namespace from dependencies. Values may either be `SaveableObject` factories
    or variables easily converted to `SaveableObject`s (as in
    `tf.compat.v1.train.Saver`'s
    `var_list` constructor argument).

    `SaveableObjects` have a name set, which Trackable needs to generate
    itself. So rather than returning `SaveableObjects` directly, this method
    should return a dictionary of callables which take `name` arguments and
    return `SaveableObjects` with that name.

    If this object may also be passed to the global-name-based
    `tf.compat.v1.train.Saver`,
    the returned callables should have a default value for their name argument
    (i.e. be callable with no arguments).

    Returned values must be saved only by this object; if any value may be
    shared, it should instead be a dependency. For example, variable objects
    save their own values with the key `VARIABLE_VALUE_KEY`, but objects which
    reference variables simply add a dependency.

    Returns:
      The dictionary mapping attribute names to `SaveableObject` factories
      described above. For example:
      {VARIABLE_VALUE_KEY:
       lambda name="global_name_for_this_object":
       SaveableObject(name=name, ...)}
    """
    # TODO(kathywu): In order to remove this circular dependency, remove all
    # external calls to _gather_saveables_for_checkpoint.
    # pylint: disable=g-import-not-at-top
    from tensorflow.python.training.saving import saveable_object_util
    # pylint: enable=g-import-not-at-top
    if saveable_object_util.trackable_has_serialize_to_tensor(self):
      def create_saveable(name=""):
        return saveable_object_util.TrackableSaveable(self, name)
      return {"": create_saveable}
    else:
      return getattr(self, "_self_saveable_object_factories", {})

  def _serialize_to_tensors(self):
    """Gathers tensors to save to the checkpoint.

    Returns:
      A dictionary mapping names to tensors.
    """
    raise NotImplementedError

  def _restore_from_tensors(self, restored_tensors):
    """Restores checkpointed values to this `Trackable`.

    Args:
      restored_tensors: A dictionary mapping names to tensors. The keys to this
        dictionary matches the names passed to _serialize_to_tensors.

    Returns:
      An op that runs the restoration.
    """
    raise NotImplementedError

  def _map_resources(self, save_options):  # pylint: disable=unused-argument
    """Makes new resource handle ops corresponding to existing resource tensors.

    Internal sub-classes can override this to inform model saving how to add new
    resource handle ops to the main GraphDef of a SavedModel (TF 1.x style
    graph), which allows session based APIs (e.g, C++ loader API) to interact
    with resources owned by this object.

    Args:
      save_options: A tf.saved_model.SaveOptions instance.

    Returns:
      A tuple of (object_map, resource_map):
        object_map: A dictionary mapping from objects that hold existing
          resource tensors to replacement objects created to hold the new
          resource tensors.
        resource_map: A dictionary mapping from existing resource tensors to
          newly created resource tensors.
    """
    return {}, {}

  def _serialize_to_proto(self, object_proto=None, **kwargs):
    """Returns a proto of any type to be saved into the SavedModel.

    Trackable classes decorated with `register_serializable` should overwrite
    this method to save metadata for this object to the SavedModel. The proto
    returned by this function will be passed to `_deserialize_from_proto` in the
    form of a `google.protobuf.Any` proto.

    This data is only saved and used by the Python API. Existing C++ loading
    APIs such as `tensorflow::LoadSavedModel` will not read this field at all.

    Args:
      object_proto: A `SavedObject` proto that may be filled by this function.
        Only the core serializable types (Variable, Function, Constant, Asset)
        should modify this argument.
      **kwargs: Future keyword arguments passed to the object during saving.

    Returns:
      A proto that serializes this class's type.
    """
    del object_proto, kwargs  # Unused.

    return None

  @classmethod
  def _deserialize_from_proto(
      cls, proto=None, dependencies=None, object_proto=None, export_dir=None,
      asset_file_def=None, operation_attributes=None, **kwargs):
    """Returns a new object restored by the SavedModel.

    Trackable classes decorated with `register_serializable` should overwrite
    this method to change how the object is loaded from SavedModel. By default,
    the object is initialized with no arguments.

    Example:

    ```
    def _serialize_to_proto(self, **unused_kwargs):
      return Message(name="a")

    @classmethod
    def _deserialize_from_proto(cls, proto, **unused_kwargs):
      if proto.Is(Message.DESCRIPTOR):
        unpacked = Message()
        proto.Unpack(unpacked)
        return cls(unpacked.name)
      else:
        return cls()
    ```

    This function is only used by the Python API. C++ and TensorFlow Serving do
    not have access to your registered class and cannot execute any of the
    non-tf.functions attached to the Python class. However, all signatures and
    tf.functions are still accessible.

    **Avoid creating duplicate trackables**

    SavedModel is saved by recursively gathering all of the trackables and their
    children. SavedModel loading reverses those steps by creating all
    trackables, then reconnecting the children trackables to their parents using
    `Trackable._add_trackable_child`.

    That means that if `_deserialize_from_proto` calls the `__init__` function,
    which creates all of the children trackables, then those children end up
    being created *twice*.

    To avoid this, structure your code so that Trackables are not created
    when deserialized from SavedModel:

    ```
    @register_serializable()
    class Serializable(trackable):
      def __init __(self, from_proto=False):
        create_non_trackable_objects()
        if not from_proto:
          create_variables_and_other_trackables()

      def _deserialize_from_proto(cls, **kwargs):
        return cls(from_proto=True)

      def _add_trackable_child(self, name, value):
        self.__setattr__(name, value)
    ```

    Args:
      proto: A `google.protobuf.Any` proto read from the `SavedModel`.
      dependencies: A dictionary mapping names to dependencies (see
        `_deserialization_dependencies`)
      object_proto: The `SavedObject` proto for this object.
      export_dir: The `SavedModel` directory
      asset_file_def: The `MetaGraphDef`'s `asset_file_def` field.
      operation_attributes: Dictionary mapping nodes to attribute from the
        imported `GraphDef`.
      **kwargs: Future keyword arguments passed to the object when loading.

    Returns:
      A new object.
    """
    del (proto, dependencies, object_proto, export_dir, asset_file_def,
         operation_attributes, kwargs)

    return cls()

  def _add_trackable_child(self, name, value):
    """Restores a connection between trackables when loading from SavedModel.

    SavedModel stores both the object metadata and its list of children. When
    loading, this function is used along with `_deserialize_from_proto` to load
    objects from the SavedModel: First, all saved objects are created with
    `_deserialize_from_proto`. After that is complete, the children are
    connected using `_add_trackable_child`.

    **Example**

    `tf.Module`, `tf.keras.Model` and Keras layers use `__setattr__` to track
    children. This is why users can call `model.v = tf.Variable(...)`, and the
    variable will be automatically saved to the checkpoint. The implementation
    of this method for the listed objects is:

    ```
    def _add_trackable_child(self, name, value):
      self.__setattr__(name, value)
    ```

    Args:
      name: The name of the connection between the parent and child `Trackable`.
      value: The child `Trackable` object.
    """
    self._track_trackable(value, name, overwrite=True)

  def _deserialization_dependencies(self, children):
    """Returns a dictionary containing `Trackables` that this object depends on.

    Dependencies define the order to serialize and deserialize objects in the
    SavedModel. For example:

    class A(Trackable):
      b = B()
      def _deserialization_dependencies(self, children):
        return {'b': self.b}

    class B(Trackable):
      pass

    We say that object `a=A()` depends on `a.b`.

    Dependencies are guaranteed to be serialized and deserialized before the
    object depending on them. The following methods use dependencies:
      - `_deserialize_from_proto` [loading]

    SavedModel loads with the bottom-up approach, by first creating all objects
    in the order defined by the dependencies, then connecting the children.

    Unlike `_trackable_children`, this function does not define the
    `SavedObjectGraph`. It only changes the order in which things are
    saved/loaded. Therefore, if there are dependencies that are not in the
    `SavedObjectGraph`, saving will fail.

    Args:
      children: Dict returned from `_trackable_children`.

    Returns:
      A dictionary mapping names to `Trackable`.
    """
    del children  # Unused.
    return {}

  def _trackable_children(self,
                          save_type=SaveType.CHECKPOINT,
                          cache=None,
                          **kwargs):
    """Returns this object's `Trackable` attributes.

    This method is used to build the object graph (or the object hierarchy,
    in pickling terms) for checkpoint save/restore, and `SavedModel` export.

    Override this method to define the children of this instance. Please read
    the implementation restrictions:

    **Rule 1: All children must be convertable to `Trackable`.**

    Must pass `isinstance` check or `tracking.convert_to_trackable`.

    **Rule 2: [Checkpoint-only] Do not create new objects.**

    When saving to a `SavedModel`, this method is called *exactly once* for each
    `Trackable` in the object graph. When saving or restoring from a checkpoint,
    this method may be called *multiple times*. Thus, this method may create
    new Trackables when `save_type == SaveType.SAVEDMODEL` but not when
    `save_type == SaveType.CHECKPOINT`.

    When saving to `SavedModel`, new `Trackable` children can be created to save
    non-Trackable attributes to the `SavedModel`. In the example below, `hyper`
    is a regular python float hyperparameter. To save this value, a new Variable
    is created to store the value of `hyper`:

    ```
    def __init__(self):
      self.hyper = 1e-5

    def _trackable_children(self, save_type, **unused_kwargs):
      # Correct implementation
      children = {}
      if format == 'saved_model':
        children['hyper'] = tf.Variable(self.hyper)
      return children
    ```

    An incorrect implementation of `_trackable_children` is shown below. This
    function would cause failures when loading the checkpoint, and calling
    `load_status.assert_consumed()` or
    `load_status.assert_existing_objects_matched`. If you want a value to be
    saved in the checkpoint, hyper must be defined as a `tf.Variable` from the
    start.

    ```
    def _trackable_children(self, save_type, **unused_kwargs):
      # Incorrect implementation
      return {'hyper': tf.Variable(self.hyper)}
    ```

    **Rule 3: [`SavedModel`-only] Watch out for un-traced tf.functions.**

    At the begining of `_trackable_children`, always call
    `get_concrete_function()` for any `tf.function` that has an input signature.

    When `tf.functions` are saved to `SavedModel`, any `tf.functions` that have
    an input signature and has never been called is traced at export time in
    order to copy the op graph into the `SavedModel`. `tf.functions` that are
    traced for the first time are allowed to create new state:


    ```
    @tf.function(input_signature=[]):
    def fn(self);
      if self.v is None:
        self.v = tf.Variable(1.)
      return self.v
    ```

    A problem occurs when there is a `Trackable` that returns `fn` as one of its
    children and `self.v` has not been created yet. When `fn` is traced,
    `self.v` is added to the `Trackable`, but `SavedModel` does not see this
    modification since the `Trackable`'s children have already been gathered.

    Therefore, as a precaution, call `get_concrete_function()` at the very
    start of `_trackable_children` to ensure that the function is traced:


    ```
    def _trackable_children(self):
      self.fn.get_concrete_function()
      return {"v": self.v, "fn": self.fn}
    ```

    Args:
      save_type: A string, can be 'savedmodel' or 'checkpoint'. Defaults to
        SaveType.CHECKPOINT.
      cache: May be `None`, or a dictionary. When `save_type == savedmodel`,
        a new cache is created at the start of the SavedModel export, and shared
        between all `Trackables` in the same object graph. This cache may be
        used for advanced saving functionality.
      **kwargs: Additional kwargs that may be added at a later time.

    Returns:
      Dictionary mapping names to child trackables.
    """
    del save_type, cache, kwargs  # Unused.

    self._maybe_initialize_trackable()
    return {name: ref for name, ref in self._checkpoint_dependencies}

  def _export_to_saved_model_graph(self,
                                   object_map=None,
                                   tensor_map=None,
                                   options=None,
                                   **kwargs):
    """Creates a copy of this object's tensors onto SavedModel graph.

    Needs to be overridden if the class contains tensors that must be saved
    into the graph. This method should update the `object_map` and `tensor_map`
    dictionaries.

    This method is called on all nodes in the Trackable Graph (generated by
    `_trackable_children`). The nodes are traversed in the order defined by
    `_deserialization_dependencies`

    All usages of _map_resources should be migrated to this method.

    Args:
      object_map: A dictionary that maps original Trackables to the copied
        Trackables. This only needs to be updated if the object is a
        tf.function, or if the copied tensors are necessary for checkpointing
        this object.
      tensor_map: Dictionary mapping original tensors to copied tensors.
      options: A `tf.saved_model.SaveOptions` object.
      **kwargs: Additional kwargs that may be added at a later time.

    Returns:
      Flat list of original tensors that have been copied.
    """
    del kwargs  # Unused.
    self_object_map, self_tensor_map = self._map_resources(options)
    object_map.update(self_object_map)
    tensor_map.update(self_tensor_map)
    return list(self_tensor_map.keys())


def _queue_children_for_restoration(checkpoint_position, visit_queue):
  """Queues the restoration of trackable's children or defers them."""
  # pylint: disable=protected-access
  trackable = checkpoint_position.trackable
  checkpoint = checkpoint_position.checkpoint
  for child in checkpoint_position.object_proto.children:
    child_position = CheckpointPosition(
        checkpoint=checkpoint, proto_id=child.node_id)
    local_object = trackable._lookup_dependency(child.local_name)
    child_proto = child_position.object_proto
    if local_object is None:
      # We don't yet have a dependency registered with this name. Save it
      # in case we do.
      if child_proto.HasField("has_checkpoint_values"):
        has_value = child_proto.has_checkpoint_values.value
      else:
        # If the field is not set, do a simple check to see if the dependency
        # has children and/or checkpointed values.
        has_value = bool(child_proto.children or
                         child_proto.attributes or
                         child_proto.slot_variables or
                         child_proto.HasField("registered_saver"))
      if has_value:
        trackable._deferred_dependencies.setdefault(child.local_name,
                                                    []).append(child_position)
    else:
      if child_position.bind_object(trackable=local_object):
        # This object's correspondence is new, so dependencies need to be
        # visited. Delay doing it so that we get a breadth-first dependency
        # resolution order (shallowest paths first). The caller is responsible
        # for emptying visit_queue.
        visit_queue.append((child_position, local_object))
