# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: trial_inputs.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='trial_inputs.proto',
  package='com.wattpad.bayesian_optimizer',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x12trial_inputs.proto\x12\x1e\x63om.wattpad.bayesian_optimizer\".\n\x0bTrialInputs\x12\x0f\n\x07version\x18\x01 \x01(\t\x12\x0e\n\x06inputs\x18\x02 \x03(\x01\x62\x06proto3')
)




_TRIALINPUTS = _descriptor.Descriptor(
  name='TrialInputs',
  full_name='com.wattpad.bayesian_optimizer.TrialInputs',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='com.wattpad.bayesian_optimizer.TrialInputs.version', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='com.wattpad.bayesian_optimizer.TrialInputs.inputs', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=54,
  serialized_end=100,
)

DESCRIPTOR.message_types_by_name['TrialInputs'] = _TRIALINPUTS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrialInputs = _reflection.GeneratedProtocolMessageType('TrialInputs', (_message.Message,), dict(
  DESCRIPTOR = _TRIALINPUTS,
  __module__ = 'trial_inputs_pb2'
  # @@protoc_insertion_point(class_scope:com.wattpad.bayesian_optimizer.TrialInputs)
  ))
_sym_db.RegisterMessage(TrialInputs)


# @@protoc_insertion_point(module_scope)
