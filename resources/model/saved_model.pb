æ
»
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8»Ð
~
conv1d_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z* 
shared_nameconv1d_0/kernel
w
#conv1d_0/kernel/Read/ReadVariableOpReadVariableOpconv1d_0/kernel*"
_output_shapes
:Z*
dtype0
r
conv1d_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameconv1d_0/bias
k
!conv1d_0/bias/Read/ReadVariableOpReadVariableOpconv1d_0/bias*
_output_shapes
:Z*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:Z*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:È*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:È*
dtype0

conv1d_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*"
shared_nameconv1d_0/kernel/m
{
%conv1d_0/kernel/m/Read/ReadVariableOpReadVariableOpconv1d_0/kernel/m*"
_output_shapes
:Z*
dtype0
v
conv1d_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z* 
shared_nameconv1d_0/bias/m
o
#conv1d_0/bias/m/Read/ReadVariableOpReadVariableOpconv1d_0/bias/m*
_output_shapes
:Z*
dtype0

conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*"
shared_nameconv1d_1/kernel/m
{
%conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpconv1d_1/kernel/m*"
_output_shapes
:Z*
dtype0
v
conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/bias/m
o
#conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpconv1d_1/bias/m*
_output_shapes
:*
dtype0
x
dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel/m
q
"dense/kernel/m/Read/ReadVariableOpReadVariableOpdense/kernel/m*
_output_shapes

:*
dtype0
p
dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense/bias/m
i
 dense/bias/m/Read/ReadVariableOpReadVariableOpdense/bias/m*
_output_shapes
:*
dtype0

conv1d_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*"
shared_nameconv1d_0/kernel/v
{
%conv1d_0/kernel/v/Read/ReadVariableOpReadVariableOpconv1d_0/kernel/v*"
_output_shapes
:Z*
dtype0
v
conv1d_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z* 
shared_nameconv1d_0/bias/v
o
#conv1d_0/bias/v/Read/ReadVariableOpReadVariableOpconv1d_0/bias/v*
_output_shapes
:Z*
dtype0

conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*"
shared_nameconv1d_1/kernel/v
{
%conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpconv1d_1/kernel/v*"
_output_shapes
:Z*
dtype0
v
conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/bias/v
o
#conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpconv1d_1/bias/v*
_output_shapes
:*
dtype0
x
dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel/v
q
"dense/kernel/v/Read/ReadVariableOpReadVariableOpdense/kernel/v*
_output_shapes

:*
dtype0
p
dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense/bias/v
i
 dense/bias/v/Read/ReadVariableOpReadVariableOpdense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Â)
value¸)Bµ) B®)

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
lmVmWmXmY mZ!m[v\v]v^v_ v`!va
*
0
1
2
3
 4
!5
*
0
1
2
3
 4
!5
 
­
	variables
trainable_variables
	regularization_losses
&layer_regularization_losses
'non_trainable_variables
(metrics

)layers
*layer_metrics
 
[Y
VARIABLE_VALUEconv1d_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
	variables
trainable_variables
+layer_regularization_losses
regularization_losses
,non_trainable_variables
-metrics

.layers
/layer_metrics
 
 
 
­
	variables
trainable_variables
0layer_regularization_losses
regularization_losses
1non_trainable_variables
2metrics

3layers
4layer_metrics
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
	variables
trainable_variables
5layer_regularization_losses
regularization_losses
6non_trainable_variables
7metrics

8layers
9layer_metrics
 
 
 
­
	variables
trainable_variables
:layer_regularization_losses
regularization_losses
;non_trainable_variables
<metrics

=layers
>layer_metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
­
"	variables
#trainable_variables
?layer_regularization_losses
$regularization_losses
@non_trainable_variables
Ametrics

Blayers
Clayer_metrics
 
 

D0
E1
F2
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Gtotal
	Hcount
I	variables
J	keras_api
D
	Ktotal
	Lcount
M
_fn_kwargs
N	variables
O	keras_api
p
Ptrue_positives
Qtrue_negatives
Rfalse_positives
Sfalse_negatives
T	variables
U	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

G0
H1

I	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1

N	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1
R2
S3

T	variables
yw
VARIABLE_VALUEconv1d_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv1d_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv1d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv1d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv1d_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv1d_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv1d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv1d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv1d_0_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_0_inputconv1d_0/kernelconv1d_0/biasconv1d_1/kernelconv1d_1/biasdense/kernel
dense/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_3416
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ã	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_0/kernel/Read/ReadVariableOp!conv1d_0/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp%conv1d_0/kernel/m/Read/ReadVariableOp#conv1d_0/bias/m/Read/ReadVariableOp%conv1d_1/kernel/m/Read/ReadVariableOp#conv1d_1/bias/m/Read/ReadVariableOp"dense/kernel/m/Read/ReadVariableOp dense/bias/m/Read/ReadVariableOp%conv1d_0/kernel/v/Read/ReadVariableOp#conv1d_0/bias/v/Read/ReadVariableOp%conv1d_1/kernel/v/Read/ReadVariableOp#conv1d_1/bias/v/Read/ReadVariableOp"dense/kernel/v/Read/ReadVariableOp dense/bias/v/Read/ReadVariableOpConst*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_3703
¶
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_0/kernelconv1d_0/biasconv1d_1/kernelconv1d_1/biasdense/kernel
dense/biastotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negativesconv1d_0/kernel/mconv1d_0/bias/mconv1d_1/kernel/mconv1d_1/bias/mdense/kernel/mdense/bias/mconv1d_0/kernel/vconv1d_0/bias/vconv1d_1/kernel/vconv1d_1/bias/vdense/kernel/vdense/bias/v*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_3791Öã
Á
µ
D__inference_sequential_layer_call_and_return_conditional_losses_3344

inputs
conv1d_0_3326
conv1d_0_3328
conv1d_1_3332
conv1d_1_3334

dense_3338

dense_3340
identity¢ conv1d_0/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall
 conv1d_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_0_3326conv1d_0_3328*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_0_layer_call_and_return_conditional_losses_32212"
 conv1d_0/StatefulPartitionedCall
max_pooling1d_0/PartitionedCallPartitionedCall)conv1d_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling1d_0_layer_call_and_return_conditional_losses_31822!
max_pooling1d_0/PartitionedCall´
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_0/PartitionedCall:output:0conv1d_1_3332conv1d_1_3334*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_32542"
 conv1d_1/StatefulPartitionedCall
$global_max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_31952&
$global_max_pooling1d/PartitionedCall¦
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_3338
dense_3340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_32822
dense/StatefulPartitionedCallà
IdentityIdentity&dense/StatefulPartitionedCall:output:0!^conv1d_0/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2D
 conv1d_0/StatefulPartitionedCall conv1d_0/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
J
.__inference_max_pooling1d_0_layer_call_fn_3188

inputs
identityÝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling1d_0_layer_call_and_return_conditional_losses_31822
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

õ
B__inference_conv1d_0_layer_call_and_return_conditional_losses_3221

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

õ
B__inference_conv1d_1_layer_call_and_return_conditional_losses_3573

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿZ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
 
_user_specified_nameinputs
ë	
Ø
?__inference_dense_layer_call_and_return_conditional_losses_3282

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
½
D__inference_sequential_layer_call_and_return_conditional_losses_3299
conv1d_0_input
conv1d_0_3232
conv1d_0_3234
conv1d_1_3265
conv1d_1_3267

dense_3293

dense_3295
identity¢ conv1d_0/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall
 conv1d_0/StatefulPartitionedCallStatefulPartitionedCallconv1d_0_inputconv1d_0_3232conv1d_0_3234*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_0_layer_call_and_return_conditional_losses_32212"
 conv1d_0/StatefulPartitionedCall
max_pooling1d_0/PartitionedCallPartitionedCall)conv1d_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling1d_0_layer_call_and_return_conditional_losses_31822!
max_pooling1d_0/PartitionedCall´
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_0/PartitionedCall:output:0conv1d_1_3265conv1d_1_3267*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_32542"
 conv1d_1/StatefulPartitionedCall
$global_max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_31952&
$global_max_pooling1d/PartitionedCall¦
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_3293
dense_3295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_32822
dense/StatefulPartitionedCallà
IdentityIdentity&dense/StatefulPartitionedCall:output:0!^conv1d_0/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2D
 conv1d_0/StatefulPartitionedCall conv1d_0/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv1d_0_input

j
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_3195

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
»
"__inference_signature_wrapper_3416
conv1d_0_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv1d_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_31732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv1d_0_input
¢6
å
D__inference_sequential_layer_call_and_return_conditional_losses_3498

inputs8
4conv1d_0_conv1d_expanddims_1_readvariableop_resource,
(conv1d_0_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity¢conv1d_0/BiasAdd/ReadVariableOp¢+conv1d_0/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
conv1d_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_0/conv1d/ExpandDims/dim±
conv1d_0/conv1d/ExpandDims
ExpandDimsinputs'conv1d_0/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_0/conv1d/ExpandDimsÓ
+conv1d_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype02-
+conv1d_0/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_0/conv1d/ExpandDims_1/dimÛ
conv1d_0/conv1d/ExpandDims_1
ExpandDims3conv1d_0/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2
conv1d_0/conv1d/ExpandDims_1Ú
conv1d_0/conv1dConv2D#conv1d_0/conv1d/ExpandDims:output:0%conv1d_0/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
paddingSAME*
strides
2
conv1d_0/conv1d­
conv1d_0/conv1d/SqueezeSqueezeconv1d_0/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_0/conv1d/Squeeze§
conv1d_0/BiasAdd/ReadVariableOpReadVariableOp(conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
conv1d_0/BiasAdd/ReadVariableOp°
conv1d_0/BiasAddBiasAdd conv1d_0/conv1d/Squeeze:output:0'conv1d_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
conv1d_0/BiasAddw
conv1d_0/ReluReluconv1d_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
conv1d_0/Relu
max_pooling1d_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_0/ExpandDims/dimÆ
max_pooling1d_0/ExpandDims
ExpandDimsconv1d_0/Relu:activations:0'max_pooling1d_0/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
max_pooling1d_0/ExpandDimsÎ
max_pooling1d_0/MaxPoolMaxPool#max_pooling1d_0/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
ksize
*
paddingSAME*
strides
2
max_pooling1d_0/MaxPool¬
max_pooling1d_0/SqueezeSqueeze max_pooling1d_0/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
squeeze_dims
2
max_pooling1d_0/Squeeze
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_1/conv1d/ExpandDims/dimË
conv1d_1/conv1d/ExpandDims
ExpandDims max_pooling1d_0/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
conv1d_1/conv1d/ExpandDimsÓ
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimÛ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2
conv1d_1/conv1d/ExpandDims_1Û
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d_1/conv1d­
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_1/conv1d/Squeeze§
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp°
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_1/Relu
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*global_max_pooling1d/Max/reduction_indices¿
global_max_pooling1d/MaxMaxconv1d_1/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_max_pooling1d/Max
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp 
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SigmoidÂ
IdentityIdentitydense/Sigmoid:y:0 ^conv1d_0/BiasAdd/ReadVariableOp,^conv1d_0/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2B
conv1d_0/BiasAdd/ReadVariableOpconv1d_0/BiasAdd/ReadVariableOp2Z
+conv1d_0/conv1d/ExpandDims_1/ReadVariableOp+conv1d_0/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
º
)__inference_sequential_layer_call_fn_3515

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_33442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢6
å
D__inference_sequential_layer_call_and_return_conditional_losses_3457

inputs8
4conv1d_0_conv1d_expanddims_1_readvariableop_resource,
(conv1d_0_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity¢conv1d_0/BiasAdd/ReadVariableOp¢+conv1d_0/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
conv1d_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_0/conv1d/ExpandDims/dim±
conv1d_0/conv1d/ExpandDims
ExpandDimsinputs'conv1d_0/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_0/conv1d/ExpandDimsÓ
+conv1d_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype02-
+conv1d_0/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_0/conv1d/ExpandDims_1/dimÛ
conv1d_0/conv1d/ExpandDims_1
ExpandDims3conv1d_0/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2
conv1d_0/conv1d/ExpandDims_1Ú
conv1d_0/conv1dConv2D#conv1d_0/conv1d/ExpandDims:output:0%conv1d_0/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
paddingSAME*
strides
2
conv1d_0/conv1d­
conv1d_0/conv1d/SqueezeSqueezeconv1d_0/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_0/conv1d/Squeeze§
conv1d_0/BiasAdd/ReadVariableOpReadVariableOp(conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
conv1d_0/BiasAdd/ReadVariableOp°
conv1d_0/BiasAddBiasAdd conv1d_0/conv1d/Squeeze:output:0'conv1d_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
conv1d_0/BiasAddw
conv1d_0/ReluReluconv1d_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
conv1d_0/Relu
max_pooling1d_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_0/ExpandDims/dimÆ
max_pooling1d_0/ExpandDims
ExpandDimsconv1d_0/Relu:activations:0'max_pooling1d_0/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
max_pooling1d_0/ExpandDimsÎ
max_pooling1d_0/MaxPoolMaxPool#max_pooling1d_0/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
ksize
*
paddingSAME*
strides
2
max_pooling1d_0/MaxPool¬
max_pooling1d_0/SqueezeSqueeze max_pooling1d_0/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
squeeze_dims
2
max_pooling1d_0/Squeeze
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_1/conv1d/ExpandDims/dimË
conv1d_1/conv1d/ExpandDims
ExpandDims max_pooling1d_0/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
conv1d_1/conv1d/ExpandDimsÓ
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimÛ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2
conv1d_1/conv1d/ExpandDims_1Û
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d_1/conv1d­
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_1/conv1d/Squeeze§
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp°
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_1/Relu
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*global_max_pooling1d/Max/reduction_indices¿
global_max_pooling1d/MaxMaxconv1d_1/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_max_pooling1d/Max
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp 
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SigmoidÂ
IdentityIdentitydense/Sigmoid:y:0 ^conv1d_0/BiasAdd/ReadVariableOp,^conv1d_0/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2B
conv1d_0/BiasAdd/ReadVariableOpconv1d_0/BiasAdd/ReadVariableOp2Z
+conv1d_0/conv1d/ExpandDims_1/ReadVariableOp+conv1d_0/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
µ
D__inference_sequential_layer_call_and_return_conditional_losses_3382

inputs
conv1d_0_3364
conv1d_0_3366
conv1d_1_3370
conv1d_1_3372

dense_3376

dense_3378
identity¢ conv1d_0/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall
 conv1d_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_0_3364conv1d_0_3366*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_0_layer_call_and_return_conditional_losses_32212"
 conv1d_0/StatefulPartitionedCall
max_pooling1d_0/PartitionedCallPartitionedCall)conv1d_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling1d_0_layer_call_and_return_conditional_losses_31822!
max_pooling1d_0/PartitionedCall´
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_0/PartitionedCall:output:0conv1d_1_3370conv1d_1_3372*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_32542"
 conv1d_1/StatefulPartitionedCall
$global_max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_31952&
$global_max_pooling1d/PartitionedCall¦
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_3376
dense_3378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_32822
dense/StatefulPartitionedCallà
IdentityIdentity&dense/StatefulPartitionedCall:output:0!^conv1d_0/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2D
 conv1d_0/StatefulPartitionedCall conv1d_0/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
y
$__inference_dense_layer_call_fn_3602

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_32822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£A
Ì
__inference__wrapped_model_3173
conv1d_0_inputC
?sequential_conv1d_0_conv1d_expanddims_1_readvariableop_resource7
3sequential_conv1d_0_biasadd_readvariableop_resourceC
?sequential_conv1d_1_conv1d_expanddims_1_readvariableop_resource7
3sequential_conv1d_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity¢*sequential/conv1d_0/BiasAdd/ReadVariableOp¢6sequential/conv1d_0/conv1d/ExpandDims_1/ReadVariableOp¢*sequential/conv1d_1/BiasAdd/ReadVariableOp¢6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¡
)sequential/conv1d_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2+
)sequential/conv1d_0/conv1d/ExpandDims/dimÚ
%sequential/conv1d_0/conv1d/ExpandDims
ExpandDimsconv1d_0_input2sequential/conv1d_0/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential/conv1d_0/conv1d/ExpandDimsô
6sequential/conv1d_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?sequential_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype028
6sequential/conv1d_0/conv1d/ExpandDims_1/ReadVariableOp
+sequential/conv1d_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/conv1d_0/conv1d/ExpandDims_1/dim
'sequential/conv1d_0/conv1d/ExpandDims_1
ExpandDims>sequential/conv1d_0/conv1d/ExpandDims_1/ReadVariableOp:value:04sequential/conv1d_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2)
'sequential/conv1d_0/conv1d/ExpandDims_1
sequential/conv1d_0/conv1dConv2D.sequential/conv1d_0/conv1d/ExpandDims:output:00sequential/conv1d_0/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
paddingSAME*
strides
2
sequential/conv1d_0/conv1dÎ
"sequential/conv1d_0/conv1d/SqueezeSqueeze#sequential/conv1d_0/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2$
"sequential/conv1d_0/conv1d/SqueezeÈ
*sequential/conv1d_0/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02,
*sequential/conv1d_0/BiasAdd/ReadVariableOpÜ
sequential/conv1d_0/BiasAddBiasAdd+sequential/conv1d_0/conv1d/Squeeze:output:02sequential/conv1d_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
sequential/conv1d_0/BiasAdd
sequential/conv1d_0/ReluRelu$sequential/conv1d_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
sequential/conv1d_0/Relu
)sequential/max_pooling1d_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential/max_pooling1d_0/ExpandDims/dimò
%sequential/max_pooling1d_0/ExpandDims
ExpandDims&sequential/conv1d_0/Relu:activations:02sequential/max_pooling1d_0/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2'
%sequential/max_pooling1d_0/ExpandDimsï
"sequential/max_pooling1d_0/MaxPoolMaxPool.sequential/max_pooling1d_0/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
ksize
*
paddingSAME*
strides
2$
"sequential/max_pooling1d_0/MaxPoolÍ
"sequential/max_pooling1d_0/SqueezeSqueeze+sequential/max_pooling1d_0/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
squeeze_dims
2$
"sequential/max_pooling1d_0/Squeeze¡
)sequential/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2+
)sequential/conv1d_1/conv1d/ExpandDims/dim÷
%sequential/conv1d_1/conv1d/ExpandDims
ExpandDims+sequential/max_pooling1d_0/Squeeze:output:02sequential/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2'
%sequential/conv1d_1/conv1d/ExpandDimsô
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?sequential_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype028
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
+sequential/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/conv1d_1/conv1d/ExpandDims_1/dim
'sequential/conv1d_1/conv1d/ExpandDims_1
ExpandDims>sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:04sequential/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2)
'sequential/conv1d_1/conv1d/ExpandDims_1
sequential/conv1d_1/conv1dConv2D.sequential/conv1d_1/conv1d/ExpandDims:output:00sequential/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
sequential/conv1d_1/conv1dÎ
"sequential/conv1d_1/conv1d/SqueezeSqueeze#sequential/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2$
"sequential/conv1d_1/conv1d/SqueezeÈ
*sequential/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv1d_1/BiasAdd/ReadVariableOpÜ
sequential/conv1d_1/BiasAddBiasAdd+sequential/conv1d_1/conv1d/Squeeze:output:02sequential/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/conv1d_1/BiasAdd
sequential/conv1d_1/ReluRelu$sequential/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/conv1d_1/Relu°
5sequential/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential/global_max_pooling1d/Max/reduction_indicesë
#sequential/global_max_pooling1d/MaxMax&sequential/conv1d_1/Relu:activations:0>sequential/global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential/global_max_pooling1d/MaxÀ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&sequential/dense/MatMul/ReadVariableOpÌ
sequential/dense/MatMulMatMul,sequential/global_max_pooling1d/Max:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/MatMul¿
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/BiasAdd
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/Sigmoid
IdentityIdentitysequential/dense/Sigmoid:y:0+^sequential/conv1d_0/BiasAdd/ReadVariableOp7^sequential/conv1d_0/conv1d/ExpandDims_1/ReadVariableOp+^sequential/conv1d_1/BiasAdd/ReadVariableOp7^sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2X
*sequential/conv1d_0/BiasAdd/ReadVariableOp*sequential/conv1d_0/BiasAdd/ReadVariableOp2p
6sequential/conv1d_0/conv1d/ExpandDims_1/ReadVariableOp6sequential/conv1d_0/conv1d/ExpandDims_1/ReadVariableOp2X
*sequential/conv1d_1/BiasAdd/ReadVariableOp*sequential/conv1d_1/BiasAdd/ReadVariableOp2p
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv1d_0_input

õ
B__inference_conv1d_1_layer_call_and_return_conditional_losses_3254

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿZ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
 
_user_specified_nameinputs
è
|
'__inference_conv1d_0_layer_call_fn_3557

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_0_layer_call_and_return_conditional_losses_32212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
½
D__inference_sequential_layer_call_and_return_conditional_losses_3320
conv1d_0_input
conv1d_0_3302
conv1d_0_3304
conv1d_1_3308
conv1d_1_3310

dense_3314

dense_3316
identity¢ conv1d_0/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall
 conv1d_0/StatefulPartitionedCallStatefulPartitionedCallconv1d_0_inputconv1d_0_3302conv1d_0_3304*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_0_layer_call_and_return_conditional_losses_32212"
 conv1d_0/StatefulPartitionedCall
max_pooling1d_0/PartitionedCallPartitionedCall)conv1d_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling1d_0_layer_call_and_return_conditional_losses_31822!
max_pooling1d_0/PartitionedCall´
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_0/PartitionedCall:output:0conv1d_1_3308conv1d_1_3310*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_32542"
 conv1d_1/StatefulPartitionedCall
$global_max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_31952&
$global_max_pooling1d/PartitionedCall¦
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_3314
dense_3316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_32822
dense/StatefulPartitionedCallà
IdentityIdentity&dense/StatefulPartitionedCall:output:0!^conv1d_0/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2D
 conv1d_0/StatefulPartitionedCall conv1d_0/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv1d_0_input
Ên
ã
 __inference__traced_restore_3791
file_prefix$
 assignvariableop_conv1d_0_kernel$
 assignvariableop_1_conv1d_0_bias&
"assignvariableop_2_conv1d_1_kernel$
 assignvariableop_3_conv1d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias
assignvariableop_6_total
assignvariableop_7_count
assignvariableop_8_total_1
assignvariableop_9_count_1&
"assignvariableop_10_true_positives&
"assignvariableop_11_true_negatives'
#assignvariableop_12_false_positives'
#assignvariableop_13_false_negatives)
%assignvariableop_14_conv1d_0_kernel_m'
#assignvariableop_15_conv1d_0_bias_m)
%assignvariableop_16_conv1d_1_kernel_m'
#assignvariableop_17_conv1d_1_bias_m&
"assignvariableop_18_dense_kernel_m$
 assignvariableop_19_dense_bias_m)
%assignvariableop_20_conv1d_0_kernel_v'
#assignvariableop_21_conv1d_0_bias_v)
%assignvariableop_22_conv1d_1_kernel_v'
#assignvariableop_23_conv1d_1_bias_v&
"assignvariableop_24_dense_kernel_v$
 assignvariableop_25_dense_bias_v
identity_27¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9«
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
value­BªB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices³
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv1d_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ª
AssignVariableOp_10AssignVariableOp"assignvariableop_10_true_positivesIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_true_negativesIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_false_positivesIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13«
AssignVariableOp_13AssignVariableOp#assignvariableop_13_false_negativesIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14­
AssignVariableOp_14AssignVariableOp%assignvariableop_14_conv1d_0_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15«
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv1d_0_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16­
AssignVariableOp_16AssignVariableOp%assignvariableop_16_conv1d_1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17«
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv1d_1_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ª
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¨
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20­
AssignVariableOp_20AssignVariableOp%assignvariableop_20_conv1d_0_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21«
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv1d_0_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22­
AssignVariableOp_22AssignVariableOp%assignvariableop_22_conv1d_1_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23«
AssignVariableOp_23AssignVariableOp#assignvariableop_23_conv1d_1_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ª
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¨
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26
Identity_27IdentityIdentity_26:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_27"#
identity_27Identity_27:output:0*}
_input_shapesl
j: ::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ý
Â
)__inference_sequential_layer_call_fn_3397
conv1d_0_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallconv1d_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_33822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv1d_0_input
ë	
Ø
?__inference_dense_layer_call_and_return_conditional_losses_3593

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
|
'__inference_conv1d_1_layer_call_fn_3582

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_32542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿZ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
 
_user_specified_nameinputs

õ
B__inference_conv1d_0_layer_call_and_return_conditional_losses_3548

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Z*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Z2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
º
)__inference_sequential_layer_call_fn_3532

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_33822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
O
3__inference_global_max_pooling1d_layer_call_fn_3201

inputs
identityÕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_31952
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
Â
)__inference_sequential_layer_call_fn_3359
conv1d_0_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallconv1d_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_33442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv1d_0_input
<


__inference__traced_save_3703
file_prefix.
*savev2_conv1d_0_kernel_read_readvariableop,
(savev2_conv1d_0_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop0
,savev2_conv1d_0_kernel_m_read_readvariableop.
*savev2_conv1d_0_bias_m_read_readvariableop0
,savev2_conv1d_1_kernel_m_read_readvariableop.
*savev2_conv1d_1_bias_m_read_readvariableop-
)savev2_dense_kernel_m_read_readvariableop+
'savev2_dense_bias_m_read_readvariableop0
,savev2_conv1d_0_kernel_v_read_readvariableop.
*savev2_conv1d_0_bias_v_read_readvariableop0
,savev2_conv1d_1_kernel_v_read_readvariableop.
*savev2_conv1d_1_bias_v_read_readvariableop-
)savev2_dense_kernel_v_read_readvariableop+
'savev2_dense_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¥
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
value­BªB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¾
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_0_kernel_read_readvariableop(savev2_conv1d_0_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop,savev2_conv1d_0_kernel_m_read_readvariableop*savev2_conv1d_0_bias_m_read_readvariableop,savev2_conv1d_1_kernel_m_read_readvariableop*savev2_conv1d_1_bias_m_read_readvariableop)savev2_dense_kernel_m_read_readvariableop'savev2_dense_bias_m_read_readvariableop,savev2_conv1d_0_kernel_v_read_readvariableop*savev2_conv1d_0_bias_v_read_readvariableop,savev2_conv1d_1_kernel_v_read_readvariableop*savev2_conv1d_1_bias_v_read_readvariableop)savev2_dense_kernel_v_read_readvariableop'savev2_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*å
_input_shapesÓ
Ð: :Z:Z:Z:::: : : : :È:È:È:È:Z:Z:Z::::Z:Z:Z:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:Z: 

_output_shapes
:Z:($
"
_output_shapes
:Z: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È:($
"
_output_shapes
:Z: 

_output_shapes
:Z:($
"
_output_shapes
:Z: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::($
"
_output_shapes
:Z: 

_output_shapes
:Z:($
"
_output_shapes
:Z: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
å
e
I__inference_max_pooling1d_0_layer_call_and_return_conditional_losses_3182

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDims°
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*º
serving_default¦
M
conv1d_0_input;
 serving_default_conv1d_0_input:0ÿÿÿÿÿÿÿÿÿ9
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:þ
 R
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
b__call__
c_default_save_signature
*d&call_and_return_all_conditional_losses"ÆO
_tf_keras_sequential§O{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_0_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "filters": 90, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_0", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 23, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_0_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "filters": 90, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_0", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 23, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
Ý


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"¸	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "filters": 90, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 1]}}
ø
	variables
trainable_variables
regularization_losses
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "MaxPooling1D", "name": "max_pooling1d_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_0", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
æ	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
i__call__
*j&call_and_return_all_conditional_losses"Á
_tf_keras_layer§{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 23, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 90}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 90]}}

	variables
trainable_variables
regularization_losses
	keras_api
k__call__
*l&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
î

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
m__call__
*n&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23]}}
mVmWmXmY mZ!m[v\v]v^v_ v`!va"
	optimizer
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
	variables
trainable_variables
	regularization_losses
&layer_regularization_losses
'non_trainable_variables
(metrics

)layers
*layer_metrics
b__call__
c_default_save_signature
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
,
oserving_default"
signature_map
%:#Z2conv1d_0/kernel
:Z2conv1d_0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
trainable_variables
+layer_regularization_losses
regularization_losses
,non_trainable_variables
-metrics

.layers
/layer_metrics
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
trainable_variables
0layer_regularization_losses
regularization_losses
1non_trainable_variables
2metrics

3layers
4layer_metrics
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
%:#Z2conv1d_1/kernel
:2conv1d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
trainable_variables
5layer_regularization_losses
regularization_losses
6non_trainable_variables
7metrics

8layers
9layer_metrics
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
trainable_variables
:layer_regularization_losses
regularization_losses
;non_trainable_variables
<metrics

=layers
>layer_metrics
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
"	variables
#trainable_variables
?layer_regularization_losses
$regularization_losses
@non_trainable_variables
Ametrics

Blayers
Clayer_metrics
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»
	Gtotal
	Hcount
I	variables
J	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ú
	Ktotal
	Lcount
M
_fn_kwargs
N	variables
O	keras_api"³
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
¯"
Ptrue_positives
Qtrue_negatives
Rfalse_positives
Sfalse_negatives
T	variables
U	keras_api"¼!
_tf_keras_metric¡!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
:  (2total
:  (2count
.
G0
H1"
trackable_list_wrapper
-
I	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
K0
L1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
<
P0
Q1
R2
S3"
trackable_list_wrapper
-
T	variables"
_generic_user_object
%:#Z2conv1d_0/kernel/m
:Z2conv1d_0/bias/m
%:#Z2conv1d_1/kernel/m
:2conv1d_1/bias/m
:2dense/kernel/m
:2dense/bias/m
%:#Z2conv1d_0/kernel/v
:Z2conv1d_0/bias/v
%:#Z2conv1d_1/kernel/v
:2conv1d_1/bias/v
:2dense/kernel/v
:2dense/bias/v
ò2ï
)__inference_sequential_layer_call_fn_3532
)__inference_sequential_layer_call_fn_3359
)__inference_sequential_layer_call_fn_3397
)__inference_sequential_layer_call_fn_3515À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
è2å
__inference__wrapped_model_3173Á
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *1¢.
,)
conv1d_0_inputÿÿÿÿÿÿÿÿÿ
Þ2Û
D__inference_sequential_layer_call_and_return_conditional_losses_3299
D__inference_sequential_layer_call_and_return_conditional_losses_3320
D__inference_sequential_layer_call_and_return_conditional_losses_3498
D__inference_sequential_layer_call_and_return_conditional_losses_3457À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
'__inference_conv1d_0_layer_call_fn_3557¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_conv1d_0_layer_call_and_return_conditional_losses_3548¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_max_pooling1d_0_layer_call_fn_3188Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¤2¡
I__inference_max_pooling1d_0_layer_call_and_return_conditional_losses_3182Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ñ2Î
'__inference_conv1d_1_layer_call_fn_3582¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_conv1d_1_layer_call_and_return_conditional_losses_3573¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
3__inference_global_max_pooling1d_layer_call_fn_3201Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
©2¦
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_3195Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Î2Ë
$__inference_dense_layer_call_fn_3602¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_dense_layer_call_and_return_conditional_losses_3593¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÐBÍ
"__inference_signature_wrapper_3416conv1d_0_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
__inference__wrapped_model_3173t !;¢8
1¢.
,)
conv1d_0_inputÿÿÿÿÿÿÿÿÿ
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿª
B__inference_conv1d_0_layer_call_and_return_conditional_losses_3548d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿZ
 
'__inference_conv1d_0_layer_call_fn_3557W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿZª
B__inference_conv1d_1_layer_call_and_return_conditional_losses_3573d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿZ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_conv1d_1_layer_call_fn_3582W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿZ
ª "ÿÿÿÿÿÿÿÿÿ
?__inference_dense_layer_call_and_return_conditional_losses_3593\ !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
$__inference_dense_layer_call_fn_3602O !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÉ
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_3195wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¡
3__inference_global_max_pooling1d_layer_call_fn_3201jE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
I__inference_max_pooling1d_0_layer_call_and_return_conditional_losses_3182E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ©
.__inference_max_pooling1d_0_layer_call_fn_3188wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
D__inference_sequential_layer_call_and_return_conditional_losses_3299t !C¢@
9¢6
,)
conv1d_0_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
D__inference_sequential_layer_call_and_return_conditional_losses_3320t !C¢@
9¢6
,)
conv1d_0_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
D__inference_sequential_layer_call_and_return_conditional_losses_3457l !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
D__inference_sequential_layer_call_and_return_conditional_losses_3498l !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_sequential_layer_call_fn_3359g !C¢@
9¢6
,)
conv1d_0_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_3397g !C¢@
9¢6
,)
conv1d_0_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_3515_ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_3532_ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ­
"__inference_signature_wrapper_3416 !M¢J
¢ 
Cª@
>
conv1d_0_input,)
conv1d_0_inputÿÿÿÿÿÿÿÿÿ"-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ