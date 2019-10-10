# Pytorch ProtoNN Examples

This directory includes an example [notebook](protoNN_example.ipynb)  and a
command line execution script of ProtoNN developed as part of EdgeML. The
example is based on the USPS dataset.

`pytorch_edgeml.graph.protoNN` implements the ProtoNN prediction functions.
The training routine for ProtoNN is decoupled from the forward graph to
facilitate a plug and play behaviour wherein ProtoNN can be combined with or
used as a final layer classifier for other architectures (RNNs, CNNs). The
training routine is implemented in `pytorch_edgeml.trainer.protoNNTrainer`.
(This is also an artifact of consistency requirements with Tensorflow
implementation).

Note that, `protoNN_example.py` assumes the data to be in a specific format.  It
is assumed that train and test data is contained in two files, `train.npy` and
`test.npy`. Each containing a 2D numpy array of dimension `[numberOfExamples,
numberOfFeatures + 1]`. The first column of each matrix is assumed to contain
label information. For an N-Class problem, we assume the labels are integers
from 0 through N-1. 

**Tested With:** pytorch > 1.1.0 with Python 2 and Python 3


## Running the ProtoNN execution script

Along with the example notebook, a command line execution script for ProtoNN is
provided in `protoNN_example.py`. After the data has been setup, this
script can be used with the following command:

```
python protoNN_example.py \
      --data-dir ./mpu \
      --projection-dim 10 \
      --num-prototypes 24 \
      --gamma 0.0014 \
      --learning-rate 0.01 \
      --epochs 200 \
      --val-step 10 \
      --output-dir ./
```

You can expect a test set accuracy of about approximately 93%.

Copyright (c) Microsoft Corporation. All rights reserved. 
Licensed under the MIT license.
