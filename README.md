# KDD Quantum Machine Learning Classifier

This project implements a quantum machine learning classifier for network packet inspection on the KDD dataset using PennyLane and JAX. The program uses PCA and Mutual Information scores to reduce the features and supports multiple circuit types with configurable parameters.

## Setup

1. Create and activate a Python virtual environment (optional but recommended).
2. Install all required packages using:

   pip install -r requirements.txt


## Running the Model

The main program is `kdd_qml.py` and supports several command-line arguments to control the parameters:

python kdd_qml.py [--num_qubits INT] [--train_rows INT] [--batch_size INT] [--epochs INT] [--stepsize FLOAT] [--test_rows INT] [--circuit INT] [--force_default]


## Arguments

`--num_qubits` : Number of qubits to use (default: 8)

`--train_rows` : Number of training samples (default: 5000)

`--batch_size` : Batch size for training (default: 64)

`--epochs` : Number of training epochs (default: 15)

`--stepsize` : Learning rate / optimizer step size (default: 0.0025)

`--test_rows` : Number of test samples (default: 10000)

`--circuit` : Circuit type (1 = Simple, 2 = Custom, 3 = Templates) (default: 2)

`--force_default` : Force use of `default.qubit` backend if `lightning.qubit` is not wanted or CPU does not support AVX

## Example

python kdd_qml.py --num_qubits 6 --train_rows 800 --epochs 10 --circuit 3

If using docker image:

docker run --rm qml_kdd --num_qubits 6 --train_rows 800 --epochs 10 --circuit 3

This will run the classifier using 6 qubits, 800 training samples, 10 epochs, and the template-based circuit.

## Possible Improvements

- Try using lightning.gpu instead of lightning qubit.
- Better circuit structure (Multiple instances of encoding, more randomized entanglement, etc...)
- Use average of all qubits for classification instead of just qubit 0
