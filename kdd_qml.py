import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers
from pennylane import DeviceError
import argparse
import os
import jax
import jax.numpy as jnp
import jax.random as random
import optax


#Command line arguments for parameters
parser = argparse.ArgumentParser(
    description="Quantum Machine Learning Classifier for KDD dataset using PennyLane and JAX.",
    epilog="""
Example:
  python kdd_qml.py --num_qubits 6 --train_rows 800 --epochs 10 --circuit 3

Circuit Types:
  1 = Simple
  2 = Custom (Default)
  3 = Templates

Use --force_default to force use of 'default.qubit' instead of 'lightning.qubit' if CPU does not support AVX.
""",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('--num_qubits', type=int, default=8, help='Number of qubits to use (default: 8)')
parser.add_argument('--train_rows', type=int, default=5000, help='Number of training samples (default: 5000)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs (default: 15)')
parser.add_argument('--stepsize', type=float, default=0.0025, help='Optimizer step size (default: 0.0025)')
parser.add_argument('--test_rows', type=int, default=10000, help='Number of test samples (default: 10000)')
parser.add_argument('--circuit', type=int, default=2, help='Circuit type: 1 = Simple, 2 = Custom, 3 = Templates (default: 2)')
parser.add_argument('--force_default', action='store_true', help="Force use of default.qubit backend")
args = parser.parse_args()

jax.config.update("jax_enable_x64", False)

print("Current Parameters:-")
print("Number of qubits - ", (args.num_qubits))
print("Training rows - ", (args.train_rows))
print("Test rows - ", (args.test_rows))
print("Circuit number - ", (args.circuit))
print("Epochs - ", (args.epochs))
print("Stepsize - ", (args.stepsize))



#Pre-processing the dataset
df = pd.read_csv("kddcup.data_10_percent_corrected", header=None)

df.columns = [ "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label" ]

df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])

X = df.drop('label', axis=1)
Y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA()
X_pca_all = pca.fit_transform(X_scaled)

if os.path.exists("top_pca_indices.npy"):
    print("Using pre-computed mutual information scores")
    top_indices = np.load("top_pca_indices.npy")
else:
    print("Pre-computed scores not found, calculating mutual information scores")
    mi_scores = mutual_info_classif(X_pca_all, Y)
    top_indices = mi_scores.argsort()[::-1]
    np.save("top_pca_indices.npy", top_indices)
    print("Mutual information scores calculated and saved")


top_indices = top_indices[:args.num_qubits]

X_pca_optimized = X_pca_all[:, top_indices]


X_train, X_test, Y_train, Y_test = train_test_split(X_pca_optimized, Y, test_size=0.2, random_state=42)


num_qubits = args.num_qubits

#Setting up backend
if args.force_default:
    print("Forcing default.qubit device")
    dev = qml.device("default.qubit", wires=num_qubits)
else:
    try:
        dev = qml.device("lightning.qubit", wires=num_qubits)
        print("Using lightning.qubit device")
    except (DeviceError, OSError, RuntimeError, ValueError) as e:
        print(f"Failed to use lightning.qubit: {e}")
        print("Falling back to default.qubit device")
        dev = qml.device("default.qubit", wires=num_qubits)

#circuit1
def circuit_simple(x, weights):

    for i in range(num_qubits):
        qml.RY(x[i], wires=i)


    for i in range(num_qubits):
        qml.Rot(*weights[i], wires=i)


    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    return qml.expval(qml.PauliZ(0))

#circuit2
def circuit_circular_layers(x, weights):
    x = jnp.pi * x / jnp.linalg.norm(x)

    for i in range(num_qubits):
        qml.RY(x[i], wires=i)

    num_pqc_layers = weights.shape[0]


    for k in range(num_pqc_layers):
        for i in range(num_qubits):
            qml.Rot(*weights[k, i], wires=i)

        for i in range(num_qubits - 1):
            target = (i + 1) % num_qubits
            qml.CRY(weights[k, i, 0], wires=[i, target])


    return qml.expval(qml.PauliZ(0))

#circuit3
def circuit_templates(x, weights):
    x = jnp.pi * x / jnp.linalg.norm(x)
    qml.AngleEmbedding(features=x, wires=range(num_qubits), rotation='Y')
    qml.AngleEmbedding(features=x, wires=range(num_qubits), rotation='Z')
    qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))


if args.circuit == 1:
    active_circuit = circuit_simple
    active_weights_shape = (num_qubits, 3)
    print("Using Simple Circuit, weights_shape: ", active_weights_shape)
elif args.circuit == 2:
    active_circuit = circuit_circular_layers
    active_num_pqc_layers = 5
    active_weights_shape = (active_num_pqc_layers, num_qubits, 3)
    print("Using Custom circuit, weights_shape: ", active_weights_shape)
elif args.circuit == 3:
    active_circuit = circuit_templates
    active_num_pqc_layers = 5
    active_weights_shape = (active_num_pqc_layers, num_qubits, 3)
    print("Using Templates, weights_shape: ", active_weights_shape)
else:
    raise ValueError("Invalid active circuit selected")


key = random.PRNGKey(0)
weights = random.uniform(key, shape=active_weights_shape, minval=0, maxval=jnp.pi, dtype=jnp.float32)

#Setup Qnode
qnode = qml.QNode(active_circuit, dev, interface="jax")

X_train_small = jnp.array(X_train[:args.train_rows], dtype=jnp.float32)
Y_train_small = jnp.array(Y_train[:args.train_rows], dtype=jnp.float32)



batch_size = args.batch_size
epochs = args.epochs


#opt = qml.AdamOptimizer(stepsize=args.stepsize)

optimizer = optax.adam(args.stepsize)
opt_state = optimizer.init(weights)


def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8
    return -jnp.mean(y_true * jnp.log(y_pred + eps) + (1 - y_true) * jnp.log(1 - y_pred + eps))

@jax.jit
def predict(x_batch, weights):
    def single_predict(x): return (qnode(x, weights) + 1) / 2
    return jax.vmap(single_predict)(x_batch)	

@jax.jit
def cost_fn(w, x, y):
    preds = predict(x, w)
    return binary_cross_entropy(y, preds)

@jax.jit
def update(w, opt_state, x, y):
    loss, grads = jax.value_and_grad(cost_fn)(w, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, w)
    w = optax.apply_updates(w, updates)
    return w, opt_state, loss

#For early stopping
patience = 5
min_change = 1e-4
wait = 0
best_loss = float('inf')
best_weights = weights

#Training Loop
print("Starting training loop")

for epoch in range(epochs):

    #Using mini batching
    indices = jax.random.permutation(key, len(X_train_small))
    X_train_epoch = X_train_small[indices]
    Y_train_epoch = Y_train_small[indices]

    for i in range(0, len(X_train_small), batch_size):
        X_batch = X_train_epoch[i:i+batch_size]
        Y_batch = Y_train_epoch[i:i+batch_size]
        weights, opt_state, loss = update(weights, opt_state, X_batch, Y_batch)

    Y_train_pred = predict(X_train_small, weights)
    epoch_loss = binary_cross_entropy(Y_train_small, Y_train_pred)
    print(f"Epoch {epoch + 1} | Loss: {epoch_loss}")
    
    #Early Stopping check
    if best_loss - epoch_loss > min_change:
        best_loss = epoch_loss
        best_weights = weights
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement in last {patience} epochs)")
            weights = best_weights
            break


#Testing accuracy
Y_test_np = jnp.array(Y_test[:args.test_rows], dtype=jnp.float32)
X_test_np = jnp.array(X_test[:args.test_rows], dtype=jnp.float32)


@jax.jit
def predict_classes(x):	
    return jnp.where(predict(x, weights) >= 0.5, 1, 0)

predictions = predict_classes(X_test_np)
accuracy = jnp.mean(predictions == Y_test_np)
print("Test accuracy:", float(accuracy))

print("\nClassification Report:")
print(classification_report(Y_test_np, predictions, digits=4))
