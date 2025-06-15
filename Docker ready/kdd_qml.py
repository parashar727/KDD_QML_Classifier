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


parser = argparse.ArgumentParser()
parser.add_argument('--num_qubits', type=int, default=6)
parser.add_argument('--train_rows', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--stepsize', type=float, default=0.0025)
parser.add_argument('--test_rows', type=int, default=10000)
parser.add_argument('--circuit', type=int, default=2)
parser.add_argument('--force_default', action='store_true')
args = parser.parse_args()

print("Current Parameters:-")
print("Number of qubits - ", (args.num_qubits))
print("Training rows - ", (args.train_rows))
print("Test rows - ", (args.test_rows))
print("Circuit number - ", (args.circuit))
print("Epochs - ", (args.epochs))
print("Stepsize - ", (args.stepsize))




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
    x = np.pi * x / np.linalg.norm(x)

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
    x = np.pi * x / np.linalg.norm(x)
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



weights = np.random.uniform(0, np.pi, active_weights_shape, requires_grad=True)

qnode = qml.QNode(active_circuit, dev, interface="autograd")

X_train_small = np.array(X_train[:args.train_rows])
Y_train_small = np.array(Y_train[:args.train_rows])



batch_size = args.batch_size
epochs = args.epochs


opt = qml.AdamOptimizer(stepsize=args.stepsize)


def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


def predict(x_batch, weights):
    predictions = [qnode(x, weights) for x in x_batch]
    return np.array([(p + 1) / 2 for p in predictions])

print("Starting training loop")

for epoch in range(epochs):

    indices = np.arange(len(X_train_small))
    np.random.shuffle(indices)



    for i in range(0, len(X_train_small), batch_size):
        X_batch = X_train_small[indices[i:i+batch_size]]
        Y_batch = Y_train_small[indices[i:i+batch_size]]


        def cost_fn(w):
            Y_pred = predict(X_batch, w)
            return binary_cross_entropy(Y_batch, Y_pred)

        weights = opt.step(cost_fn, weights)


    Y_train_pred = predict(X_train_small, weights)
    epoch_loss = binary_cross_entropy(Y_train_small, Y_train_pred)
    print(f"Epoch {epoch + 1} | Loss: {epoch_loss}")

def predict_qnode(x):
    output = qnode(x, weights)
    prob = (output + 1) / 2
    return 1 if prob >= 0.5 else 0


Y_test_np = np.array(Y_test[:args.test_rows])
X_test_np = np.array(X_test[:args.test_rows])

predictions = [predict_qnode(x) for x in X_test_np]
accuracy = np.mean(predictions == Y_test_np)
print("Test accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(Y_test_np, predictions, digits=4))
