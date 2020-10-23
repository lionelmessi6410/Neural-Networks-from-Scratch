import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time
import argparse
from model import DeepNeuralNetwork

# Settings
parser = argparse.ArgumentParser(description='Neural Networks from Scratch')
parser.add_argument('--activation', action='store', dest='activation', required=False, default='sigmoid', help='activation function: sigmoid/relu')
parser.add_argument('--batch_size', action='store', dest='batch_size', required=False, default=128)
parser.add_argument('--optimizer', action='store', dest='optimizer', required=False, default='momentum', help='optimizer: sgd/momentum')
parser.add_argument('--l_rate', action='store', dest='l_rate', required=False, default=1e-3, help='learning rate')
parser.add_argument('--beta', action='store', dest='beta', required=False, default=.9, help='beta in momentum optimizer')
args = parser.parse_args()

# Helper function
def show_images(image, num_row=2, num_col=5):
    # plot images
    image_size = int(np.sqrt(image.shape[-1]))
    image = np.reshape(image, (image.shape[0], image_size, image_size))
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(image[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)

def main():
    # Load data
    print("Loading data...")
    mnist_data = fetch_openml("mnist_784")
    x = mnist_data["data"]
    y = mnist_data["target"]

    # Normalize
    print("Preprocessing data...")
    x /= 255.0

    # One-hot encode labels
    num_labels = 10
    examples = y.shape[0]
    y_new = one_hot(y.astype('int32'), num_labels)

    # Split, reshape, shuffle
    train_size = 60000
    test_size = x.shape[0] - train_size
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y_new[:train_size], y_new[train_size:]
    shuffle_index = np.random.permutation(train_size)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    print("Training data: {} {}".format(x_train.shape, y_train.shape))
    print("Test data: {} {}".format(x_test.shape, y_test.shape))
#     show_images(x_train)

    # Train
    print("Start training!")
    dnn = DeepNeuralNetwork(sizes=[784, 64, 10], activation=args.activation)
    dnn.train(x_train, y_train, x_test, y_test, 
              batch_size=int(args.batch_size), optimizer=args.optimizer, l_rate=float(args.l_rate), beta=float(args.beta))
    
if __name__ == '__main__':
    main()