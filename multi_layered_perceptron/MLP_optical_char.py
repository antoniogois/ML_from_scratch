import numpy as np
from math import sqrt
# from deep_structured_learning.linear_optical_char import read_chars
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
np.random.seed(0)

char_to_idx = {}
idx_to_char = {}


def read_chars(path, samples=0):
    x_train = []
    x_dev = []
    x_test = []
    y_train = []
    y_dev = []
    y_test = []
    # chars = set()  # won't be random with new use
    # chars = []  # list instead  of set to avoid randomness

    with open(path) as f:
        count = 0
        for sample in f:
            if samples > 0:
                count += 1
                if count > samples:
                    break

            sample = sample.split('\t')
            char = sample[1]  # character
            # chars.add(sample[1])
            if char not in char_to_idx:
                new_idx = len(char_to_idx)
                char_to_idx[char] = new_idx
                idx_to_char[new_idx] = char

            char = char_to_idx[char]

            s = [int(sample[i]) for i in range(6,len(sample)-1)]  # -1 ignores \n
            fold_id = int(sample[5])
            if fold_id < 8:
                x_train.append(s)
                y_train.append(char)
            elif fold_id == 8:
                x_dev.append(s)
                y_dev.append(char)
            else:  # fold_id == 9
                x_test.append(s)
                y_test.append(char)

    print('train size: {}, dev size: {}, test size: {}'.format(len(x_train),len(x_dev),len(x_test)))
    print('unique chars: {}'.format(len(char_to_idx)))

    # idx = 0
    # for char in chars:
    #     char_to_idx[char] = idx
    #     idx_to_char[idx] = char
    #     idx += 1

    return np.array(x_train,dtype=np.int), np.array(x_dev,dtype=np.int), np.array(x_test,dtype=np.int),\
           np.array(y_train,dtype=np.int), np.array(y_dev,dtype=np.int), np.array(y_test,dtype=np.int)


def plot_accuracies(train, dev, title=None, save_name=None):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # x = np.linspace(0, len(train)-1, len(train))
    plt.plot(train, label='train')
    plt.plot(dev, label='validation')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    if title is not None:
        plt.title(title)
    plt.legend()
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def sigmoid_derivative(z):
    g_z = 1/(1 + np.exp(-z))
    return g_z*(1 - g_z)


class SigmoidLayer:
    def __init__(self, input_dim, output_dim, reg_lambda=0.0, init_fc=None, is_output_layer=False, is_input_layer=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reg_lambda = reg_lambda
        self.is_output_layer = is_output_layer
        self.is_input_layer = is_input_layer
        self.b = np.zeros(output_dim)
        if init_fc is not None:
            self.W = init_fc(output_dim, input_dim)
        else:
            self.W = np.random.rand(output_dim, input_dim) - 0.5
        self.z = np.zeros(output_dim)  # output before non-linearization
        self.h = np.zeros(input_dim)  # input of this layer
        self.b_grad = np.zeros(output_dim)  # grads are stored and weights only updated in the end of each backprop iter
        self.W_grad = np.zeros([output_dim, input_dim])

    def forward(self, h):  # for first layer, input h = x
        self.h = h  # save input
        W = self.W
        b = self.b
        is_output_layer = self.is_output_layer

        z = np.matmul(W, h) + b
        self.z = z  # save output before non-linearization

        if not is_output_layer:
            return 1/(1+np.exp(-z))  # using sigmoid
        else:
            scores = z  # send without non-linearization (applying softmax is only important during backprop)
            y_hat = np.argmax(scores)  # also send index of highest score
            return y_hat, scores

    def backward(self, grad):  # receives grad from layer above
        if self.is_output_layer:
            z_grad = grad  # receives z_grad directly
        else:
            h_grad = grad  # z_grad has to be computed
            z_grad = np.multiply(h_grad, sigmoid_derivative(self.z))

        self.b_grad = z_grad
        self.W_grad = np.outer(z_grad, self.h)

        if not self.is_input_layer:
            h_grad = np.matmul(self.W.transpose(), z_grad)
            return h_grad
        else:
            return 0

    def update(self, eta=0.01):
        # use stored gradients to do one SGD step
        self.W = self.W * (1 - eta * self.reg_lambda) - eta * self.W_grad
        self.b -= eta * self.b_grad


class MLPDeep:
    def __init__(self, input_dim, output_dim, hidden_dim, reg_lambda=0.0, hidden_layers=1):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reg_lambda = reg_lambda

        self.layers = []

        if hidden_layers >= 1:
            self.layers.append(SigmoidLayer(input_dim, hidden_dim, reg_lambda, is_input_layer=True))
            hidden_layers -= 1  # first hidden layer added
        for i in range(hidden_layers):  # add remaining hidden layers
            self.layers.append(SigmoidLayer(hidden_dim, hidden_dim, reg_lambda))
        # add output layer
        self.layers.append(SigmoidLayer(hidden_dim, output_dim, reg_lambda, is_output_layer=True))

    def forward(self, x):
        h = x  # first input is x
        for layer in self.layers[:-1]:
            h = layer.forward(h)
        y_hat, scores = self.layers[-1].forward(h)
        return y_hat, scores

    def train_step(self, x, y, eta=0.001):
        y_hat, scores = self.forward(x)

        exps = np.exp(scores)
        Z_normalize = np.sum(exps)
        z_grad = exps / Z_normalize  # probabilities
        z_grad[y] -= 1  # subtract one to the correct label

        grad = z_grad  # in layers[-1].backward(), z_grad is used; in all others it's h_grad

        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        for layer in self.layers:
            layer.update(eta)  # acho q posso juntar isto no ciclo de cima

        if np.argmax(scores) != y:
            return 1  # to count mistakes
        else:
            return 0

    def eval(self, x_set, y_set):
        mistakes = 0
        for x, y in zip(x_set, y_set):
            y_hat, _ = self.forward(x)
            if y != y_hat:
                mistakes += 1
        return 1 - (mistakes / len(y_set))


class MLP:
    def __init__(self, input_dim, output_dim, hidden_dim, reg_lambda=0.0):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reg_lambda = reg_lambda

        # self.w_out = np.zeros([output_dim, hidden_dim])
        # self.w_hid = np.zeros([hidden_dim, input_dim])

        glorot_t = sqrt(6) / sqrt(input_dim + hidden_dim)  # glorot initialization
        glorot_t = 0.5
        self.w_hid = (np.random.rand(hidden_dim, input_dim) - 0.5) * 2 * glorot_t
        self.b_hid = np.zeros(hidden_dim)

        glorot_t = sqrt(6) / sqrt(hidden_dim + output_dim) # glorot initialization
        glorot_t=0.5  # glorot seems worse than U[-0.5, 0.5[
        self.w_out = (np.random.rand(output_dim, hidden_dim) - 0.5) * 2 * glorot_t  # U[-t, t[
        self.b_out = np.zeros(output_dim)

        # TODO: improve initialization! right now there's no way to distinguish hard coded biases, so all weights will have to be initialized the same way

    def forward(self, x):
        w_hid = self.w_hid
        b_hid = self.b_hid
        w_out = self.w_out
        b_out = self.b_out

        z_hid = np.matmul(w_hid, x) + b_hid
        h = 1/(1+np.exp(-z_hid))  # values of hidden units, after non-linearization (using sigmoid)
        scores = np.matmul(w_out, h) + b_out

        y_hat = np.argmax(scores)
        return y_hat, scores, h, z_hid

    def train_step(self, x, y, eta=0.001):
        y_hat, scores, h, z_hid = self.forward(x)

        exps = np.exp(scores)
        Z_normalize = np.sum(exps)
        z_grad = exps / Z_normalize  # probabilities
        z_grad[y] -= 1  # subtract one to the correct label
        # x multiplied by the negative probability of each class (+1 in the right label):
        # w_out_grad = np.matmul(np.diag(z_grad), np.outer(np.ones(len(z_grad)), h))  # diag(z_grad) * [x repeated "len(z_grad)" times]
        w_out_grad = np.outer(z_grad, h)

        h_grad = np.matmul(self.w_out.transpose(), z_grad)
        z_hid_grad = np.multiply(h_grad, sigmoid_derivative(z_hid))
        w_hid_grad = np.outer(z_hid_grad, x)

        self.w_out = self.w_out*(1 - eta*self.reg_lambda) - eta * w_out_grad
        self.b_out -= eta * z_grad

        self.w_hid = self.w_hid*(1 - eta*self.reg_lambda) - eta * w_hid_grad
        self.b_hid -= eta * z_hid_grad


        if np.argmax(scores) != y:
            return 1  # to count mistakes
        else:
            return 0

    def eval(self, x_set, y_set):
        mistakes = 0
        for x, y in zip(x_set, y_set):
            y_hat, _, _, _ = self.forward(x)
            if y != y_hat:
                mistakes += 1
        return 1 - (mistakes / len(y_set))


def main():
    x_train, x_dev, x_test, y_train, y_dev, y_test = read_chars('letter.data')

    print(char_to_idx)

    input_dim = len(x_train[0])
    hidden_dim = 100
    output_dim = len(char_to_idx)

    classifier = MLPDeep(input_dim, output_dim, hidden_dim, hidden_layers=3)  # , reg_lambda=5e-5)  # all lamb lowered accuracy on train+dev

    train_acc_list, dev_acc_list = [], []
    train_accuracy = classifier.eval(x_train, y_train)
    dev_accuracy = classifier.eval(x_dev, y_dev)
    print('accuracy before training -> trainset: {} -> devset:{}'.format(train_accuracy, dev_accuracy))

    for epoch in range(20):
        mistakes = 0
        shuf = np.random.permutation(len(x_train))  # the training data is ordered!
        x_train_shuf = x_train[shuf]
        y_train_shuf = y_train[shuf]
        for idx, (x, y) in enumerate(zip(x_train_shuf, y_train_shuf)):
            mistakes += classifier.train_step(x, y)
            if epoch == 0 and idx % 2000 == 0:
                train_accuracy = 1 - (mistakes / (idx + 1))
                dev_accuracy = classifier.eval(x_dev, y_dev)
                print('accuracy in epoch {} after {:>5} samples ->  train: {:<5}, dev: {:<5}'.format(epoch + 1, idx + 1, train_accuracy, dev_accuracy))

        # train_accuracy = 1 - (mistakes / len(y_train))
        train_accuracy = classifier.eval(x_train, y_train)
        dev_accuracy = classifier.eval(x_dev, y_dev)
        print('accuracy after epoch {:>2} -> train: %.2f, dev: %.2f'.format(epoch+1) % (train_accuracy*100, dev_accuracy*100))
        train_acc_list.append(train_accuracy)
        dev_acc_list.append(dev_accuracy)

    test_accuracy = classifier.eval(x_test, y_test)
    print('\nfinal accuracy -> dev: %.2f, test: %.2f' % (dev_acc_list[-1] * 100, test_accuracy * 100))

    plot_accuracies(train_acc_list, dev_acc_list, title="Multi-Layered Perceptron",
                    save_name='fig.png')


if __name__ == '__main__':
    main()