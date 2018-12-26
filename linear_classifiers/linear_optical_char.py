import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
np.random.seed(0)

char_to_idx = {}
idx_to_char = {}


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


def visualize_char(x, y):
    print('letter: {}'.format(y))
    for i in range(0, 8 * 16, 8):
        print(''.join(['*' if pixel == 1 else ' ' for pixel in x[i:i + 7]]))


class Perceptron():
    def __init__(self, input_dim, output_dim):
        self.output_dim = output_dim
        self.input_dim = input_dim
        # self.w = np.zeros([input_dim*output_dim])  # 1 vector for all w's
        self.w = np.zeros([output_dim, input_dim])  # matrix with 1 vector per class

    def forward(self, x):
        out_d = self.output_dim
        in_d = self.input_dim
        w = self.w

        assert in_d == len(x)
        # phis = np.zeros([out_d, in_d*out_d])  # store the phi (features) of each label y
        # for i in range(out_d):
        #     one_hot = np.zeros([out_d])
        #     one_hot[i] = 1
        #     phi = np.kron(one_hot, x)
        #     phis[i] = phi

        scores = np.dot(w, x)
        y_hat = np.argmax(scores)
        return y_hat

    def train_step(self, x, y):
        y_hat = self.forward(x)
        if y_hat != y:
            self.w[y] += x
            self.w[y_hat] -= x
            return 1  # to count mistakes
        else:
            return 0

    def eval(self, x_set, y_set):
        mistakes = 0
        for x, y in zip(x_set, y_set):
            y_hat = self.forward(x)
            if y != y_hat:
                mistakes += 1
        return 1 - (mistakes / len(y_set))


class LogisticRegr():
    def __init__(self, input_dim, output_dim, reg_lambda=0.):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.reg_lambda = reg_lambda
        self.w = np.zeros([output_dim, input_dim])  # matrix with 1 vector per class

    def forward(self, x):
        w = self.w

        scores = np.dot(w, x)
        y_hat = np.argmax(scores)
        return y_hat

    def train_step(self, x, y, eta=0.001):
        scores = np.dot(self.w, x)
        exps = np.exp(scores)
        z = np.sum(exps)
        probs = exps / z
        # x multiplied by the probability of each class:
        # weighted_x_ = np.matmul(np.diag(probs), np.outer(np.ones(len(probs)), x))  # diag(probs) * [x repeated "len(probs)" times]
        weighted_x = np.outer(probs, x)

        # assert np.all(weighted_x == weighted_x_)

        if self.reg_lambda > 0:  # regularize
            self.w -= self.reg_lambda*self.w
        self.w -= eta*weighted_x  # subtract P(y|x)*x for each y
        self.w[y] += eta*x  # add x for correct y

        if np.argmax(scores) != y:
            return 1  # to count mistakes
        else:
            return 0

    def eval(self, x_set, y_set):
        mistakes = 0
        for x, y in zip(x_set, y_set):
            y_hat = self.forward(x)
            if y != y_hat:
                mistakes += 1
        return 1 - (mistakes / len(y_set))


def get_pairwise(x_data):
    x_transformed = []
    for idx, sample in enumerate(x_data):
        sample_transformed = []  # np.zeros(int( (len(sample)*(len(sample)+1)) / 2 ))  # triangular number (1+2+3+...+len(sample))  # NOQA
        iterator = 0
        for i in range(len(sample)-1):
            for j in range(i+1, len(sample)):
                # sample_transformed[iterator] = i*j
                sample_transformed.append(sample[i]*sample[j])
                iterator += 1
        x_transformed.append(sample_transformed)
    return x_transformed


def get_pairwise_np(x_data):
    x_transformed = np.zeros([len(x_data), int(len(x_data[0])*(len(x_data[0])-1)/2)]) # triangular number (1+2+3+...+[len(sample)-1])
    for sample_idx, sample in enumerate(x_data):
        feature_idx = 0
        for i in range(len(sample) - 1):
            for j in range(i + 1, len(sample)):
                x_transformed[sample_idx, feature_idx] = sample[i]*sample[j]
                feature_idx += 1
    return x_transformed


def get_pairwise_keep_original_np(x_data):
    x_transformed = np.zeros([len(x_data), int(len(x_data[0])*(len(x_data[0]) +1)/2)])  # -1)/2)]) # triangular number (1+2+3+...+[len(sample)-1])
    for sample_idx, sample in enumerate(x_data):
        feature_idx = 0
        for i in range(len(sample) - 1):
            for j in range(i , len(sample)):  # + 1, len(sample)):
                x_transformed[sample_idx, feature_idx] = sample[i]*sample[j]
                feature_idx += 1
    return x_transformed


def main():
    x_train, x_dev, x_test, y_train, y_dev, y_test = read_chars('letter.data')

    # x_train = get_pairwise(x_train)
    # x_dev = get_pairwise(x_dev)

    # idx_vis = 27
    # visualize_char(x_train[idx_vis], idx_to_char[y_train[idx_vis]])
    print(char_to_idx)
    # print(idx_to_char)

    use_pairwise_features = True
    if use_pairwise_features:
        import time
        t=time.clock()
        print('begin pairwise')
        x_train = get_pairwise_np(x_train)
        x_dev = get_pairwise_np(x_dev)
        x_test = get_pairwise_np(x_test)
        print('end in {} seconds'.format(time.clock()-t))

    add_bias = True
    if add_bias:
        x_train = np.hstack((x_train, np.ones([len(x_train), 1])))
        x_dev = np.hstack((x_dev, np.ones([len(x_dev), 1])))
        x_test = np.hstack((x_test, np.ones([len(x_test), 1])))

    input_dim = len(x_train[0])
    output_dim = len(char_to_idx)

    print("in {}, out {}".format(input_dim,output_dim))

    # classifier = Perceptron(input_dim, output_dim)
    classifier = LogisticRegr(input_dim, output_dim, reg_lambda=1e-6)

    # classifier.forward(x_train[0])
    # classifier.train_step(x_train[0], char_to_idx[y_train[0]])

    train_acc_list, dev_acc_list = [], []
    train_accuracy = classifier.eval(x_train, y_train)
    dev_accuracy = classifier.eval(x_dev, y_dev)
    print('accuracy before training -> trainset: {} -> devset:{}'.format(train_accuracy, dev_accuracy))
    train_acc_list.append(train_accuracy)
    dev_acc_list.append(dev_accuracy)

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
    print('\nfinal accuracy -> dev: %.2f, test: %.2f' % (dev_acc_list[-1]*100, test_accuracy*100))

    plot_accuracies(train_acc_list, dev_acc_list, title="perceptron using pairwise pixel multiplications", save_name ='fig.png')


if __name__ == '__main__':
    main()


