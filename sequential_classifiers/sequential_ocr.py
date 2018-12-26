import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from random import shuffle

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
    x_train = [[]]
    x_dev = [[]]
    x_test = [[]]
    y_train = [[]]
    y_dev = [[]]
    y_test = [[]]

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

            s = [int(sample[i]) for i in range(6, len(sample)-1)]  # -1 ignores \n
            # s.append(1)  # add bias - done later in main()

            fold_id = int(sample[5])
            end_of_word = int(sample[2]) == -1
            if fold_id < 8:
                x_train[-1].append(s)
                y_train[-1].append(char)
                if end_of_word:
                    x_train.append([])
                    y_train.append([])
            elif fold_id == 8:
                x_dev[-1].append(s)
                y_dev[-1].append(char)
                if end_of_word:
                    x_dev.append([])
                    y_dev.append([])
            else:  # fold_id == 9
                x_test[-1].append(s)
                y_test[-1].append(char)
                if end_of_word:
                    x_test.append([])
                    y_test.append([])

    # each set contains an empty list - remove it
    x_train.pop()
    x_dev.pop()
    x_test.pop()
    y_train.pop()
    y_dev.pop()
    y_test.pop()

    print('train size: {}, dev size: {}, test size: {}'.format(len(x_train),len(x_dev),len(x_test)))
    print('unique chars: {}'.format(len(char_to_idx)))

    # idx = 0
    # for char in chars:
    #     char_to_idx[char] = idx
    #     idx_to_char[idx] = char
    #     idx += 1

    return [np.array(i, dtype=np.int) for i in x_train], \
           [np.array(i, dtype=np.int) for i in x_dev], \
           [np.array(i, dtype=np.int) for i in x_test], \
           [np.array(i, dtype=np.int) for i in y_train], \
           [np.array(i, dtype=np.int) for i in y_dev], \
           [np.array(i, dtype=np.int) for i in y_test]


class StructuredPerceptron:
    def __init__(self, input_dim, output_dim):
        self.output_dim = output_dim
        self.input_dim = input_dim
        # self.w_unigram = np.zeros([input_dim*output_dim])  # 1 vector for all w's
        self.w_unigram = np.zeros([output_dim, input_dim])  # matrix with 1 vector per class
        # matrix with transition probs (the only bigram feat. here)
        self.w_bigram = np.zeros([output_dim, output_dim])
        self.w_start = np.zeros(output_dim)  # p(y_i|start)
        self.w_stop = np.zeros(output_dim)  # p(stop|y_i)

    def forward(self, x):
        # compute score of first char
        # unigram
        unig_scores = np.dot(self.w_unigram, x[0])
        # bigram
        bigr_scores_start = self.w_start  # score of departing from <start> and arriving at each state
        # score so far, for each possible y_0
        scores = unig_scores + bigr_scores_start

        # to keep track of which y_i's maximize the score at each step
        backtrack = np.zeros([len(x)-1, self.output_dim])
        for i in range(1, len(x)):
            prev_scores = scores
            unig_scores = np.dot(self.w_unigram, x[i])
            bigr_scores = self.w_bigram  # departing from each state, arriving at each state

            scores = bigr_scores + prev_scores[np.newaxis].transpose()  # add prev_scores along dim=1, instead of dim=0

            backtrack[i-1] = np.argmax(scores, axis=0)  # get idx of most likely y_{i-1}, for each y_i
            scores = np.max(scores, axis=0)  # get score of each y_i, using the most likely y_{i-1}
            scores += unig_scores

        scores += self.w_stop  # add stop scores
        last_y = np.argmax(scores)  # most likely label for last position
        # backtrack previous states to reach last_y
        y_hat = [last_y]
        for i in range(len(backtrack) - 1, -1, -1):  # traverse list from last to first
            last_y = backtrack[i, int(last_y)]
            y_hat.insert(0, last_y)
        return y_hat

    def train_step(self, x, y):
        mistakes = 0
        y_hat = self.forward(x)

        if y_hat[0] != y[0]:
            mistakes += 1
            self.w_start[y[0]] += 1  # Assuming prediction was one-hot of the chosen transition
            self.w_start[int(y_hat[0])] -= 1
            self.w_unigram[y[0]] += x[0]
            self.w_unigram[int(y_hat[0])] -= x[0]

        # for y_hat_i, y_i, x_i in zip(y_hat, y, x):
        for i in range(1, len(x)):
            y_hat_i = y_hat[i]
            y_i = y[i]
            x_i = x[i]
            if y_hat_i != y_i:
                mistakes += 1

                self.w_bigram[int(y_hat[i-1]), int(y_hat_i)] -= 1
                self.w_bigram[int(y[i-1]), int(y[i])] += 1

                self.w_unigram[y_i] += x_i
                self.w_unigram[int(y_hat_i)] -= x_i

        # update stop probabilities
        if y_hat[-1] != y[-1]:
            # unigrams were already updated
            self.w_stop[y[-1]] += 1  # Assuming prediction was one-hot of the chosen transition
            self.w_stop[int(y_hat[-1])] -= 1

        return mistakes

    def eval(self, x_set, y_set):
        mistakes = 0
        total = 0
        for x, y in zip(x_set, y_set):
            y_hat = self.forward(x)
            total += len(y)
            for y_i, y_hat_i in zip(y, y_hat):
                if y_i != y_hat_i:
                    mistakes += 1
        return 1 - (mistakes / total)


class CRF:
    def __init__(self, input_dim, output_dim):
        self.output_dim = output_dim
        self.input_dim = input_dim
        # self.w_unigram = np.zeros([input_dim*output_dim])  # 1 vector for all w's
        self.w_unigram = np.zeros([output_dim, input_dim])  # matrix with 1 vector per class
        # matrix with transition probs (the only bigram feat. here)
        self.w_bigram = np.zeros([output_dim, output_dim])
        self.w_start = np.zeros(output_dim)  # p(y_i|start)
        self.w_stop = np.zeros(output_dim)  # p(stop|y_i)

    def log_sum_matrix(self, m):
        """returns a log-sum over a whole matrix"""
        max_ = np.max(m, axis=0)
        return max_ + np.log(sum([np.exp(l - max_) for l in m]))

    def forward(self, x):
        # compute score of first char
        # unigram
        unig_scores = np.dot(self.w_unigram, x[0])
        # bigram
        bigr_scores_start = self.w_start  # score of departing from <start> and arriving at each state
        # score so far, for each possible y_0
        scores = unig_scores + bigr_scores_start

        # to keep track of which y_i's maximize the score at each step
        backtrack = np.zeros([len(x)-1, self.output_dim])
        for i in range(1, len(x)):
            prev_scores = scores
            unig_scores = np.dot(self.w_unigram, x[i])
            bigr_scores = self.w_bigram  # departing from each state, arriving at each state

            scores = bigr_scores + prev_scores[np.newaxis].transpose()  # add prev_scores along dim=1, instead of dim=0

            backtrack[i - 1] = np.argmax(scores, axis=0)  # get idx of most likely y_{i-1}, for each y_i
            scores = np.max(scores, axis=0)  # get score of each y_i, using the most likely y_{i-1}
            scores += unig_scores

        scores += self.w_stop  # add stop scores
        last_y = np.argmax(scores)  # most likely label for last position
        # backtrack previous states to reach last_y
        y_hat = [last_y]
        for i in range(len(backtrack) - 1, -1, -1):  # traverse list from last to first
            last_y = backtrack[i, int(last_y)]
            y_hat.insert(0, last_y)
        return y_hat

    def forward_alpha(self, x, i, for_unigrams=True):  # forward pass of forward-backward
        # if not "for_unigrams", compute for bigrams

        if i == 0 and for_unigrams:
            return self.w_start

        unig_scores = np.dot(self.w_unigram, x[0])
        # bigram
        bigr_scores_start = self.w_start  # score of departing from <start> and arriving at each state
        # score so far, for each possible y_0
        scores = unig_scores + bigr_scores_start

        for j in range(1, i+1):
            prev_scores = scores
            bigr_scores = self.w_bigram  # departing from each state, arriving at each state

            scores = bigr_scores + prev_scores[np.newaxis].transpose()  # add prev_scores along dim=1, instead of dim=0

            scores = self.log_sum_matrix(scores)  # get score of each y_i, for the sum of all y_{i-1}

            if for_unigrams and j == i:  # if last iteration and we want the alpha for computing unigram marginals
                return scores

            unig_scores = np.dot(self.w_unigram, x[j])
            scores += unig_scores
        return scores

    def backward_beta(self, x, i, for_unigrams=True):
        # if not "for_unigrams", compute for bigrams

        if i == len(x)-1 and for_unigrams:
            return self.w_stop

        unig_scores = np.dot(self.w_unigram, x[-1])
        # bigram
        bigr_scores_stop = self.w_stop  # score of departing from <start> and arriving at each state
        # score so far, for each possible y_0
        scores = unig_scores + bigr_scores_stop

        for j in range(len(x)-2,i-1,-1):
            prev_scores = scores
            bigr_scores = self.w_bigram  # departing from each state, arriving at each state
            scores = bigr_scores + prev_scores

            scores = self.log_sum_matrix(scores.T)  # get score of each y_{i-1}, for the sum of all y_i

            if for_unigrams and j == i:  # if last iteration and we want the beta for computing unigram marginals
                return scores

            unig_scores = np.dot(self.w_unigram, x[j])
            scores += unig_scores
        return scores

    def scores_to_probs(self, s):
        s_exp = np.exp(s)
        return s_exp / np.sum(s_exp)

    def train_step(self, x, y, eta=0.001):
        mistakes = 0

        # update w_start
        y_start_scores = self.backward_beta(x, 0, for_unigrams=False) + self.w_start
        y_start_probs = self.scores_to_probs(y_start_scores)
        self.w_start -= eta*y_start_probs
        self.w_start[y[0]] += eta
        if np.argmax(y_start_probs) != y[0]:
            mistakes += 1

        # update w_unigram for i=0
        unig_scores = np.dot(self.w_unigram, x[0])
        unig_scores += self.forward_alpha(x, 0, for_unigrams=True) + self.backward_beta(x, 0, for_unigrams=True)
        unig_probs = self.scores_to_probs(unig_scores)
        weighted_x = np.outer(unig_probs, x[0])
        self.w_unigram -= eta * weighted_x  # subtract P(y|x)*x for each y
        self.w_unigram[y[0]] += eta * x[0]  # add x for correct y

        for i in range(1, len(x)):
            # update bigram weights for i-1 -> i
            bigr_scores = self.forward_alpha(x, i-1, for_unigrams=False)[np.newaxis].transpose() + self.w_bigram
            bigr_scores += self.backward_beta(x, i, for_unigrams=False)
            bigr_probs = self.scores_to_probs(bigr_scores)
            self.w_bigram -= eta * bigr_probs
            self.w_bigram[y[i-1], y[i]] += eta

            # update unigram weights for i
            unig_scores = np.dot(self.w_unigram, x[i])
            unig_scores += self.forward_alpha(x, i, for_unigrams=True) + self.backward_beta(x, i, for_unigrams=True)
            unig_probs = self.scores_to_probs(unig_scores)
            weighted_x = np.outer(unig_probs, x[i])
            self.w_unigram -= eta * weighted_x  # subtract P(y|x)*x for each y
            self.w_unigram[y[i]] += eta * x[i]  # add x for correct y

        # update w_stop
        y_stop_scores = self.forward_alpha(x, len(x)-1, for_unigrams=False) + self.w_stop
        y_stop_probs = self.scores_to_probs(y_stop_scores)
        self.w_stop -= eta * y_stop_probs
        self.w_stop[y[-1]] += eta

        return mistakes

    def eval(self, x_set, y_set):
        mistakes = 0
        total = 0
        for x, y in zip(x_set, y_set):
            y_hat = self.forward(x)
            total += len(y)
            for y_i, y_hat_i in zip(y, y_hat):
                if y_i != y_hat_i:
                    mistakes += 1
        return 1 - (mistakes / total)


def get_pairwise_keep_original_np(x_data):
    x_transformed = np.zeros([len(x_data), int(len(x_data[0])*(len(x_data[0])+1)/2)])  # -1)/2)]) # triangular number (1+2+3+...+[len(sample)-1])
    for sample_idx, sample in enumerate(x_data):
        feature_idx = 0
        for i in range(len(sample) - 1):
            for j in range(i, len(sample)):  # + 1, len(sample)):
                x_transformed[sample_idx, feature_idx] = sample[i]*sample[j]
                feature_idx += 1
    return x_transformed


def main():
    x_train, x_dev, x_test, y_train, y_dev, y_test = read_chars('letter.data')
    print('finished reading data')

    use_pairwise_features = True
    if use_pairwise_features:
        import time
        t = time.clock()
        print('begin pairwise')
        x_train = [get_pairwise_keep_original_np(x_word) for x_word in x_train]
        x_dev = [get_pairwise_keep_original_np(x_word) for x_word in x_dev]
        x_test = [get_pairwise_keep_original_np(x_word) for x_word in x_test]
        print('end in {} seconds'.format(time.clock() - t))

    add_bias = True
    if add_bias:
        x_train = [np.hstack((x, np.ones([len(x), 1]))) for x in x_train]
        x_dev = [np.hstack((x, np.ones([len(x), 1]))) for x in x_dev]
        x_test = [np.hstack((x, np.ones([len(x), 1]))) for x in x_test]

    input_dim = len(x_train[0][0])
    output_dim = len(char_to_idx)
    print("in {}, out {}".format(input_dim, output_dim))

    # classifier = StructuredPerceptron(input_dim, output_dim)
    classifier = CRF(input_dim, output_dim)
    # test=[classifier.forward_alpha(x_train[0], i, False) + classifier.backward_beta(x_train[0],i,False) for i in range(len(x_train[0]))]

    train_acc_list, dev_acc_list = [], []
    for epoch in range(20):
        mistakes = 0
        total = 0

        xy = list(zip(x_train, y_train))
        shuffle(xy)
        x_train_shuf, y_train_shuf = zip(*xy)

        for idx, (x, y) in enumerate(zip(x_train_shuf, y_train_shuf)):
            mistakes += classifier.train_step(x, y)
            total += len(x)
            if epoch == 0 and idx % 200 == 0:
                train_accuracy = 1 - (mistakes / total)
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

    plot_accuracies(train_acc_list, dev_acc_list, title="Structured Perceptron using pairwise pixel multiplications",
                    save_name='fig.png')


if __name__ == '__main__':
    main()
