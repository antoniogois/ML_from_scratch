# Linear Classifiers

This code covers two linear classification algorithms and their application to identify a digit in an image with binary pixels, either by using the raw features or pair-wise pixel multiplications as a manually engineered feature.

* [Logistic Regression](#logistic-regression)
* [Perceptron](#perceptron)

## Logistic Regression

TODO: include link to proof for multiclass lr that I found in princeton; mention interpretation of multi-class logits (show sum doesn't change probabilities; setting a class c to zero gives logits of all others with respect to c)

The Logistic Regression has a simple derivation that I will show here. Surprisingly, it was not easy to find a good derivation of the multinomial case (the best explanation I could find was in section 6.2.3 of [this link](https://data.princeton.edu/wws509/notes/c6s2)).

We begin by defining the Logit function, also known as log-odds. The odds of an event are the ratio between the probability of an event happening and not happening (i.e. \frac{p}{1-p} TODO: fix LaTex)). The Logit function simply maps a probability to the logarithm of its odds.

Using this definition of a Logit function, the Logistic function is simply the inverse function of a Logit: mapping log-odds (in the domain ]-inf,+inf\[) to probabilities (in the domain [0,1], although it can only reach 0 with an input of -inf − see SparseMax for a neat way to overcome this). Using the Logistic function, what the Logistic regression does is simply to assume that the log-odds can be predicted using a linear combination of its inputs. Then, the predicted log-odds are converted into a probability using the logistic function and voilá − we have and estimation of a probability (if it holds true that the log-odds are a linear combination of the inputs). This is a convenient assumption, since we can receive any linear value and map it to a valid probability.


## Perceptron
