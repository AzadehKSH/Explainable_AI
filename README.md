# Explainable_AI
# Every week I recieve the challang from the Prof.Martin Becker and its my answers to it.

<h1> Questions week 1 </h1>
<h2> 1-Specifying ML Tasks </h2>
<p> For the following problem descriptions, explain whether the problem is a supervised or an
unsupervised problem and whether it is classification, regression, or clustering.
  </p>

 1.  We have a set of images, with descriptions of what object is depicted in the images.Given new images, we want to decide what object is on it?!
<p> This is a supervised problem of image classification. The goal is to train a model using the set of images with their corresponding labels, so that it can learn to identify and classify objects in new images. The input data is the set of images, and the output is the predicted label of the object depicted in the image.
  </p>
 2. Given part of a text, we want to predict the next word?!

  <p> 
This is a supervised problem of language modeling. The goal is to train a model using a large corpus of text data, so that it can learn to predict the next word given a sequence of previous words. The input data is the sequence of words, and the output is the predicted probability distribution over the next word in the vocabulary. This problem can be solved using techniques such as recurrent neural networks (RNNs) and transformer-based models, which are specifically designed for language modeling tasks.
  </P>
  3. We want to predict the income of graduates of the university. Only few people answer our question, but we know via social networks who they are friends with ?!
  <p> This is a supervised problem of social network-based inference. The goal is to predict the income of graduates using the available data, which includes the income information of a few graduates, as well as their social network connections. This problem can be solved using techniques such as network-based regularization, which can leverage the information contained in the social network to improve the accuracy of the predictions. Specifically, the social network connections can be used to create a graph structure, and then the graph regularization can be added to the loss function of the model to encourage the predicted income of connected graduates to be similar. However, it is important to note that the accuracy of the predictions may be limited by the sparsity and representativeness of the available social network data.
  </p>
 4.  We want to identify genres of books given a large collection of them?!
  <p> This is an unsupervised problem of text clustering. The goal is to group books into different genres based on their textual content. The input data is the collection of books, and the output is the set of clusters, each corresponding to a different genre. This problem can be solved using techniques such as k-means clustering, hierarchical clustering, or spectral clustering, which are designed to partition a dataset into a set of clusters based on their similarity or distance in a high-dimensional feature space. In this case, the features may be the bag-of-words representations of the book texts, or more advanced features such as word embeddings or topic models. It is important to note that the quality of the clustering results will depend on the choice of the clustering algorithm, the choice of features, and the subjective definition of what constitutes a genre.
  </p>

<h2>2 Generative vs Discriminative Models</h2>
Explain the difference between a generative and a discriminative model. Give an example for each of them?!
<p>
A generative model learns the joint probability distribution of the input features and the output labels, and can be used to generate new examples by sampling from the learned distribution. A discriminative model learns the conditional probability distribution of the output labels given the input features, and can be used to make predictions for new examples.

An example of a generative model is a Naive Bayes classifier, which models the joint probability distribution of the input features and the output labels using the Bayes rule and the assumption of conditional independence between the features. Given a set of training examples, a Naive Bayes classifier can estimate the parameters of the model and use them to generate new examples by sampling from the learned distribution.

An example of a discriminative model is a logistic regression classifier, which models the conditional probability distribution of the output labels given the input features using a logistic function. Given a set of training examples, a logistic regression classifier can estimate the parameters of the model and use them to make predictions for new examples by computing the conditional probabilities of the output labels given the input features and choosing the label with the highest probability.

The main difference between a generative and a discriminative model is the type of probability distribution that they learn. A generative model learns the joint probability distribution of the input features and the output labels, which can be used to generate new examples or perform other tasks such as data augmentation or missing data imputation. A discriminative model learns the conditional probability distribution of the output labels given the input features, which is directly relevant for the task of making predictions.
</p>

<h2>3 Definitions</h2>
1. Mean squared error (MSE):
MSE is a common loss function used for regression problems. It measures the average squared difference between the predicted values and the actual values of the target variable. The formula for MSE is:
MSE = (1/n) * Σ(yi - ŷi)²

where n is the number of examples in the dataset, yi is the actual value of the target variable for example i, and ŷi is the predicted value of the target variable for example i.

2. Cross-entropy loss:
Cross-entropy loss is a common loss function used for classification problems. It measures the difference between the predicted probability distribution and the actual probability distribution of the target variable. The formula for binary cross-entropy loss is:
CE = - (1/n) * Σ[yi * log(ŷi) + (1-yi) * log(1-ŷi)]

where n is the number of examples in the dataset, yi is the actual value of the target variable (0 or 1) for example i, and ŷi is the predicted probability of the target variable being 1 for example i.

The formula for categorical cross-entropy loss is similar, but it sums over all possible classes k instead of just two:

CE = - (1/n) * ΣΣ[yi,k * log(ŷi,k)]

where n is the number of examples in the dataset, yi,k is the actual probability of example i belonging to class k, and ŷi,k is the predicted probability of example i belonging to class k.

3. Sigmoid function:
The sigmoid function is a common activation function used in neural networks. It maps any real-valued input to a value between 0 and 1, which can be interpreted as a probability. The formula for the sigmoid function is:
σ(x) = 1 / (1 + exp(-x))

where x is the input to the function. The sigmoid function has the property that σ(-∞) = 0 and σ(∞) = 1, and its derivative is given by:

dσ(x) / dx = σ(x) * (1 - σ(x))

which is useful for backpropagation in neural networks.

4. Mean absolute error (MAE) is a common loss function used for regression problems. It measures the average absolute difference between the predicted values and the actual values of the target variable. The formula for MAE is:

MAE = (1/n) * Σ|yi - ŷi|

where n is the number of examples in the dataset, yi is the actual value of the target variable for example i, and ŷi is the predicted value of the target variable for example i.

The MAE is a good measure of the average absolute deviation of the predictions from the true values, and is less sensitive to outliers compared to the mean squared error (MSE). However, it does not penalize large errors as strongly as the MSE, and may not be as effective in cases where small errors are more important than large errors.


