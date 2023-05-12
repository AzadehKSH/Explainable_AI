# Explainable_AI
Every week I recieve the challang from the Prof.Martin Becker and its my answers to it.

# Table of contents
1. [02_exercises](#02_exercises)
  1. [Specifying ML Tasks](#02_exercises) 






<h1> Questions 02_exercises file </h1> <a name="02_exercises"></a>
<h2> 1-Specifying ML Tasks </h2> <a name="Specifying_ML_Tasks"></a>
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

<h2> 4 Hyperparameters </h2>
What is the difference between hyperparameters and parameters. Give an example of a hyperparameter for a logistic regression model with L2 regularization?!
In machine learning, a model is typically defined by a set of parameters and hyperparameters, which are used to control the behavior of the model during training and inference.

Parameters are the variables that the model learns from the training data to make predictions. They are the weights that are updated during the training process to minimize the loss function. In a logistic regression model, the parameters are the coefficients of the input features that are used to compute the predicted probability of the target class.

Hyperparameters, on the other hand, are the variables that control the behavior of the learning algorithm itself, and are set before the training process begins. They determine the settings of the algorithm that cannot be learned from the data itself. Examples of hyperparameters include the learning rate, regularization strength, number of hidden layers in a neural network, etc.

An example of a hyperparameter for a logistic regression model with L2 regularization is the regularization strength parameter, often denoted as λ (lambda). L2 regularization, also known as ridge regression, adds a penalty term to the loss function to prevent overfitting of the model. The regularization strength determines how much weight is given to this penalty term, relative to the original loss function. A higher value of λ will result in a more heavily regularized model, which may be less prone to overfitting but also less flexible in fitting the training data. The value of λ is typically determined using a validation set or cross-validation to select the hyperparameter that results in the best performance on a held-out set.

<h2> 5 Model Evaluation </h2>
To calculate precision, recall, F1-score, and accuracy from a confusion matrix, we first need to define the following terms:

True Positive (TP): The number of examples that are actually positive and are correctly predicted as positive.
False Positive (FP): The number of examples that are actually negative but are incorrectly predicted as positive.
False Negative (FN): The number of examples that are actually positive but are incorrectly predicted as negative.
True Negative (TN): The number of examples that are actually negative and are correctly predicted as negative.

|               | Positive          | Negative  |
| ------------- |:-------------:| -----:|
| Actual Positive     | 8 | 2 |
| Actual Negative     | 16     |   974 |
|

precision = TP / (TP + FP)

precision = 8 / (8 + 16) = 0.333

recall = TP / (TP + FN)

recall = 8 / (8 + 2) = 0.8

F1-score = 2 * (precision * recall) / (precision + recall)

F1-score = 2 * (0.333 * 0.8) / (0.333 + 0.8) = 0.470

accuracy = (TP + TN) / (TP + TN + FP + FN)

accuracy = (8 + 974) / (8 + 2 + 16 + 974) = 0.973


2. Given the output ˆy of a classifier and the ground truth labels y, calculate and draw the ROC curve?!
yˆ = [0.5, 0.35, 0.8, 0.1, 0.2]T
y = [1, 0, 1, 0, 1]T

```python
import numpy as np
import matplotlib.pyplot as plt

y_hat = np.array([0.5, 0.35, 0.8, 0.1, 0.2])
y = np.array([1, 0, 1, 0, 1])

# sort predictions by confidence
sorted_HL = np.argsort(y_hat)[::-1]
y_hat_sorted = y_hat[sorted_HL]
y_sorted = y[sorted_HL]

# calculate TPR and FPR for different thresholds
thresh= np.append(y_hat_sorted, 1)
tpr = np.zeros(len(thresh))
fpr = np.zeros(len(thresh))
for i, th in enumerate(thresh):
    y_pred = (y_hat_sorted >= th).astype(int)
    tp = np.sum((y_pred == 1) & (y_sorted == 1))
    fn = np.sum((y_pred == 0) & (y_sorted == 1))
    fp = np.sum((y_pred == 1) & (y_sorted == 0))
    tn = np.sum((y_pred == 0) & (y_sorted == 0))
    tpr[i] = tp / (tp + fn)
    fpr[i] = fp / (fp + tn)

# plot ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC Curve')
plt.show()
```

<p align="center">
  <img src="https://github.com/AzadehKSH/Explainable_AI/blob/main/roc.JPG" width="350" title="ROC curve">
</p>

3. Explain when using a ROC curve make sense compared to just using accuracy etc?!
ROC curves are typically used in binary classification problems where the positive class is rare or the cost of false positives and false negatives are significantly different. In such cases, accuracy alone may not be a good metric to evaluate the performance of a classifier.

For example, consider a medical diagnostic test where the goal is to identify patients who have a rare disease. In this case, the positive class (patients with the disease) is rare, and misclassifying even a small fraction of positive cases as negative can have serious consequences. On the other hand, misclassifying negative cases as positive is not as critical. In such scenarios, a classifier's performance can be better evaluated using a ROC curve, which provides a more comprehensive view of the classifier's trade-offs between true positives and false positives.

The ROC curve is also useful when comparing the performance of different classifiers, especially when they have different decision thresholds. In such cases, the ROC curve allows us to compare the classifiers' performances across all possible decision thresholds.

In summary, ROC curves are useful when the cost of false positives and false negatives is significantly different, the positive class is rare, or when comparing the performance of different classifiers. In such cases, using accuracy alone may not provide a comprehensive view of the classifier's performance, and the ROC curve can provide more insight into the classifier's behavior across different decision thresholds.

4. What is a problem of the ROC curve in heavily imbalanced classification problems?!
A problem with using the ROC curve in heavily imbalanced classification problems is that it can be misleading. When the dataset is heavily imbalanced, meaning that one class is much rarer than the other, the ROC curve may not provide an accurate representation of the classifier's performance. This is because the ROC curve measures the classifier's trade-offs between true positives and false positives across all possible decision thresholds, but in imbalanced datasets, the classifier may perform well in terms of false positives but poorly in terms of true positives.

For example, consider a medical diagnostic test where the goal is to identify patients who have a rare disease. If the positive class (patients with the disease) is very rare, the ROC curve may show good performance because the classifier can achieve low false positive rates, but this may be misleading because the true positive rate may still be very low. In such cases, it may be more appropriate to use evaluation metrics that focus on the performance of the positive class, such as precision, recall, or the F1-score.

In summary, while the ROC curve is a useful tool for evaluating binary classifiers, it may not provide an accurate representation of the classifier's performance in heavily imbalanced datasets. In such cases, it may be more appropriate to use other evaluation metrics that focus on the performance of the positive class.

5. Explain the tradeoff between k-fold crossvalidation and splitting data into a single train and test part?!
The tradeoff between k-fold cross-validation and splitting data into a single train and test part lies in the balance between the bias and variance of the performance estimate of the model.

On the one hand, splitting the data into a single train and test part is a simple and fast method for evaluating the performance of a model. However, this approach may result in a high variance in the performance estimate, as the performance can be highly dependent on the random selection of the train and test sets. This means that the performance estimate may not be representative of the true performance of the model on unseen data. Moreover, this method uses only a single split of the data, which may lead to overfitting of the model to the specific train-test split.

On the other hand, k-fold cross-validation addresses the high variance issue by averaging the performance over multiple train-test splits. This method divides the data into k-folds, trains the model on k-1 folds, and tests it on the remaining fold. This process is repeated k times, with each fold serving as the test set once. The performance is then averaged across all the k folds. K-fold cross-validation provides a more reliable estimate of the performance of the model as it evaluates the model on different subsets of the data, reducing the impact of the random selection of the train and test sets. However, this method is computationally more expensive than splitting the data into a single train and test part, as it requires training the model k times.

In summary, the choice between k-fold cross-validation and splitting data into a single train and test part depends on the balance between the bias and variance of the performance estimate of the model. If a quick evaluation is required, or if the dataset is very large, splitting the data into a single train and test part may be appropriate. However, if a more accurate estimate of the model performance is required, k-fold cross-validation is a better choice, despite its higher computational cost.

<h2> 6 Information Leakage </h2>

1. Define the term information leakage with respect to model training?!

Information leakage in model training refers to a situation where information from the test set (or any other data that the model should not have access to during training) is unintentionally used in the training process, leading to overly optimistic performance estimates. In other words, the model may learn to exploit patterns in the test data that are not generalizable to new, unseen data.

Information leakage can occur in various ways, such as when features that are not present in the training set are used to train the model, or when the test set is used to tune hyperparameters, select features, or preprocess the data. Leakage can also occur when the model is evaluated multiple times during the training process, and the evaluation metrics are used to adjust the model's parameters.

Information leakage can be detrimental to the performance of the model, as it may lead to overfitting and poor generalization to new data. To avoid information leakage, it is important to carefully separate the training, validation, and test data and ensure that the model does not have access to the test set during training. It is also important to follow best practices for feature selection, hyperparameter tuning, and model evaluation to avoid leakage and ensure that the performance estimates are reliable and generalizable.

2. Decide, whether the following scenarios have information leakage. Explain your decicsion?!
(a) You want to train an object detector on a video of football. To later on estimate
the performance of the detector, you split the data randomly into training and
test parts. You only use the training data to fit the model.
(b) You want to classify gene data. Since each instance has 10000 features, you first
want to select the top k features to avoid overfitting. You first run a statistical
test to select these features, then you split the data into training and test parts.
(c) You want to classify nodes in a citation graph with respect to the contents of
the documents. You augment the nodes with word frequencies, which are used as
features for the classifier. Then you split the data.

(a) This scenario does not have information leakage. The data has been split randomly into training and test sets, and only the training set has been used to fit the model. As long as the test set is not used in any way during the training process, there is no risk of information leakage.

(b) This scenario may have information leakage. If the statistical test used to select the top k features is based on the entire dataset (i.e., both the training and test sets), then information from the test set may have leaked into the feature selection process, leading to overly optimistic performance estimates. To avoid information leakage, the feature selection should be performed on the training set only, and the test set should be kept completely separate.

(c) This scenario does not have information leakage, as long as the word frequencies are computed based only on the training set and not on the entire dataset. The data has been split into training and test sets, and the features have been generated based on the training set only. As long as the test set is not used in any way during the feature engineering process, there is no risk of information leakage.


<h2> 7 KNN Classifier</h2>
For point 3:  [3, 2, 1, 10, 8]. The three closest labeled points are [1, 1, 2],  predict the majority class, which is 1.
For point 10:  [10, 9, 6, 3, 1]. The three closest labeled points are [1, 1, 2], predict the majority class, which is 1.
For point 6:  [6, 5, 2, 7, 5]. The three closest labeled points are [2, 2, 1],  predict the majority class, which is 2.
then [3, 10, 6] = [1, 1, 2].

<h2> 8 Over- and Underfitting </h2>

1. Define overfitting?!

Overfitting is a common problem in machine learning where a model is too complex and fits the training data too closely, to the point that it captures the noise in the training data rather than the underlying pattern. As a result, the model has poor generalization performance and fails to accurately predict outcomes on new, unseen data. In other words, an overfit model is too specialized to the training data, and thus, it does not capture the true relationship between the input features and the target variable. Overfitting occurs when the model has too many parameters or when it is trained for too many iterations.

2. Define underfitting?!

Underfitting is the opposite of overfitting and occurs when a machine learning model is too simple to capture the underlying pattern of the data. In other words, the model is not able to capture the complexity of the data and the relationship between the input features and the target variable. An underfit model performs poorly on both the training data and new, unseen data because it cannot learn the underlying patterns and relationships in the data. This can happen if the model is not complex enough or if it is not trained for enough iterations.

3. Explain the role that regularization plays with respect to over- and underfitting. Give Two xamples of regularization techniques?!

Regularization is a technique used to prevent overfitting by adding a penalty term to the model's objective function that discourages overly complex models. This penalty term aims to control the model's complexity, forcing it to focus on the most relevant features and avoid learning noise and irrelevant details. Regularization can help prevent underfitting by providing additional structure to the model, allowing it to better capture the underlying patterns in the data.

Two examples of regularization techniques are:

L1 and L2 regularization: L1 and L2 regularization are two common techniques used to control model complexity by adding a penalty term to the model's objective function. L1 regularization adds a penalty term proportional to the absolute value of the model's weights, while L2 regularization adds a penalty term proportional to the square of the model's weights. Both techniques encourage the model to use only the most important features by shrinking the weights of irrelevant or noisy features towards zero.

Dropout regularization: Dropout is a regularization technique used in neural networks to prevent overfitting. Dropout works by randomly dropping out a proportion of the nodes in the network during each training iteration, forcing the remaining nodes to learn more robust features that are less dependent on the specific input features. This helps prevent overfitting by reducing the model's dependence on specific input features and encouraging it to learn more generalizable patterns.

4.Draw a plot that represents an underfitted, overfitted, and well fitted model, respectively?!
``` Python
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X = np.linspace(0, 1, 10)
y = np.sin(X * 4 * np.pi) + np.random.normal(0, 0.3, size=X.shape)


models = [
    {"name": "Underfitting", "degree": 1},
    {"name": "Well-fitted", "degree": 4},
    {"name": "Overfitting", "degree": 8},
]


plt.figure(figsize=(15, 4))
for i, model in enumerate(models):
    ax = plt.subplot(1, 3, i+1)
    ax.scatter(X, y, label="Data", color="black", alpha=0.5)
    coefs = np.polyfit(X, y, deg=model["degree"])
    y_pred = np.polyval(coefs, X)
    ax.plot(X, y_pred, label="Model", color="blue")
    ax.set_title(model["name"])
    ax.legend(loc="best")
plt.tight_layout()
plt.show()
```
<p align="center">
  <img src="https://github.com/AzadehKSH/Explainable_AI/blob/main/fitting.JPG" width="600" title="Fitting">
</p>

