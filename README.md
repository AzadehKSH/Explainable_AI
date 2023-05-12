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


