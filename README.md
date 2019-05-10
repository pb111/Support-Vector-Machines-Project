# Support Vector Machines with Python and Scikit-Learn

In this project, I build a Support Vector Machines classifier to classify a Pulsar star. I have used the **Predicting a Pulsar Star** dataset for this project. I have downloaded this dataset from the Kaggle website.


==============================================================================

## Table of Contents

I have categorized this project into various sections which are listed below:-

1.	Introduction to Support Vector Machines
2.	Types of SVM classifier
3.	Support Vector Machines intuition
4.	Kernel trick
5.	Advantages and disadvantages of SVM
6.	SVM Scikit-Learn packages
7.	The problem statement
8.	Results and conclusion
9.	Applications of SVM
10.	References


















## 1. Introduction to Support Vector Machines

**Support Vector Machines** (SVMs in short) are machine learning algorithms that are used for classification and regression purposes. SVMs are one of the powerful machine learning algorithms for classification, regression and outlier detection purposes. An SVM classifier builds a model that assigns new data points to one of the given categories. Thus, it can be viewed as a non-probabilistic binary linear classifier.

The original SVM algorithm was developed by Vladimir N Vapnik and Alexey Ya. Chervonenkis in 1963. At that time, the algorithm was in early stages. The only possibility is to draw hyperplanes for linear classifier. In 1992, Bernhard E. Boser, Isabelle M Guyon and Vladimir N Vapnik suggested a way to create non-linear classifiers by applying the kernel trick to maximum-margin hyperplanes. The current standard was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.

SVMs can be used for linear classification purposes. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using the **kernel trick**. It enable us to implicitly map the inputs into high dimensional feature spaces.



## 2. Types of SVM classifier
In a dataset, where we have features and labels, an SVM classifier builds a model to predict classes for new examples.  It assigns new data points to one of the predicted classes. If there are only two classes, then it can be called as a `Binary SVM Classifier`. If we have more than two classes, then it is called ` Multi SVM Classifier`.

There are 2 kinds of SVM classifiers –

1.	Linear SVM Classifier
2.	Non-linear SVM Classifier

These are described below-
** 1. Linear SVM Classifier**
In the linear SVM Classifier, we assume that training samples are plotted in space. These samples are expected to be separated by an apparent gap. It predicts a straight hyperplane that divides the two classes. The primary focus while drawing the hyperplane is on maximizing the distance from hyperplane to the nearest sample data point of either class. The drawn hyperplane is called a `maximum-margin hyperplane`.

** 2. Non-linear SVM Classifier**
In general, any real world dataset is dispersed up to some extent. So, the datasets cannot be separated into different classes on the basis of a straight line hyperplane. So, Vapnik suggested to create non-linear classifiers by applying the kernel trick (discussed later) to maximum-margin hyperplanes. In non-linear SVM classification, the data points are plotted in a higher dimensional space.






## 3. Support Vector Machines intuition

Now, we should be familiar with some SVM terminology. 

### Hyperplane

A hyperplane is a decision boundary which separates between given set of data points having different class labels. The SVM classifier separates data points using a hyperplane with the maximum amount of margin. This hyperplane is known as the `maximum margin hyperplane` and the linear classifier it defines is known as the `maximum margin classifier`.

### Support Vectors

Support vectors are the sample data points, which are closest to the hyperplane.  These data points will define the separating line or hyperplane better by calculating margins.

### Margin

A margin is a separation gap between the two lines on the closest data points. It is calculated as the perpendicular distance from the line to support vectors or closest data points. In SVMs, we try to maximize this separation gap so that we get maximum margin.

The following diagram illustrates these concepts visually.

## D – SVM Terminology


### SVM Under the hood

In SVMs, our main objective is to select a hyperplane with the maximum possible margin between support vectors in the given dataset. SVM searches for the maximum margin hyperplane in the following 2 step process –
1.	Generate hyperplanes which segregates the classes in the best possible way. There are many hyperplanes that might classify the data. We should look for the best hyperplane that represents the largest separation, or margin, between the two classes.

2.	So, we choose the hyperplane so that distance from it to the support vectors on each side is maximized. If such a hyperplane exists, it is known as the **maximum margin hyperplane** and the linear classifier it defines is known as a **maximum margin classifier**. 

The following diagram illustrates the concept of **maximum margin** and **maximum margin hyperplane** in a clear manner. 

## D – SVM - MMH



### The problem of dispersed datasets
Sometimes, the sample data points are so dispersed that it is not possible to separate them using a linear hyperplane. 
In such a situation, SVMs uses a `kernel trick` to transform the input space to a higher dimensional space as shown in the diagram below. It uses a mapping function to transform the 2-D input space into the 3-D input space. Now, we can easily segregate the data points using linear separation.

## D – SVM kernel trick



## 4. Kernel trick
In practice, SVM algorithm is implemented using a `kernel`. It uses a technique called the `kernel trick`. In simple words, a `kernel` is just a function that maps the data to a higher dimension where data is separable. A kernel transforms a low-dimensional input data space into a higher dimensional space. So, it converts non-linear separable problems to linear separable problems by adding more dimensions to it. Thus, the kernel trick helps us to build a more accurate classifier. Hence, it is useful in non-linear separation problems.
We can define a kernel function as follows-

## D – kernel function

The value of this function is 1 inside the closed ball of radius 1 centered at the origin, and 0 otherwise.
In the context of SVMs, there are 4 popular kernels – `Linear kernel`, `Polynomial kernel` and `Radial Basis Function (RBF) kernel` (also called Gaussian kernel) and `Sigmoid kernel`. These are described below-

### 1. Linear kernel 
In linear kernel, the kernel function takes the form of a linear function as follows-
**linear kernel : K(xi , xj ) = xiT xj**
Linear kernel is used when the data is linearly separable. It means that data can be separated using a single line. It is one of the most common kernels to be used. It is mostly used when there are large number of features in a dataset. Linear kernel is often used for text classification purposes. 
Training with a linear kernel is usually faster, because we only need to optimize the C regularization parameter. When training with other kernels, we also need to optimize the γ parameter. So, performing a grid search will usually take more time.


Linear kernel can be visualized with the following figure.
## D – SVM linear kernel

### 2. Polynomial Kernel 
Polynomial kernel represents the similarity of vectors (training samples) in a feature space over polynomials of the original variables. The polynomial kernel looks not only at the given features of input samples to determine their similarity, but also combinations of the input samples. 
For degree-d polynomials, the polynomial kernel is defined as follows –


**Polynomial kernel : K(xi , xj ) = (γxiT xj + r)d , γ > 0**
 
Polynomial kernel is very popular in Natural Language Processing. The most common degree is d = 2 (quadratic), since larger degrees tend to overfit on NLP problems. It can be visualized with the following diagram.

## D – SVM Polynomial kernel



### 3. Radial Basis Function Kernel
Radial basis function kernel is a general purpose kernel. It is used when we have no prior knowledge about the data. The RBF kernel on two samples x and y is defined by the following equation –

## D – SVM RBF Kernel

It is a popular kernel function and is commonly used in support vector machine classification. We can visualize the RBF kernel with the following figure -

## D – SVM RBF Kernel diagram


### 4. Sigmoid kernel
Sigmoid kernel has its origin in neural networks. We can use it as the proxy for neural networks. Sigmoid kernel is given by the following equation –
** sigmoid kernel – k (x, y) = tanh(αxTy + c) **



## 5. Advantages and disadvantages of SVM
SVMs have quite a number of advantages and disadvantages. These are listed in the following paragraphs.
The advantages of SVM classifier are as follows:-
1.	SVMs are effective when we have a very large feature space or number of features are very large.
2.	It works effectively even if the number of features are greater than the number of samples.
3.	Non-linear datasets can also be effectively classified using customized hyperplanes formed by using the kernel trick.
4.	It is a very robust model to solve prediction problems because it maximizes margin.

The disadvantages of SVM classifier are listed below:-
1.	Probably the biggest drawback of SVM is that as sample size increases, it performs poorly.
2.	Another disadvantage of SVM is the choice of kernel. The wrong kernel can lead to an increase in error percentage.
3.	SVMs have good generalization performance but they can be extremely slow in the testing phase.
4.	They have high algorithmic complexity and extensive memory requirements due to the use of quadratic programming.

## 6. SVM Scikit-Learn libraries
Scikit-Learn provides useful libraries to implement Support Vector Machine algorithm on       a dataset. There are many libraries that can help us to implement SVM smoothly. We just need to call the library with parameters that suit to our needs. In this project, I am dealing with a classification task. So, I will mention the Scikit-Learn libraries for SVM classification purposes.
First, there is a **LinearSVC()** classifier. As the name suggests, this classifier uses only linear kernel. In LinearSVC() classifier, we don’t pass the value of kernel since it is used only for linear classification purposes.
Scikit-Learn provides two other classifiers - **SVC()** and **NuSVC()** which are used for classification purposes. These classifiers are mostly similar with some difference in parameters.  NuSVC() is similar to SVC but uses a parameter to control the number of support vectors. We pass the values of `kernel`, `gamma` and `C` along with other parameters. By default kernel parameter uses “rbf” as its value but we can pass values like “poly”, “linear”, “sigmoid” or callable function. 


## 7. The problem statement
## 8. Results and conclusion


## 9. Applications of SVM
SVMs are the by-product of neural network. They are widely used in pattern classification and classification and regression problems. Some applications of SVMs are listed below-
1.	Facial expression classification – SVMs can be used to classify facial expressions. It uses statistical models of shape and SVMs.

2.	Speech recognition – SVMs are used to accept keywords and reject non-keywords. Hence, they are used to build a model to recognize speech.

3.	Handwritten digit recognition – SVMs can be applied to the recognition of isolated handwritten digits optically scanned.

4.	Text categorization – Information retrieval and categorization of data using labels can be done by SVM.

5.	The SVM algorithm has been widely applied in biological and other sciences.




## 10. References

The work done in this project is inspired from following books and websites:-

1.	Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron
2.	Introduction to Machine Learning with Python
3.	Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves
4.	Udemy course – Feature Engineering for Machine Learning by Soledad Galli
5.	https://en.wikipedia.org/wiki/Support-vector_machine
6.	https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
7.	http://dataaspirant.com/2017/01/13/support-vector-machine-algorithm/
8.	https://www.ritchieng.com/machine-learning-evaluate-classification-model/ 

9.	https://en.wikipedia.org/wiki/Kernel_method

10.	https://en.wikipedia.org/wiki/Polynomial_kernel

11.	https://en.wikipedia.org/wiki/Radial_basis_function_kernel

12.	https://data-flair.training/blogs/svm-kernel-functions/



