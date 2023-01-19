### what is f1 score?
- f1 score is the harmonic mean of precision and recall
### what is precision?
- precision is the number of true positives divided by the number of true positives plus the number of false positives
### what is recall?
- recall is the number of true positives divided by the number of true positives plus the number of false negatives
### what is accuracy?
- accuracy is the number of correct predictions divided by the total number of predictions

### random forest classifier and parameters
- random forest classifier is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
- parameters => n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, class_weight
. n_estimators => the number of trees in the forest
. criterion => the function to measure the quality of a split
. max_depth => the maximum depth of the tree
. min_samples_split => the minimum number of samples required to split an internal node
### KNN classifier and parameters
- KNN classifier is a non-parametric method used for classification and regression
- parameters => n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs
- n_neighbors => number of neighbors to use by default for kneighbors queries

### Decision Tree Classifier and parameters
- Decision Tree Classifier is a class capable of performing multi-class classification on a dataset
- parameters => criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight, presort
- criterion => the function to measure the quality of a split
- splitter => the strategy used to choose the split at each node
- max_depth => the maximum depth of the tree

### Support Vector Machine and parameters
- Support Vector Machine is a supervised machine learning model which is used for classification and regression problems
- parameters => C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, random_state
- C => penalty parameter C of the error term
- kernel => specifies the kernel type to be used in the algorithm

### K-Means Clustering and parameters
- K-Means Clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining
- parameters => n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state, copy_x, n_jobs, algorithm
- n_clusters => the number of clusters to form as well as the number of centroids to generate
- init => method for initialization
- max_iter => maximum number of iterations of the k-means algorithm for a single run

### different methods of feature selection
- univariate feature selection
- recursive feature elimination
- incremental feature selection
- decremental feature selection


### what is correlation and what is np.corrcoef?
- correlation is a statistical measure that expresses the extent to which two variables are linearly related
- np.corrcoef => computes the correlation coefficient between two arrays
- example => np.corrcoef([1, 0, 1], [0, 2, 1])