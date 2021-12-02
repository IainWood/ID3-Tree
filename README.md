# ID3-Tree
Decision tree implemented according to the ID3 algorithm, supports pruning and maximum depth

With the FULL adult.test dataset, the ID3 algorithm runs at almost exactly 5 minutes
The prediction functions runs very quickly

If the training dataset is too large, however, then there is risk of violating Python's max recursion limit, which is 1000 (i.e. if the depth of the tree reaches 1000 then the program overflows)

Arguments:
    training file - data to train a model for
    testing file - data to run against an already trained tree
    Model:
        vanilla - trains and executes normally, building as large a tree as necessary
        prune - prunes the tree built by the algorithm
        maxdepth - set a maximum height to which the decision tree will recursively grow (a fifth argument will have to be passed for the max depth of the tree)
    training percent[1, 100] - amount of the training file to use as a percentage
    testing percent[1, 100] - amount of the test file to use as a percentage

Example:
    >python ID3.py adult.data adult.test vanilla 50 10
        Train set accuracy: 88.1400
        Test set accuracy: 79.7300

    >python ID3.py adult.data adult.test vanilla 10 40
        Train set accuracy: 92.2000
        Test set accuracy: 77.6000
    
    >python ID3.py adult.data adult.test prune 80 20
        Train set accuracy: 87.8000
        Test set accuracy: 80.7200
    
    >python ID3.py adult.data adult.test maxdepth 70 10 10
        Train set accuracy: 83.2286
        Test set accuracy: 81.4600