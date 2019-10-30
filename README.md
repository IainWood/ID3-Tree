# ID3-Tree
Decision tree implemented according to the ID3 algorithm, supports pruning and maximum depth

With the FULL adult.test dataset, the ID3 algorithm runs at almost exactly 5 minutes
The prediction functions runs very quickly

If the training dataset is too large, however, then there is risk of violating Python's max recursion limit, which is 1000 (i.e. if the depth of the tree reaches 1000 then the program overflows)
