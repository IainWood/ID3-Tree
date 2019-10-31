# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:57:00 2019
@author: Iain Woodburn
"""

import pandas as pd
import numpy as np
import sys

#hard code in the label globally
label = 'income_level'

#turns every attribute into a binary version of itself (except for the label)
def pre_process(data):
    for column_name in data.columns:
        if column_name != label:
            for unique_attr in np.unique(data[column_name]):
                data[unique_attr] = data[column_name] == unique_attr
            data.drop(column_name, axis=1, inplace=True)
    return data

#returns entropy
def entropy(S):
    total = sum(S)
    #to avoid a possible divide by zero error
    if total is 0:
        total = 1
    entropy = 0
    for i in S:
        p_i = float(i)/total
        if abs(p_i) > 1e-6:
            entropy -= p_i * np.log2(p_i)
    return entropy

#returns information gain
def info_gain(S, S_v):
    gain = entropy(S)
    for v in S_v:
        gain -= float(sum(v))/sum(S) * entropy(v)
    return gain

#from the data given, return the majority label
def majority_label(data):
    above = data.loc[data[label] == '>50K']
    below = data.loc[data[label] == '<=50K']

    if len(above) > len(below):
        return '>50K'
    else:
        return '<=50K'

#returns count of the node and all subnodes
def count(node):
    c =  1
    if node.is_leaf:
        return 1
    else:
        return c + count(node.left) + count(node.right)

#returns depth of a tree with root = node
def depth(node): 
    if node.is_leaf: 
        return 1  
    else: 
        right = depth(node.right) 
        left = depth(node.left) 
  
        if (right > left): 
            return right + 1
        else: 
            return left + 1

#used to store information along the tree
class Node():
    def __init__(self, data, is_leaf, label, split_attr):
        self.data = data
        self.is_leaf = is_leaf
        self.label = label
        self.split_attr = split_attr
        self.left = None
        self.right = None
        self.pruned = False

#based on the ID3 algorithm from wikipedia
def ID3(data, attributes, max_depth=0, depth=1):

    if max_depth > 0 and depth == max_depth:
        return Node(data, True, majority_label(data), None)
    
    #Create a root node for the tree
    root = Node(data, False, majority_label(data), None)

    #If all examples are positive, Return the single-node tree Root, with label = +.
    if len(data.loc[data[label] == '>50K']) is len(data):
        return Node(data, True, '>50K', None)

    #If all examples are negative, Return the single-node tree Root, with label = -.
    if len(data.loc[data[label] == '<=50K']) is len(data):
        return Node(data, True, '<=50K', None)

    #If number of predicting attributes is empty, then Return the single node tree Root,
    #with label = most common value of the target attribute in the examples.
    if attributes is None:
        return Node(data, True, majority_label(data), None)

    #A ← The Attribute that best classifies examples.
    split_attr = get_split(data)

    #Decision Tree attribute for Root = A.
    root.split_attr = split_attr

    #For each possible value, v_i, of A (all attributes are binary, so just T/F)
    right_data = data.loc[data[split_attr] == True]
    left_data = data.loc[data[split_attr] == False]

    try:
        temp_attr = list(attributes)
        temp_attr.remove(split_attr)
    except:
        temp_attr = None

    #If Examples(v_i) is empty
    if right_data.dropna().empty:
        #Then below this new branch add a leaf node with label = most common target value in the examples
        root.right = Node(None, True, majority_label(data), None)
    else:
        #Else below this new branch add the subtree ID3 (Examples(v_i), Attributes – {A})
        root.right = ID3(right_data, temp_attr, max_depth, depth + 1)

    #same but for left
    if left_data.dropna().empty:
        root.left = Node(None, True, majority_label(data), None)
    else:
        root.left = ID3(left_data, temp_attr, max_depth, depth + 1)

    #Return Root
    return root

def prune(root, test_root, data):
    
    node = root
    if root is None or root.pruned is True or root.is_leaf:
        return
        
    #recursively prunes the subtrees until a node with two leaves is found
    if node.right.is_leaf and node.left.is_leaf:
        
        base_acc = predict(test_root, data)
        #this tricks the predict funtion into thinking the tree stops here, 
        #don't have to actually remove the data and make a new label
        node.is_leaf = True
        new_acc = predict(test_root, data)
        
        #if this does not improve the tree then do not prune it, 
        #but mark it pruned so it doesn't get tested again
        if new_acc <= base_acc:
            node.is_leaf = False
            node.pruned = True
    else:
        prune(node.right, test_root, data)
        prune(node.left, test_root, data)

#finds the attribute with the greatest information gain based on the label
def get_split(data):
    above = data[label] == '>50K'
    below = data[label] == '<=50K'
    total_cnts = [len(data.loc[above]), len(data.loc[below])]
    best_info_gain = -1
    split_attr = None

    #try every column and see which has the biggest gain
    for attr in data.columns:
        if attr != label:
            indv_cnts = []
            indv_cnts.append([len(data.loc[(data[attr] == True) & above]), len(data.loc[(data[attr] == True) & below])])
            indv_cnts.append([len(data.loc[(data[attr] == False) & above]), len(data.loc[(data[attr] == False) & below])])
            gain = info_gain(total_cnts, indv_cnts)

            if  gain > best_info_gain:
                best_info_gain = gain
                split_attr = attr

    return split_attr

#walks down the tree based on values in the test data
def predict(root, data):
    correct = 0
    for index, row in data.iterrows():
        node = root
        while node.is_leaf is False:
            
            try:
                test = row[node.split_attr]
            except:
                #if the test record doesn't have that attribute then return the
                #majority label at that node
                break
            
            if test == True:
                node = node.right
            elif test == False:
                node = node.left
        
        if node.label == row[label]:
            correct += 1
    return correct/len(data) * 100

if __name__ == '__main__':

    columns = ['workclass', 'education', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'native_country', 'income_level']

    train_file, test_file = None, None
    model = None
    train_percent, validate_percent = -1, -1

    try:
        train_file = str(sys.argv[1])
        test_file = str(sys.argv[2])
        model = str(sys.argv[3])
        train_percent = int(sys.argv[4])
        validate_percent = int(sys.argv[5])
        if train_percent < 1 or train_percent > 100 or validate_percent < 1 or validate_percent > 100:
            print('please enter an integer[1, 100] for percents')
            sys.exit(0)
    except:
        if len(sys.argv) is 5 and train_percent > 0 or train_percent <= 100:
            validate_percent = None
        else:
            print('arguments error')
            sys.exit(0)

    train_data = pd.read_csv(train_file, delimiter=',', index_col=None, names=columns, engine='python')
    test_set = pd.read_csv(test_file, delimiter=',', index_col=None, names=columns, engine='python')

    #trim whitespace, some of the values are like ' <50K' and some '<50K'
    for column in train_data:
        train_data[column] = train_data[column].str.strip()
    for column in test_set:
        test_set[column] = test_set[column].str.strip()
    
    
    t_split_point = int(np.floor((train_percent * train_data.shape[0]) / 100))      #find the correct number of rows
    train_set = train_data.head(t_split_point).reset_index(drop=True)              #the first n percent
    

    #pre-process the data
    train_set = pre_process(train_set)
    test_set = pre_process(test_set)
    root = None

    #build the type of tree specified
    if model.lower() == 'vanilla':
        root = ID3(train_set, train_set.columns)
    elif model.lower() == 'prune':
        v_split_point = int(np.floor((validate_percent * train_data.shape[0]) / 100))
        validate_set = pre_process(train_data.tail(v_split_point).reset_index(drop=True))
        root = ID3(train_set, train_set.columns)
        prune(root, root, validate_set)
    elif model.lower() == 'maxdepth':
        v_split_point = int(np.floor((validate_percent * train_data.shape[0]) / 100))
        validate_set = pre_process(train_data.tail(v_split_point).reset_index(drop=True))
        max_depth = int(sys.argv[6])
        base_acc = -1
        best_tree = None
        best_depth = -1
        for i in range(max_depth):
            root = ID3(train_set, train_set.columns, i+1)
            acc = predict(root, validate_set)
            if acc > base_acc:
                base_acc = acc
                best_tree = root
                best_depth = i + 1
        root = best_tree
        
    #predict training and testing accuracies
    print('Train set accuracy: %.4f' % predict(root, train_set))
    print('Test set accuracy: %.4f' % predict(root, test_set))
