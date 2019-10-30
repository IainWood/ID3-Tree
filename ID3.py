# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:57:00 2019
@author: Iain Woodburn
"""

import pandas as pd
import numpy as np
import time
import sys

label = 'income_level'

#turns every attribute into a binary version of itself
def pre_process(data):
    for column_name in data.columns:
        if column_name != 'income_level':
            for unique_attr in np.unique(data[column_name]):
                data[unique_attr] = data[column_name] == unique_attr
            data.drop(column_name, axis=1, inplace=True)
    return data

def entropy(freqs):
    all_freqs = sum(freqs)
    if all_freqs is 0:
        all_freqs = 1
    entropy = 0
    for freq in freqs:
        prob = float(freq)/all_freqs
        if abs(prob) > 1e-8:
            entropy = entropy + (-prob * np.log2(prob))
    return entropy

def info_gain(before_split, after_split):
    gain = entropy(before_split)
    overall_size = sum(before_split)
    for freq in after_split:
        ratio = float(sum(freq))/overall_size
        gain = gain - (ratio * entropy(freq))
    return gain

def most_popular_label(data):
    above = data.loc[data[label] == '>50K']
    below = data.loc[data[label] == '<=50K']

    if len(above) > len(below):
        return '>50K'
    else:
        return '<=50K'

class Node():
    def __init__(self, data, is_leaf, label, split_attr):
        self.data = data
        self.is_leaf = is_leaf
        self.label = label
        self.split_attr = split_attr
        self.left = None
        self.right = None
        self.pruned = False

def count_nodes(node):
    cnt =  1
    if node is None:
        return 0
    else:
        cnt += count_nodes(node.left)
        cnt += count_nodes(node.right)
        return cnt

def maxDepth(node): 
    if node is None: 
        return 0 ;  
  
    else : 
  
        # Compute the depth of each subtree 
        lDepth = maxDepth(node.left) 
        rDepth = maxDepth(node.right) 
  
        # Use the larger one 
        if (lDepth > rDepth): 
            return lDepth+1
        else: 
            return rDepth+1


#based on the ID3 algorithm from wikipedia
def ID3(data, attributes):

    #Create a root node for the tree
    root = Node(data, False, most_popular_label(data), None)

    #If all examples are positive, Return the single-node tree Root, with label = +.
    if len(data.loc[data[label] == '>50K']) is len(data):
        return Node(data, True, '>50K', None)

    #If all examples are negative, Return the single-node tree Root, with label = -.
    if len(data.loc[data[label] == '<=50K']) is len(data):
        return Node(data, True, '<=50K', None)

    #If number of predicting attributes is empty, then Return the single node tree Root,
    #with label = most common value of the target attribute in the examples.
    if attributes is None:
        return Node(data, True, most_popular_label(data), None)

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
        root.right = Node(None, True, most_popular_label(data), None)
    else:
        #Else below this new branch add the subtree ID3 (Examples(v_i), Attributes – {A})
        root.right = ID3(right_data, temp_attr)

    #same but for left
    if left_data.dropna().empty:
        root.left = Node(None, True, most_popular_label(data), None)
    else:
        root.left = ID3(left_data, temp_attr)

    #Return Root
    return root

#based on the ID3 algorithm from wikipedia
def depth_ID3(data, attributes, max_depth=0, depth=1):

    if max_depth > 0 and depth == max_depth:
        return Node(data, True, most_popular_label(data), None)
        
    
    #Create a root node for the tree
    root = Node(data, False, most_popular_label(data), None)

    #If all examples are positive, Return the single-node tree Root, with label = +.
    if len(data.loc[data[label] == '>50K']) is len(data):
        return Node(data, True, '>50K', None)

    #If all examples are negative, Return the single-node tree Root, with label = -.
    if len(data.loc[data[label] == '<=50K']) is len(data):
        return Node(data, True, '<=50K', None)

    #If number of predicting attributes is empty, then Return the single node tree Root,
    #with label = most common value of the target attribute in the examples.
    if attributes is None:
        return Node(data, True, most_popular_label(data), None)

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
        root.right = Node(None, True, most_popular_label(data), None)
    else:
        #Else below this new branch add the subtree ID3 (Examples(v_i), Attributes – {A})
        root.right = depth_ID3(right_data, temp_attr, max_depth, depth + 1)

    #same but for left
    if left_data.dropna().empty:
        root.left = Node(None, True, most_popular_label(data), None)
    else:
        root.left = depth_ID3(left_data, temp_attr, max_depth, depth + 1)

    #Return Root
    return root

def prune(root, test_root, data):
    
    node = root
    if root.pruned is True or root is None or root.is_leaf:
        return
        
    #finds lowest level leaves
    if not node.right.is_leaf and not node.right.is_leaf:
        prune(node.left, test_root, data)
        if not node.right.is_leaf and not node.right.is_leaf:
            prune(node.right, test_root, data)
        
        
    if node.left.is_leaf and not node.right.is_leaf:
        prune(node.right, test_root, data)

    elif node.right.is_leaf and not node.left.is_leaf:
        prune(node.left, test_root, data)
    
    
    #when both children are leaves
    base_acc = predict(test_root, data)
    node.is_leaf = True
    new_acc = predict(test_root, data)
    
    if new_acc <= base_acc:
        node.is_leaf = False
        node.pruned = True


def get_split(data):
    above = data[label] == '>50K'
    below = data[label] == '<=50K'
    before_split = [len(data.loc[above]), len(data.loc[below])]
    best_info_gain = -1
    split_attr = None
    for attr in data.columns:
        if attr != label:
            after_split = []
            for value in data[attr].unique():
                after_split.append([len(data.loc[(data[attr] == value) & above]), len(data.loc[(data[attr] == value) & below])])

            gain = info_gain(before_split, after_split)

            if  gain > best_info_gain:
                best_info_gain = gain
                split_attr = attr

    return split_attr

def predict(root, data):
    correct = 0
    for index, row in data.iterrows():
        node = root
        while node.is_leaf is False:
            
            try:
                test = row[node.split_attr]
            except:
                break
            
            if test == True:
                node = node.right
            elif test == False:
                node = node.left
        
#        node.label = '<=50K'
        if node.label == row[label]:
            correct += 1
    return correct/len(data) * 100

if __name__ == '__main__':

    columns = ['workclass', 'education', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'native_country', 'income_level']

    train_file, test_file = None, None
    model = None
    train_percent, validate_percent = -1, -1

    #try:
    #    train_file = str(sys.argv[1])
    #    test_file = str(sys.argv[2])
    #    model = str(sys.argv[3])
    #    train_percent = int(sys.argv[4])
    #    validate_percent = int(sys.argv[5])
    #    if train_percent < 1 or train_percent > 99 or validate_percent < 1 or validate_percent > 99:
    #        print('please enter an integer[1, 99] for percents')
    #        sys.exit(0)
    #except ValueError:
    #    if len(sys.argv) is 5 and train_percent < 1 or train_percent > 99:
    #        validate_percent = None
    #    else:
    #        sys.exit(0)

    #train_percent = 100
    train_file = 'adult.data'
    test_file = 'adult.test'
    model = 'vanilla'

    train_data = pd.read_csv(train_file, delimiter=',', index_col=None, names=columns, engine='python')
    test_set = pd.read_csv(test_file, delimiter=',', index_col=None, names=columns, engine='python')

    #trim whitespace, some of the values are like ' <50K' and some '<50K'
    for column in train_data:
        train_data[column] = train_data[column].str.strip()
    for column in test_set:
        test_set[column] = test_set[column].str.strip()
    train_percent = 30
    for train_percent in [1]:#, 50, 60, 70, 80]:
#    for train_percent in [10, 50, 60, 70, 80]:
        print('training percent: ', train_percent)
    
        t_split_point = int(np.floor((train_percent * train_data.shape[0]) / 100))      #find the correct number of rows
        train_set = train_data.head(t_split_point).reset_index(drop=True)              #the first n percent
        
    
        train_set = pre_process(train_set)
        test_set = pre_process(test_set)
        root = None
    
        #build the type of tree specified
        if model.lower() == 'vanilla':
            start = time.time()
            root = ID3(train_set, train_set.columns)
            end = time.time()
            print('number of nodes: ', count_nodes(root))
            print(end - start)
            print()
    
        elif model.lower() == 'prune':
            v_split_point = int(np.floor((validate_percent * train_data.shape[0]) / 100))   #find the correct number of rows
            #set for testing
            v_split_point = 20
            validate_set = pre_process(train_data.tail(v_split_point).reset_index(drop=True))           #the last n percent
            
            
            start = time.time()
            root = ID3(train_set, train_set.columns)
            end = time.time()
            print('number of nodes: ', count_nodes(root))
            print(end - start)
            
            
#            print('Validate set accuracy: %0.4f' % predict(root, validate_set))
#            
#            print('Train set accuracy: %0.4f' % predict(root, train_set))
#            print('Test set accuracy: %0.4f' % predict(root, test_set))
            prune(root, root, validate_set)
    
        elif model.lower() == 'maxdepth':
            v_split_point = int(np.floor((validate_percent * train_data.shape[0]) / 100))   #find the correct number of rows
            validate_set = pre_process(train_data.tail(v_split_point).reset_index(drop=True))           #the last n percent
            max_depth = 6#int(sys.argv[6])
            root = depth_ID3(train_set, train_set.columns, max_depth)
            print('height: ', maxDepth(root))
    
        print('Train set accuracy: %0.4f' % predict(root, train_set))
        print('Test set accuracy: %0.4f' % predict(root, test_set))
        print()
