import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from collections import Counter

# Load the Iris dataset
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=['sl', 'sw', 'pl', 'pw'])
    y = pd.Series(iris.target)
    return X, y

# The next two functions are used to encode a continous value 'val' (petal 
# width, petal length, sepal width, sepal length) with one of the four 
# labels (a, b, c, d):
#
# LABEL a: if (MIN_Value) <= val < (MIN_Value + Mean_Value)/2
# LABEL b: if (MIN_Value + Mean_Value)/2 <= val < (Mean_Value)
# LABEL c: if (Mean_Value) <= val < (Mean_Value + MAX_Value)/2
# LABEL d: if (Mean_Value + MAX_Value)/2 <= val <= (MAX_Value)

def encode_to_label(value, boundaries):
    if (value < boundaries[0]):
        return 'a'
    elif (value < boundaries[1]):
        return 'b'
    elif (value < boundaries[2]):
        return 'c'
    else:
        return 'd'

# Defines the encoding thresholds and encodes feature samples to one 
# of four labels
def label_features(df, feature_name):
    # There are four labels: a, b, c, d
    feature = df[feature_name]
    min_val, max_val = feature.min(), feature.max()
    mean_val = feature.mean()
    thresholds = [
        (min_val + mean_val) / 2,
        mean_val,
        (max_val + mean_val) / 2,
    ]
    return feature.apply(encode_to_label, boundaries=thresholds)

# Calculate and display the intervals for encoding values as 'a', 'b', 'c', 'd'
def print_encoding_intervals(df):
    print("Encoding intervals for each feature:")
    for feature in df.columns:
        feature_data = df[feature]
        min_val = feature_data.min()
        max_val = feature_data.max()
        mean_val = feature_data.mean()
        a = (min_val, (min_val + mean_val) / 2)
        b = ((min_val + mean_val) / 2, mean_val)
        c = (mean_val, (max_val + mean_val) / 2)
        d = ((max_val + mean_val) / 2, max_val)
        print(f"  Feature '{feature}':")
        print(f"    'a': {a[0]:.2f} <= val < {a[1]:.2f}")
        print(f"    'b': {b[0]:.2f} <= val < {b[1]:.2f}")
        print(f"    'c': {c[0]:.2f} <= val < {c[1]:.2f}")
        print(f"    'd': {d[0]:.2f} <= val <= {d[1]:.2f}")
    print("\n")

# Compute entropy of a set
def parent_node_entropy(targets, base=3):
    probabilities = np.bincount(targets) / len(targets)
    return -np.sum(np.fromiter((p * np.emath.logn(base, p) for p in probabilities if p > 0), dtype=np.float32))

# Compute entropy for the child nodes and split information
def child_node_entropy(df, targets, selected_feature, base=3):
    total_entropy = 0.0
    split_info = 0.0
    n_samples = len(targets)

    for _, group in df.groupby(selected_feature):
        aux_targets = targets[group.index]
        label_prob = len(aux_targets) / n_samples
        total_entropy += label_prob * parent_node_entropy(aux_targets, base=3)
        split_info -= label_prob * np.emath.logn(base, label_prob)

    return total_entropy, split_info


def gain(df, targets, selected_feature):
    entropy = parent_node_entropy(targets, base=3)
    child_entropy, split_info = child_node_entropy(df, targets, selected_feature, base=3)
    information_gain = entropy - child_entropy
    return information_gain / split_info if split_info != 0 else 0


def build_tree(df, targets, features, depth=0):
    # Count the samples for each class at the current level
    class_counts = Counter(targets)
    print(f"Level {depth}: Sample count by class: {class_counts}")
    
    # Check if all targets are the same
    if len(set(targets)) == 1:
        print(f"Level {depth}: Leaf node reached with class {targets.iloc[0]}\n")
        return

    # Check if no features are left to split
    if not features:
        most_common = Counter(targets).most_common(1)[0][0]
        print(f"Level {depth}: Leaf node with majority class {most_common}\n")
        return

    # Select the feature with the highest gain ratio
    gains = {feature: gain(df, targets, feature) for feature in features}
    best_feature = max(gains, key=gains.get)
    print(f"Level {depth}: Splitting on feature '{best_feature}' with gain ratio {gains[best_feature]:.4f}")

    # Remove the feature used for splitting from the list
    remaining_features = [f for f in features if f != best_feature]

    # Split on the best feature and recurse
    for label, group in df.groupby(best_feature):
        branch_targets = targets[group.index]
        branch_class_counts = Counter(branch_targets)

        print(f"--> Branch '{label}' at Level {depth + 1}:")
        print(f"    Sample count by class in this branch: {branch_class_counts}")
        print(f"    Total samples in branch '{label}': {len(branch_targets)}")

        # Recursive call for the subtree
        build_tree(group, branch_targets, remaining_features, depth + 1)

if __name__ == '__main__':
    # Load data and encode features
    X, y = load_data()
    print_encoding_intervals(X)  # Print intervals at the start
    for feature in X.columns:
        X[feature + '_labeled'] = label_features(X, feature)
    labeled_features = [col for col in X.columns if '_labeled' in col]
    
    # Drop original (unlabeled) features
    X = X[labeled_features]

    # Build classification tree
    build_tree(X, y, labeled_features)