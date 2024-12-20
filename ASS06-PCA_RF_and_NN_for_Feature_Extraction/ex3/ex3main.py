'''
ex3main.py
Code by M11351802 - Herranz Gancedo, Lucas
This script loads the Iris dataset, feeds the data through a feature extraction deep
neural network. The output of this feature extractor model is fed to a Support Vector
Machine (SVM) classifier, which will perform a binary classification of setosa samples
non-setosa. The model will be tested with part of the data and its metrics displayed.
'''

# Local libraries
from ex3datasets.ex3datasets import ToTensor, StandardScaler, IrisDataset, ExtractedFeatureDataset
from ex3featureextractor.ex3featureextractor import feature_extraction, DEVICE
from ex3svc.ex3svc import binaryClass_SVM

# Third-party libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

if __name__ == "__main__":
    # 0) GPU device selection (if avaliable)
    print(f"Using {DEVICE} device")

    # Define transformations
    to_tensor = ToTensor()
    scaler = StandardScaler(load_iris())

    # Create the IrisDataset with both transformations
    iris_dataset = IrisDataset(transform=to_tensor, feature_scaling=scaler)

    # Extract the features of the IrisDataset
    features, labels = feature_extraction(iris_dataset)
    
    n_samples = len(features)
    n_features = len(features[0])

    print(f'Output of feature extractor shape / Input of SVM classifier shape: {n_samples}, {n_features}')
    print()

    # Create a new dataset class for the output of the first
    cl_dataset = ExtractedFeatureDataset(features, labels)

    binaryClass_SVM(cl_dataset)
    plt.show()