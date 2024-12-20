import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

if __name__ == "__main__":
    # Create random 2-D data
    data_mean = np.array([10, 13])
    data_covariance = np.array([[3.5, -1.8], [-1.8, 3.5]])

    print('data_mean: ', data_mean.shape)
    print('data_covariance: ', data_covariance.shape)

    # Create 1000 samples using mean and variance
    input_data = rnd.multivariate_normal(data_mean, data_covariance, size= (1000))
    print('input_data: ', input_data.shape)


    plt.figure(0)
    plt.subplot(1,3,1)
    plt.title("Input data")
    plt.scatter(input_data[:,0], input_data[:,1])
    plt.grid(True)

    # Centering the data on its mean
    mean = np.mean(input_data, axis= 0)
    print("Mean: ", mean.shape)
    mean_centered_data = input_data - mean
    print("Mean centered data: ", mean_centered_data.shape)

    plt.subplot(1,3,2)
    plt.scatter(mean_centered_data[:,0], mean_centered_data[:,1])
    plt.title("Mean centered data")
    plt.grid(True)

    # Compute and print the covariance matrix
    cov_matrix = np.cov(mean_centered_data.T)
    cov_matrix = np.round(cov_matrix, 2)
    print('Covariance matrix: ', cov_matrix.shape)

    # Obtain the eigen decomposition of the covariance matrix
    eigen_val, eigen_vec = np.linalg.eig(cov_matrix)
    print('Eigen vectors: ', eigen_vec)
    print('Eigen values: ', eigen_val)

    # Sort eigenvectors and eigenvalues in descending order to
    # find the direction of biggest variance
    indexes = np.arange(0, len(eigen_val), 1)
    indexes = ([i for _, i in sorted(zip(eigen_val, indexes))])[::-1]
    eigen_val = eigen_val[indexes]
    eigen_vec = eigen_vec[:, indexes]
    print("Sorted Eigen vectors: ", eigen_vec)
    print("Sorted Eigen values: ", eigen_val)

    # Compute explained variance to decide which feature to keep for
    # dimension reduction
    sum_eigen_val = np.sum(eigen_val)
    explained_variance = eigen_val / sum_eigen_val
    print('Explained variance: ', explained_variance)
    cumulative_variance = np.cumsum(explained_variance)
    print('Cumulative variance: ', cumulative_variance)

    # Take transpose of eigen vectors with data
    pca_data = np.dot(mean_centered_data, eigen_vec)
    print("Transformed data ", pca_data.shape)

    plt.subplot(1,3,3)
    plt.scatter(pca_data[:,0], pca_data[:,1])
    plt.title("Transformed data")
    plt.grid(True)
    plt.show()