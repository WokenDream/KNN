import tensorflow as tf
import numpy as np
import Part1 as p1
import matplotlib.pyplot as plt

def compute_R(D, k=1):
    """
    Compute responsibility matrix for each test point.
    Row i in D is the pairwise distance between test point i and each training point
    :param D: pairwise distance matrix
    :param k: number of closest neighbours to consider
    :return: responsibility matrix for each test point
    """
    vals, inds = tf.nn.top_k(-D, k=k)

    # the indices returned by top_k is not directly usable
    # we need to make it into a usable boolean tensor that has encodes row number
    inds = tf.expand_dims(inds, axis=2)  # reshape indices to 3D to make each index stand alone
    # a 3D row vector representing column indices
    test_point_indices = tf.reshape(tf.range(tf.shape(D)[1]), shape=[1, 1, -1])

    # broadcast both (to match dimensions) to 3D boolean tensor:
    # 1st D: true/false (top_k or not), 2nd D: row vector of len(# of train) 3rdD: num of test * k
    top_k_boolean = tf.equal(inds, test_point_indices)

    # convert true to 1, and  undo the broadcasting to make it 2D
    R = tf.reduce_sum(tf.cast(top_k_boolean, tf.int32), axis=1)
    return R/k

def pred_KNN(True_Y, D, k=1):
    """

    :param True_Y: a 1 by N matrix where each entry represents true labels Y of Xi
    :param D: pairwise distance matrix of shape [number of test point, number of training point]
    :param k: number of closest neighbours to consider
    :return: 1 by N matrix representing predicted values for all test point
    """
    R = compute_R(D, k)
    return tf.matmul(True_Y, R, transpose_b=True)
    # return tf.transpose(tf.matmul(R, True_Y)) # (A^T)*(B^T) = (B*A)^T

def compute_MSE(True_Y, Pred_Y):
    """
    Compute mean squrared error between [1 by N] matrices Y_True and Y_pred
    :param True_Y: true label matrix
    :param Pred_Y: prediction matrix
    :return: mean squared error tf integer
    """
    error = tf.reduce_sum(tf.squared_difference(True_Y, Pred_Y))
    return error / tf.cast(2 * tf.shape(Pred_Y)[1], dtype=tf.float64)

if __name__ == "__main__":

    # code from handout to generates 100 points
    np.random.seed(521)
    Data = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
    Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

    # reshape for ease of use
    trainTarget_row_vec = np.reshape(trainTarget, [1, -1])
    validTarget_row_vec = np.reshape(validTarget, [1, -1])
    testTarget_row_vec = np.reshape(testTarget, [1, -1])

    # compute pairwise distance matrix
    Train_D = p1.compute_pairwise_distance(trainData, trainData)
    Vali_D = p1.compute_pairwise_distance(validData, trainData)
    Test_D = p1.compute_pairwise_distance(testData, trainData)
    with tf.Session() as sess:
        for k in [1, 3, 5, 50]:
            print("k: ", k)

            Train_Pred = pred_KNN(trainTarget_row_vec, Train_D, k=k)
            Vali_Pred = pred_KNN(trainTarget_row_vec, Vali_D, k=k)
            Test_Pred = pred_KNN(trainTarget_row_vec, Test_D, k=k)
            train_error = compute_MSE(trainTarget_row_vec, Train_Pred).eval()
            vali_error = compute_MSE(validTarget_row_vec, Vali_Pred).eval()
            test_error = compute_MSE(testTarget_row_vec, Test_Pred).eval()

            print("training error: ", train_error)
            print("validation error: ", vali_error)
            print("test error: ", test_error)

    # generate another 1000 points for prediction and plotting as required
    Data = np.linspace(0.0, 11.0, num=1000)[:, np.newaxis]
    Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(1000, 1)
    randIdx = np.arange(1000)
    np.random.shuffle(randIdx)
    testData, testTarget = Data[randIdx[:1000]], Target[randIdx[:1000]]

    # reshape for ease of use
    testTarget_row_vec = np.reshape(testTarget, [1, -1])
    print("1000-test set shape: ", testTarget_row_vec.shape)
    Test_D = p1.compute_pairwise_distance(testData, trainData)
    with tf.Session() as sess:
        # i = 1
        for k in [1, 3, 5, 50]:
            print("k: ", k)

            Test_Pred = pred_KNN(trainTarget_row_vec, Test_D, k=k)
            test_error = compute_MSE(testTarget_row_vec, Test_Pred).eval()
            print("test error: ", test_error)

            # plt.subplot(4, 1, i)
            plt.scatter(testData, tf.transpose(Test_Pred).eval(), s=1)
            plt.scatter(testData, testTarget, s=1)
            plt.title("kNN Regression on 1000 points, k = " + str(k))
            plt.show()
            # i += 1
        # plt.show()