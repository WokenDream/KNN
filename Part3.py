import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Part1 as p1
import Part2 as p2


def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8 * len(rnd_idx))
    validBatch = int(0.1 * len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch], :], \
                                     data[rnd_idx[trBatch + 1:trBatch + validBatch], :], \
                                     data[rnd_idx[trBatch + validBatch + 1:-1], :]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
                                           target[rnd_idx[trBatch + 1:trBatch + validBatch], task], \
                                           target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def compute_accuracy(True_Y, Pred_Y):
    """
    Compute the similarity between true labels and predicted labels
    :param True_Y: 1 by N matrix of true labels
    :param Pred_Y: 1 by N matrix of predicted labels
    :return: similarity as a ratio between True_Y and Pred_Y
    """
    difference = tf.count_nonzero(True_Y - Pred_Y, dtype=tf.int32)
    num = tf.shape(Pred_Y)[1]
    return (num - difference) / num


def predict_KNN(trainTarget, D, k):
    """
    Predict labels for each test point in D.
    :param trainTarget: a vector of true labels for each training point
    :param D: pairwise distance matrix of shape [number of test point, number of training point]
    :param k: how many neighbours to consider
    :return: prediction vector containing prediction for each test point
    """
    num_of_test = tf.shape(D)[0].eval()
    D = -D
    predictions = np.zeros(num_of_test, dtype=np.int32)
    for i in range(num_of_test):
        top_k_inds = tf.nn.top_k(D[i], k=k).indices.eval()
        top_k_labels = trainTarget[top_k_inds]
        unique_labels, _, counts = tf.unique_with_counts(top_k_labels)
        label_index = tf.argmax(counts)
        predictions[i] = unique_labels[label_index].eval()
    return tf.reshape(predictions, [1, -1])


def predict_face_KNN():
    trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('./data.npy',
                                                                                             './target.npy', 0)
    # compute pairwise distance
    Vali_D = p1.compute_pairwise_distance(validData, trainData)
    Test_D = p1.compute_pairwise_distance(testData, trainData)

    # save repeated computation
    validTarget_row_vec = tf.cast(tf.reshape(validTarget, [1, -1]), tf.int32)
    testTarget_row_vec = tf.cast(tf.reshape(testTarget, [1, -1]), tf.int32)

    best_accuracy = 0.0
    best_k = 1

    with tf.Session() as sess:
        # for k in [1, 5, 10, 25, 50, 100, 200]:
        #     print("k: ", k)
        #     Vali_Pred = predict_KNN(trainTarget, Vali_D, k=k)
        #     accuracy = compute_accuracy(validTarget_row_vec, Vali_Pred).eval()
        #     print("accuracy", accuracy)
        #     if accuracy > best_accuracy:
        #         best_accuracy = accuracy
        #         best_k = k
        # print("best k: ", best_k, " best accuracy: ", best_accuracy)
        #
        # Test_Pred = predict_KNN(trainTarget, Test_D, k=best_k)
        # accuracy = compute_accuracy(testTarget_row_vec, Test_Pred).eval()
        # print("test accuracy with best k = ", best_k, " is: ", accuracy)

        get_label = get_name
        display_failure(trainTarget, validTarget, trainData, validData, Vali_D, get_label, sample_num=1)



def predict_gender_KNN():
    trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('./data.npy',
                                                                                             './target.npy', 1)
    # compute pairwise distance
    Vali_D = p1.compute_pairwise_distance(validData, trainData)
    Test_D = p1.compute_pairwise_distance(testData, trainData)

    # save repeated computation
    validTarget_row_vec = tf.cast(tf.reshape(validTarget, [1, -1]), tf.int32)
    testTarget_row_vec = tf.cast(tf.reshape(testTarget, [1, -1]), tf.int32)

    best_accuracy = 0.0
    best_k = 1

    with tf.Session() as sess:
        # for k in [1, 5, 10, 25, 50, 100, 200]:
        #     print("k: ", k)
        #     Vali_Pred = predict_KNN(trainTarget, Vali_D, k=k)
        #     accuracy = compute_accuracy(validTarget_row_vec, Vali_Pred).eval()
        #     print("accuracy", accuracy)
        #     if accuracy > best_accuracy:
        #         best_accuracy = accuracy
        #         best_k = k
        # print("best k: ", best_k, " best accuracy: ", best_accuracy)
        #
        # Test_Pred = predict_KNN(trainTarget, Test_D, k=best_k)
        # accuracy = compute_accuracy(testTarget_row_vec, Test_Pred).eval()
        # print("test accuracy with best k = ", best_k, " is: ", accuracy)

        get_label = get_gender
        display_failure(trainTarget, validTarget, trainData, validData, Vali_D, get_label, sample_num=2)


def display_failure(trainTarget, validTarget, trainData, validData, Vali_D, get_label, sample_num=0, k=10):
    """
    Display the failure case:
     1. save the image and label of a validation point.
     2. print the predicted label
     2. save all the top k neighbours images and their labels.
    :param trainTarget:  a vector of true labels for each training point
    :param validTarget: a vector of true labels for each validation point
    :param trainData: matrix represent the greyscale pixel of each training point
    :param validData:  matrix represent the greyscale pixel of each validation point
    :param Vali_D: pairwise distance matrix between validation and training points
    :param get_label: a function for decoding integers to labels
    :param sample_num: the index of the failure to examine
    :param k: how many neigbours to consider
    :return: None
    """
    Vali_Pred = predict_KNN(trainTarget, Vali_D, k=k)
    validTarget_row_vec = tf.cast(tf.reshape(validTarget, [1, -1]), tf.int32)
    diff = (validTarget_row_vec - Vali_Pred).eval()  # non-zero entries mean misclassified
    miss_inds = np.nonzero(np.reshape(diff, -1))  # 1 by N matrix represent indices of miss
    sample_ind = miss_inds[0][sample_num]  # take the 2nd miss to comment on

    # display the image of true label
    true_label = get_label(validTarget[sample_ind])
    true_image = validData[sample_ind]
    plt.imshow(np.reshape(true_image, [32, 32]))
    plt.title(true_label)
    plt.savefig(true_label + " true")

    # print the label of the prediction
    print("prediction: ", get_label(Vali_Pred[0][sample_ind].eval()))

    # display the k neighbours
    inds = tf.nn.top_k((-Vali_D)[sample_ind], k=k).indices.eval()
    top_k_labels = trainTarget[inds]
    top_k_images = trainData[inds]
    i = 0
    for label, image in zip(top_k_labels, top_k_images):
        neighbour_label = get_label(label)
        plt.imshow(np.reshape(image, [32, 32]))
        plt.title(neighbour_label)
        plt.savefig(neighbour_label + str(i))
        i += 1


def get_name(label):
    names = ['Lorraine Bracco', 'Gerard Butler', 'Peri Gilpin', 'Angie Harmon', 'Daniel Radcliffe', 'Michael Vartan']
    return names[label]


def get_gender(label):
    return ['Male', 'Female'][label]


if __name__ == "__main__":
    with tf.Session() as sess:
        predict_face_KNN()
        predict_gender_KNN()