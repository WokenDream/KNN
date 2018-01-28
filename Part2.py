import tensorflow as tf

def compute_R_solution(Distance, K = 2):
    numTrainData = tf.shape(Distance)[1]
    dist_k, ind_k = tf.nn.top_k(-Distance, k=K)
    R = tf.reduce_sum(tf.to_float(tf.equal(tf.expand_dims(ind_k, 2),
                                           tf.reshape(tf.range(numTrainData), [1, 1, -1]))), 1)
    return R / tf.to_float(K)

def compute_R(D, k=2):
    # num of row = num of train points
    # num of coln = num of test points
    # or the other way around

    vals, inds = tf.nn.top_k(-D, k=k)

    # the indices returned by top_k is not usable
    # we need to make it into a usable tensor that has encodes row number
    inds = tf.expand_dims(inds, axis=2) # reshape indices to 3D
    test_point_indices = tf.reshape(tf.range(tf.shape(D)[1]), shape=[1, 1, -1])  # a 3D row vector representing column indices

    # broadcast test_point_indices
    # 3D boolean tensor: 1stD: true/false, 2ndD: row vector of len(# of test) 3rdD: num of train * k
    top_k_boolean = tf.equal(inds, test_point_indices)

    # convert true to 1
    R = tf.reduce_sum(tf.cast(top_k_boolean, tf.int32), axis=1)
    return R/k

# def compute_R(D, k=2):
#     vals = tf.nn.top_k(-D, k=k).values * -1
#     min_kmax = tf.reduce_min(vals, axis=1)
#     R = tf.transpose(tf.transpose(D) - min_kmax).eval()
#     R[R < 0] = 0
#     R[R > 0] = 1
#
#     return R/k

# def compute_R(D, k=2):
#     vals, inds = tf.nn.top_k(-D, k=k)
#     kth = tf.reduce_min(vals)
#     topk = tf.greater_equal(D, kth).eval()
#     R = tf.zeros(tf.shape(D)).eval()
#     R[topk] = 1
#     return R/k


if __name__ == "__main__":
    X = tf.constant([1, 3, 2, 4, 3, 2, 3, 4, 5], shape=[3, 3])
    Y = tf.constant([5, 5, 5, 5, 5, 5, 5, 5, 5], shape=[3, 3])
    with tf.Session() as sess:
        print(compute_R(X))
        print(sess.run(compute_R_solution(X)))
        print(compute_R(Y))
        print(sess.run(compute_R_solution(Y)))
    # np.random.seed(521)
    # Data = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
    # Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
    # randIdx = np.arange(100)
    # np.random.shuffle(randIdx)
    # trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
    # validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    # testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
