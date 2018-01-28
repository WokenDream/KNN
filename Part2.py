import tensorflow as tf

def compute_R_solution(Distance, K = 2):
    numTrainData = tf.shape(Distance)[1]
    dist_k, ind_k = tf.nn.top_k(-Distance, k=K)
    R = tf.reduce_sum(tf.to_float(tf.equal(tf.expand_dims(ind_k, 2),
                                           tf.reshape(tf.range(numTrainData), [1, 1, -1]))), 1)
    return R / tf.to_float(K)

# def compute_R_2(D, k=2):
#     vals, indices = tf.nn.top_k(-D, k=k)
#     shape = tf.shape(D, out_type=tf.int64)
#     R = tf.SparseTensor(indices=tf.cast(indices, tf.int64), values=k*shape[0].eval() * [1/k], dense_shape=shape)
#     # return tf.sparse_tensor_to_dense(R)
#     return indices

def compute_R(D, k=2):
    vals = tf.nn.top_k(-D, k=k).values * -1
    min_kmax = tf.reduce_min(vals, axis=1)
    R = tf.transpose(tf.transpose(D) - min_kmax).eval()
    R[R < 0] = 0
    R[R > 0] = 1
    return R/k


if __name__ == "__main__":
    a = tf.convert_to_tensor([[40, 30, 20, 10], [10, 20, 30, 40]])
    b = tf.nn.top_k(a, 2)
    b.indices


    X = tf.constant([1, 2, 3, 2, 3, 4, 3, 4, 5], shape=[3, 3])
    with tf.Session() as sess:
        # print(sess.run(b))
        # print(sess.run(b.indices))
        print(compute_R(X))
        # print(sess.run(compute_R_2(X)))
        # print(sess.run(compute_R(X, 3)))
        # sampleNum = compute_R(X, 3)
        # print(sess.run(sampleNum[0]))
    # np.random.seed(521)
    # Data = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
    # Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
    # randIdx = np.arange(100)
    # np.random.shuffle(randIdx)
    # trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
    # validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    # testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
