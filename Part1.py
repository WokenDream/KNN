import tensorflow as tf

def compute_pairwise_distance(X, Z):
    # explicitly write out the expression of pairwise distance of X and Z
    # => (x-d)^2 = x^2 - 2xd + d^2

    # compute X^2 and sum all the columns (i.e. along rows)
    X_2 = tf.reduce_sum(tf.square(X), axis=1)
    # compute Z^2 and sum all the columns
    Z_2 = tf.reduce_sum(tf.square(Z), axis=1)
    # compute -2XZ
    XZ = tf.matmul(X, Z, transpose_b=True)
    XZ = tf.scalar_mul(-2, XZ)

    # compute the matrix equivalent equation of (x-d)^2
    # since broadcasting only tile each row, we need to transpose before adding X_2 and then transpose back
    result = tf.transpose(tf.transpose(XZ + Z_2) + X_2)
    # with tf.Session() as sess:
    #     print(sess.run(result))
    #     print("X_2 reduced:")
    #     print(sess.run(X_2))
    #     print()
    #     print("Z_2 reduced:")
    #     print(sess.run(Z_2))
    #     print()
    #     print("XZ reduced:")
    #     print(sess.run(XZ))
    return result


if __name__ == "__main__":
    X = tf.constant([1, 1, 1, 2, 2, 2, 3, 3, 3], shape=[3, 3])

    Z = tf.constant([4, 4, 4, 5, 5, 5], shape=[2, 3])
    compute_pairwise_distance(X, Z)
    # print(compute_pairwise_distance(X, Z))