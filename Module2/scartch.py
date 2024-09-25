import tensorflow as tf

# A session object encapsulates the environment in which Tensor objects are evaluated.
with tf.compat.v1.Session() as sess:
    # TensorFlow returns a reference to the desired tensor
    # rather than the value of the tensor itself. To force
    # the value of the tensor to be returned, use the method
    # tf.Tensor.eval() of tensor objects
    a = tf.zeros(2)
    print(a.eval())
    print()
    # [0. 0.]

    b = tf.zeros((2, 3))
    print(b.eval())
    print()
    # [[0. 0. 0.]
    # [0. 0. 0.]]

    # A tensor can be filled with values you
    # desire using the tensor fill command:

    c = tf.fill((2, 2, 2), 5)
    print(c.eval())
    print()
    #[[[5 5]
    # [5 5]]

    # [[5 5]
    # [5 5]]]


    # “tf.constant” is another function, similar to
    # tf.fill, which allows for construction of tensors
    # that shouldn’t change during the program execution:

    d = tf.constant(3)
    print(d.eval())
    print()
    # 3

    #In machine learning, neural network weights
    # (the nodes) are initialized with random values
    # from tensors rather than constants. One way to
    # do this would be to select a random value from
    # a normal distribution with a given mean and standard
    # deviation.

    e = tf.random.normal((2, 2), mean=0, stddev=1)
    print(e.eval())
    print()
    # [[-0.59414846  0.03326153]
    # [-0.8353475  -0.9765181 ]]

    # In more complex machine learning programs using normal
    # distributions for tensors, numerical instability can arise
    # if a random value is far from the mean. To compensate for
    # this, in order to get to convergence, use tf.truncated_normal().
    # The function will resample values from more than two standard
    # deviations from the mean.

    f = tf.random.uniform((2, 2), minval=-2, maxval=2)
    print(f.eval())
    print()
    # [[ 1.0133305  1.447001 ]
    # [-1.979507  -1.5288396]]

    # Addition and scalar multiplication apply to tensors:
    g = tf.ones(2, 2)
    h = tf.ones(2, 2)
    i = g + h
    print(i.eval())
    print()
    # [2. 2.]

    j = i * 2
    print(j.eval())
    print()
    # [4. 4.]

    # Tensor multiplication occurs elementwise and not matrix multiplication:
    k = tf.fill((2,2), 2.)
    l = tf.fill((2,2), 7.)
    m = k * l
    print(m.eval())
    print()
    # [[14. 14.]
    # [14. 14.]]

    # Identity matrices are square matrices that are 0
    # everywhere except on the diagonal where they are 1.
    # tf.eye() allows for fast construction of identity
    # matrices of desired size.

    n = tf.eye(4)
    print(n.eval())
    print()
    # [[1. 0. 0. 0.]
    # [0. 1. 0. 0.]
    # [0. 0. 1. 0.]
    # [0. 0. 0. 1.]]

    # Diagonal matrices are another common type of matrix.
    # Like identity matrices, diagonal matrices are only nonzero
    # along the diagonal. Unlike identity matrices, they may take
    # arbitrary values along the diagonal. The easiest way for doing
    # this is to invoke tf.range(start, limit, delta).
    # The resulting vector can then be fed to
    # tf.diag(diagonal), which will construct a matrix with the
    # specified diagonal.

    o = tf.range(1, 5, 1)
    print(o.eval())
    print()
    # [1 2 3 4]

    p = tf.linalg.diag(o)
    print(p.eval())
    print()
    # [[1 0 0 0]
    # [0 2 0 0]
    # [0 0 3 0]
    # [0 0 0 4]]

    # Use tf.matrix_transpose() to take the transpose of a matrix:
    q = tf.ones((2, 3))
    print(q.eval())
    print()
    # [[1. 1. 1.]
    # [1. 1. 1.]]

    r = tf.transpose(q)
    print(r.eval())
    print()
    # [[1. 1.]
    # [1. 1.]
    # [1. 1.]]

    #To multiply two matrices, use tf.matmul.
    s = tf.ones((2, 3))
    t = tf.ones((3, 4))
    u = tf.matmul(s, t)

    print(s.eval())
    print()
    # [[1. 1. 1.]
    # [1. 1. 1.]]

    print(t.eval())
    print()
    # [[1. 1. 1. 1.]
    # [1. 1. 1. 1.]
    # [1. 1. 1. 1.]]

    print(u.eval())
    print()
    # [[3. 3. 3. 3.]
    # [3. 3. 3. 3.]]

    #tf.reshape() allows tensors to be converted into tensors with different shapes:
    v = tf.ones(8)
    print(v.eval())
    print()
    # [1. 1. 1. 1. 1. 1. 1. 1.]

    w = tf.reshape(v, (2, 4))
    print(w.eval())
    print()
    # [[1. 1. 1. 1.]
    # [1. 1. 1. 1.]]

    # Broadcasting is a term (introduced by NumPy) for when a tensor
    # system’s matrices and vectors of different sizes can be added together.
    # These rules allow for conveniences like adding a vector to every row of
    # a matrix.
    x = tf.ones((2, 2))
    print(x.eval())
    print()
    # [[1. 1.]
    # [1. 1.]]

    y = tf.range(0, 2, 1, dtype=tf.float32)
    print(y.eval())
    print()
    # [0. 1.]

    z = x + y
    print(z.eval())
    print()
    # [[1. 2.]
    # [1. 2.]]

    # In this case vector y is added to every row of matrix x.
    # Be sure to set the dtype when doing this, because, otherwise,
    # Python will report an error.