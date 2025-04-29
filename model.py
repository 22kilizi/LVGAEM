from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class GraphAttentionLayer(tf.layers.Layer):
    def __init__(self, in_features, out_features, dropout_rate, alpha, concat=True, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.concat = concat

        # Initialize the weights
        self.W = tf.get_variable('W', [in_features, out_features], initializer=tf.glorot_uniform_initializer())
        self.a = tf.get_variable('a', [2 * out_features, 1], initializer=tf.glorot_uniform_initializer())

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, adj):
        N = tf.shape(inputs)[0]  # Batch size

        # Linear transformation
        h = tf.matmul(inputs, self.W)

        # Attention mechanism
        a_input = tf.concat([h[:, tf.newaxis], h], axis=1)  # Concatenate along the second dimension
        a_input = tf.reshape(a_input, [N, N, 2 * self.out_features])

        e = tf.nn.leaky_relu(tf.matmul(a_input, self.a), alpha=self.alpha)

        # Apply the adjacency matrix and zero out the non-edges
        zero_vec = -1e12 * tf.ones_like(e)
        attention = tf.where(adj > 0, e, zero_vec)

        # Softmax for attention weights
        attention = tf.nn.softmax(attention, axis=1)
        attention = tf.nn.dropout(attention, keep_prob=(1 - self.dropout_rate), training=self.is_training)

        # Matrix multiplication to get the output
        h_prime = tf.matmul(attention, h)

        if self.concat:
            return tf.nn.elu(h_prime)
        else:
            return h_prime


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)  # z_mean is miu

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)  # z_log_std is ethro

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        self.z = self.z_mean * self.att[1] + self.z * self.att[2]  # z=a1*u + a2*z

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.z)
