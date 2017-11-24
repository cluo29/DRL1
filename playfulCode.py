


fc1 = tf.contrib.layers.fully_connected(input, num_outputs=neurons, activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        biases_initializer=tf.constant_initializer(0.1))