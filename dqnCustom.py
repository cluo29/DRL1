import tensorflow as tf
import itertools
import numpy as np
import os
import random
import sys
import time
from collections import deque, namedtuple

VALID_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        # Chu input, 3 moments, temperature, power, fan
        # not change power at first, power =2
        self.X_pl = tf.placeholder(shape=[None, 1, 3, 3], dtype=tf.int32, name="X")
        # The TD target value
        # Q value
        ##probably we need to get this at runtime, on the fly!
        # Ground truth
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # X = tf.to_float(self.X_pl) / 255.0
        X = tf.to_float(self.X_pl) / 150.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 14, 1, 1, activation_fn=tf.nn.relu)

        conv3 = tf.contrib.layers.conv2d(
            conv1, 14, 1, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 14)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only,

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)



        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, sess, s):
        """
        Predicts action values.
        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]
        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.
        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]
        Returns:
          The calculated loss on the batch.
        """
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        global_step, _, loss = sess.run(
            [  tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)

        return loss


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)



def executeAction(action,  L, F, T):
    L_Next = 0
    F_Next = 0
    T_Next = T
    if action in [0,3,6]:
        L_Next = L
    elif action in [1,4,7]:
        L_Next = L + 1
    elif action in [2,5,8]:
        L_Next = L - 1
    if L_Next == 3:
        L_Next = 2
    elif L_Next == 0:
        L_Next = 1

    if action in [0, 1, 2]:
        F_Next = F
    elif action in [3,4,5]:
        F_Next = F + 1
    elif action in [6,7,8]:
        F_Next = F -1
    if F_Next == 3:
        F_Next = 2
    elif F_Next == -1:
        F_Next = 0

    if L_Next == 2:
        T_Next = T_Next + 5
    else:
        T_Next = T_Next + 3
    if F_Next ==2:
        T_Next = T_Next - 4
    elif F_Next ==1:
        T_Next = T_Next - 2

    reward = 0


    if T_Next<= 70:
        reward     = reward + L_Next + 153 - F_Next
    else:
        reward = - T_Next

    if T_Next>150:
        T_Next = 150
    elif T_Next<-40:
        T_Next = -40



    reward = reward / 155.0

    return L_Next, F_Next, T_Next, reward


def stateTransit(state, action):
    stateVectorNow = state[:, :, 2]
    #print("state")
    #print(state)
    #print("stateVectorNow")
    #print(stateVectorNow)
    #print("stateVectorNow 1st power L")
    #print(stateVectorNow[0][0])
    L = stateVectorNow[0][0]
    #print("stateVectorNow 2nd fan F")
    #print(stateVectorNow[0][1])
    F = stateVectorNow[0][1]
    #print("stateVectorNow 3rd temperature T")
    #print(stateVectorNow[0][2])
    T = stateVectorNow[0][2]
    #print("action")
    #print(action)
    state_next = state[:, :, 2]
    L_Next, F_Next, T_Next, reward = executeAction(action, L, F, T)
    state_next[0][0] = L_Next
    state_next[0][1] = F_Next
    state_next[0][2] = T_Next
    return state_next, reward


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.
    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


class StateProcessor():
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    """

    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[1, 3], dtype=tf.int32)

            self.output = self.input_state

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State
        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, {self.input_state: state})


def deep_q_learning(sess,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    replay_memory_size=500000,
                    replay_memory_init_size=60,
                    update_target_estimator_every=60,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    discount_factor=0.99,
                    batch_size=32
                    ):
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])

    # replay memory
    replay_memory = []

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # the epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # the policy we're following
    policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))

    # populate the replay memory with initial experience

    # state: 1*3*1   , note that input has more moments of this
    state = tf.placeholder(shape=[None, 1, 3, 3], dtype=tf.int32)
    stateNP = np.random.randint(0, 1, size=(1, 3))
    # test code
    stateNP = np.random.randint(0, 3, size=(1, 3))
    # test code
    state = state_processor.process(sess, stateNP)
    state = np.stack([state] * 3, axis=2)

    state[0][0][0] = 1

    state[0][0][1] = 1

    state[0][0][2] = 1

    state[0][1][0] = 0

    state[0][1][1] = 0

    state[0][1][2] = 0

    state[0][2][0] = 0

    state[0][2][1] = 3

    state[0][2][2] = 6

    # run replay_memory_init_size for test, if in practice, forever,999999
    for i in range(999999999):

        #update target Function
        if total_t % update_target_estimator_every ==0:
            copy_model_parameters(sess, q_estimator, target_estimator)

        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps - 1)])
        print("action_probs")
        print(action_probs)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward = stateTransit(state, action)
        print("now state")
        print(state[:,:,2])
        print("action")
        print(action)
        print("reward")
        print(reward)

        #make vector into matrix
        next_state = np.append(state[:,:,1:],np.expand_dims(next_state,2), axis=2)

        #if out of memory, kill one
        if len(replay_memory) == replay_memory_size:
            replay_memory.pop(0)

        #Save transition to replay memory
        replay_memory.append(Transition(state, action, reward, next_state))

        if i>replay_memory_init_size:

            #get a batch
            samples = random.sample(replay_memory,batch_size)
            states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))

            # do q
            q_values_next = q_estimator.predict(sess, next_states_batch)
            #print("states_batch")
            #print(states_batch)
            #print("prediction")
            #print(q_values_next)
            best_actions = np.argmax(q_values_next, axis = 1)
            #print("best_actions")
            #print(best_actions)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch #+ discount_factor * q_values_next_target[np.arange(batch_size),best_actions]

            #gradient
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            print("loss")
            print(loss)
            total_t += 1

        state = next_state

    return 0

tf.reset_default_graph()

global_step = tf.Variable(0, name='global_step', trainable=False)

# State processor
state_processor = StateProcessor()

q_estimator = Estimator(scope="q")
target_estimator = Estimator(scope="target_q")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    deep_q_learning(sess,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    state_processor=state_processor,
                    replay_memory_size=500000,
                    replay_memory_init_size=600,
                    update_target_estimator_every=600,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    discount_factor=0.99,
                    batch_size=32
                    )
    print("good")




