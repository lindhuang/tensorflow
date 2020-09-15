"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset."""

import pickle
import tensorflow as tf
import numpy as np
import os
import pretty_print
import functions
from symbolic_autoencoder import SymbolicPropagatorUnit, SymbolicMultPropagatorUnit
from regularization import l12_smooth
from inspect import signature


N_TRAIN = 256       # Size of training dataset
N_VAL = 100         # Size of validation dataset
DOMAIN = (-1, 1)    # Domain of dataset
N_TEST = 100        # Size of test dataset
DOMAIN_TEST = (-2, 2)   # Domain of test dataset - should be larger than training domain to test extrapolation
NOISE_SD = 0        # Standard deviation of noise for training dataset
var_names = ["x", "y", "z"]

# Standard deviation of random distribution for weight initializations.
init_sd_first = 0.1
init_sd_last = 1.0
init_sd_middle = 0.5


def generate_data(func, N, range_min=DOMAIN[0], range_max=DOMAIN[1]):
    """Generates datasets."""
    x_dim = len(signature(func).parameters)
    x = (range_max - range_min) * np.random.random([N, x_dim]) + range_min
    y = np.random.normal([[func(*x_i)] for x_i in x], NOISE_SD)
    return x, y


class Benchmark:
    def __init__(self, results_dir):
        """Set hyper-parameters"""
        self.ACTIVATION_FUNCS = [
            functions.Constant(),
            functions.Identity(norm=1.0),
            functions.Square(norm=1.0),
            functions.Pow(norm=1.0, power=3),
            functions.Sin(norm=1.0),
            functions.Sigmoid(norm=1.0),
            functions.Exp()
        ]

        self.DEPTH = 2
        self.REG_WEIGHT = 0.005
        self.LEARNING_RATE = 0.1
        self.SUMMARY_STEP = 1000
        self.TRAINING_STEP1 = 2000
        self.TRAINING_STEP2 = 10000
        self.TRAINING_STEP1 = 100
        self.TRAINING_STEP2 = 100

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir

        self.saved_params = False

    def benchmark(self, func, func_name, trials):
        """Benchmark the given function. Print the results ordered by test error."""
        print("Starting benchmark for function:\t%s" % func_name)
        print("==============================================")

        func_dir = os.path.join(self.results_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        # Save parameters to file (only once)
        if not self.saved_params:
            result = {
                "LEARNING_RATE": self.LEARNING_RATE,
                "SUMMARY_STEP": self.SUMMARY_STEP,
                "TRAINING_STEP1": self.TRAINING_STEP1,
                "TRAINING_STEP2": self.TRAINING_STEP2,
                "ACTIVATION_FUNCS_NAME": [func.name for func in self.ACTIVATION_FUNCS],
                "DEPTH": self.DEPTH,
                "REG_WEIGHT": self.REG_WEIGHT,
            }
            with open(os.path.join(self.results_dir, 'params.pickle'), "wb+") as f:
                pickle.dump(result, f)
            self.saved_params = True

        expr_list, error_test_list = self.run_network(func, trials, func_dir)

        # Sort the results by test error and print them to file
        error_expr_sorted = sorted(zip(error_test_list, expr_list))
        error_test_sorted = [x for x, _ in error_expr_sorted]
        expr_list_sorted = [x for _, x in error_expr_sorted]

        fi = open(os.path.join(self.results_dir, 'eq_summary.txt'), 'a')
        fi.write("\n{}\n".format(func_name))
        for i in range(trials):
            fi.write("[%f]\t\t%s\n" % (error_test_sorted[i], str(expr_list_sorted[i])))
        fi.close()

    def run_network(self, func, trials=1, func_dir='results/test'):
        """Train the network to find a given function"""

        x_dim = len(signature(func).parameters)  # Number of input arguments to the function
        # Generate training data and test data
        x, y = generate_data(func, N_TRAIN)
        # x_val, y_val = generate_data(func, N_VAL)
        x_test, y_test = generate_data(func, N_TEST, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1])

        # Setting up the symbolic regression network
        x_placeholder = tf.placeholder(shape=(None, x_dim), dtype=tf.float32)
        width = len(self.ACTIVATION_FUNCS)
        # sym = SymbolicPropagatorUnit(DEPTH,
        #                              funcs=[func.tf for func in ACTIVATION_FUNCS],
        #                              initial_weights=[
        #                                  tf.truncated_normal([x_dim, width], stddev=init_sd_first),
        #                                  tf.truncated_normal([width, width], stddev=init_sd_middle),
        #                                  tf.truncated_normal([width, 1], stddev=init_sd_last)
        #                              ])
        sym = SymbolicMultPropagatorUnit(self.DEPTH,
                                         funcs=[func.tf for func in self.ACTIVATION_FUNCS],
                                         initial_weights=[
                                             tf.truncated_normal([x_dim, width+2], stddev=init_sd_first),
                                             tf.truncated_normal([width+1, width+2], stddev=init_sd_middle),
                                             tf.truncated_normal([width+1, 1], stddev=init_sd_last)
                                         ])
        y_hat = sym(x_placeholder)

        # Label and errors
        error = tf.losses.mean_squared_error(labels=y, predictions=y_hat)
        error_test = tf.losses.mean_squared_error(labels=y_test, predictions=y_hat)
        # Regularization oscillates as a function of epoch
        l12_penalty = l12_smooth(sym.get_symbolic_weights())
        out_penalty = l12_smooth(sym.get_output_weight())
        epoch = tf.placeholder(tf.float32)
        pump_freq = 0.002
        pump = tf.cos(pump_freq * epoch)**2
        pump2 = tf.cos(1.4142 * pump_freq * epoch)**2
        print(pump)
        loss = error + self.REG_WEIGHT * pump * l12_penalty + self.REG_WEIGHT * pump2 * out_penalty

        # Training
        learning_rate = tf.placeholder(tf.float32)
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train = opt.minimize(loss)

        loss_list = []  # Total loss (MSE + regularization)
        error_list = []     # MSE
        l12_list = []
        error_test_list = []

        error_test_final = []
        eq_list = []

        # Only take GPU memory as needed - allows multiple jobs on a single GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            for trial in range(trials):
                print("Training " + str(trial+1) + " out of " + str(trials))

                loss_val = np.nan
                # Restart training if loss goes to NaN (which happens when gradients blow up)
                while np.isnan(loss_val):
                    sess.run(tf.global_variables_initializer())
                    # 1st stage of training with oscillating regularization weight
                    for i in range(self.TRAINING_STEP1):
                        feed_dict = {x_placeholder: x,
                                     epoch: i,
                                     learning_rate: self.LEARNING_RATE}
                        _ = sess.run(train, feed_dict=feed_dict)
                        if i % self.SUMMARY_STEP == 0:
                            loss_val, error_val, l12_val, = sess.run((loss, error, l12_penalty), feed_dict=feed_dict)
                            error_test_val = sess.run(error_test, feed_dict={x_placeholder: x_test})
                            print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (i, loss_val, error_test_val))
                            loss_list.append(loss_val)
                            error_list.append(error_val)
                            l12_list.append(l12_val)
                            error_test_list.append(error_test_val)
                            if np.isnan(loss_val):  # If loss goes to NaN, restart training
                                break
                    # 2nd stage of training with no oscillation
                    for i in range(self.TRAINING_STEP2):
                        feed_dict = {x_placeholder: x,
                                     epoch: 0, #do eopch is i
                                     learning_rate: self.LEARNING_RATE/10}
                        _ = sess.run(train, feed_dict=feed_dict)
                        if i % self.SUMMARY_STEP == 0:
                            loss_val, error_val, l12_val, = sess.run((loss, error, l12_penalty), feed_dict=feed_dict)
                            error_test_val = sess.run(error_test, feed_dict={x_placeholder: x_test})
                            print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (i, loss_val, error_test_val))
                            loss_list.append(loss_val)
                            error_list.append(error_val)
                            l12_list.append(l12_val)
                            error_test_list.append(error_test_val)
                            if np.isnan(loss_val):  # If loss goes to NaN, restart training
                                break

                # Print the expressions
                weights = sess.run(sym.get_weights())
                expr = pretty_print.network(weights, [func.sp for func in self.ACTIVATION_FUNCS], var_names[:x_dim], mult=True)
                print(expr)

                # Save results
                trial_file = os.path.join(func_dir, 'trial%d.pickle' % trial)

                results = {
                    "weights": weights,
                    "loss_list": loss_list,
                    "error_list": error_list,
                    "l12_list": l12_list,
                    "error_test": error_test_list,
                    "expr": expr
                }

                with open(trial_file, "wb+") as f:
                    pickle.dump(results, f)

                error_test_final.append(error_test_list[-1])
                eq_list.append(expr)

        return eq_list, error_test_final


#if __name__ == "__main__":

#    bench = Benchmark(results_dir='results/benchmark/test')
#    bench.benchmark(lambda a: a, func_name="a", trials=2)
#    bench.benchmark(lambda a: a**2, func_name="a^2", trials=2)
    # bench.benchmark(lambda a: a**3, func_name="a^3", trials=2)
    # bench.benchmark(lambda a: np.sin(2*np.pi*a), func_name="sin(2*pi*a)", trials=20)
    # bench.benchmark(lambda a: np.exp(a), func_name="e^a", trials=20)
    # bench.benchmark(lambda a, b: a*b, func_name="a*b", trials=20)
