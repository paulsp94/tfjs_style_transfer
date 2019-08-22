import * as tf from '@tensorflow/tfjs-node';

export const MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3]);
