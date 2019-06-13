import { Initializer } from '@tensorflow/tfjs-layers/src/initializers';
import {
  DataType,
  Tensor,
  randomNormal,
  randomUniform,
  serialization
} from '@tensorflow/tfjs';
import { NotImplementedError } from '@tensorflow/tfjs-layers/src/errors';
import {
  DataFormat,
  Shape
} from '@tensorflow/tfjs-layers/src/keras_format/common';
import { checkDataFormat } from '@tensorflow/tfjs-layers/src/common';
import { arrayProd } from '@tensorflow/tfjs-layers/src/utils/math_utils';

/**
 * Initialization for convolutional kernels.
 *
 * The main difference with tf.variance_scaling_initializer is that
 * tf.variance_scaling_initializer uses a truncated normal with an uncorrected
 * standard deviation, whereas here we use a normal distribution. Similarly,
 * tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
 * a corrected standard deviation.
 *
 * Returns:
 *   an initialization for the variable
 */
export class EfficientConv2DKernelInitializer extends Initializer {
  /** @nocollapse */
  static className = 'EfficientConv2DKernelInitializer';
  apply(shape: Shape, dtype?: DataType): Tensor {
    const fans = computeFans(shape);
    const fanOut = fans[1];

    dtype = dtype || 'float32';
    if (dtype !== 'float32' && dtype !== 'int32') {
      throw new NotImplementedError(
        `${this.getClassName()} does not support dType ${dtype}.`
      );
    }

    return randomNormal(shape, 0.0, Math.sqrt(2.0 / fanOut), dtype);
  }

  getConfig(): serialization.ConfigDict {
    return {};
  }
}

serialization.registerClass(EfficientConv2DKernelInitializer);

/**
 * Initialization for dense kernels.
 *
 * This initialization is equal to
 *    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
 *                                    distribution='uniform').
 * It is written out explicitly here for clarity.
 *
 *  Args:
 *    shape: shape of variable
 *    dtype: dtype of variable
 *
 *  Returns:
 *    an initialization for the variable
 */
export class EfficientDenseKernelInitializer extends Initializer {
  /** @nocollapse */
  static className = 'EfficientDenseKernelInitializer';
  apply(shape: Shape, dtype?: DataType): Tensor {
    const initRange = 1.0 / Math.sqrt(shape[1]);
    return randomUniform(shape, -initRange, initRange, dtype);
  }

  getConfig(): serialization.ConfigDict {
    return {};
  }
}

serialization.registerClass(EfficientDenseKernelInitializer);

/**
 * Computes the number of input and output units for a weight shape.
 * @param shape Shape of weight.
 * @param dataFormat data format to use for convolution kernels.
 *   Note that all kernels in Keras are standardized on the
 *   CHANNEL_LAST ordering (even when inputs are set to CHANNEL_FIRST).
 * @return An length-2 array: fanIn, fanOut.
 */
function computeFans(
  shape: Shape,
  dataFormat: DataFormat = 'channelsLast'
): number[] {
  let fanIn: number;
  let fanOut: number;
  checkDataFormat(dataFormat);
  if (shape.length === 2) {
    fanIn = shape[0];
    fanOut = shape[1];
  } else if ([3, 4, 5].indexOf(shape.length) !== -1) {
    if (dataFormat === 'channelsFirst') {
      const receptiveFieldSize = arrayProd(shape, 2);
      fanIn = shape[1] * receptiveFieldSize;
      fanOut = shape[0] * receptiveFieldSize;
    } else if (dataFormat === 'channelsLast') {
      const receptiveFieldSize = arrayProd(shape, 0, shape.length - 2);
      fanIn = shape[shape.length - 2] * receptiveFieldSize;
      fanOut = shape[shape.length - 1] * receptiveFieldSize;
    }
  } else {
    const shapeProd = arrayProd(shape);
    fanIn = Math.sqrt(shapeProd);
    fanOut = Math.sqrt(shapeProd);
  }

  return [fanIn, fanOut];
}
