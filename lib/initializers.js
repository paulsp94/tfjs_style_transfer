"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
exports.__esModule = true;
var initializers_1 = require("@tensorflow/tfjs-layers/src/initializers");
var tfjs_1 = require("@tensorflow/tfjs");
var errors_1 = require("@tensorflow/tfjs-layers/src/errors");
var common_1 = require("@tensorflow/tfjs-layers/src/common");
var math_utils_1 = require("@tensorflow/tfjs-layers/src/utils/math_utils");
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
var EfficientConv2DKernelInitializer = /** @class */ (function (_super) {
    __extends(EfficientConv2DKernelInitializer, _super);
    function EfficientConv2DKernelInitializer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    EfficientConv2DKernelInitializer.prototype.apply = function (shape, dtype) {
        var fans = computeFans(shape);
        var fanOut = fans[1];
        dtype = dtype || 'float32';
        if (dtype !== 'float32' && dtype !== 'int32') {
            throw new errors_1.NotImplementedError(this.getClassName() + " does not support dType " + dtype + ".");
        }
        return tfjs_1.randomNormal(shape, 0.0, Math.sqrt(2.0 / fanOut), dtype);
    };
    EfficientConv2DKernelInitializer.prototype.getConfig = function () {
        return {};
    };
    /** @nocollapse */
    EfficientConv2DKernelInitializer.className = 'EfficientConv2DKernelInitializer';
    return EfficientConv2DKernelInitializer;
}(initializers_1.Initializer));
exports.EfficientConv2DKernelInitializer = EfficientConv2DKernelInitializer;
tfjs_1.serialization.registerClass(EfficientConv2DKernelInitializer);
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
var EfficientDenseKernelInitializer = /** @class */ (function (_super) {
    __extends(EfficientDenseKernelInitializer, _super);
    function EfficientDenseKernelInitializer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    EfficientDenseKernelInitializer.prototype.apply = function (shape, dtype) {
        var initRange = 1.0 / Math.sqrt(shape[1]);
        return tfjs_1.randomUniform(shape, -initRange, initRange, dtype);
    };
    EfficientDenseKernelInitializer.prototype.getConfig = function () {
        return {};
    };
    /** @nocollapse */
    EfficientDenseKernelInitializer.className = 'EfficientDenseKernelInitializer';
    return EfficientDenseKernelInitializer;
}(initializers_1.Initializer));
exports.EfficientDenseKernelInitializer = EfficientDenseKernelInitializer;
tfjs_1.serialization.registerClass(EfficientDenseKernelInitializer);
/**
 * Computes the number of input and output units for a weight shape.
 * @param shape Shape of weight.
 * @param dataFormat data format to use for convolution kernels.
 *   Note that all kernels in Keras are standardized on the
 *   CHANNEL_LAST ordering (even when inputs are set to CHANNEL_FIRST).
 * @return An length-2 array: fanIn, fanOut.
 */
function computeFans(shape, dataFormat) {
    if (dataFormat === void 0) { dataFormat = 'channelsLast'; }
    var fanIn;
    var fanOut;
    common_1.checkDataFormat(dataFormat);
    if (shape.length === 2) {
        fanIn = shape[0];
        fanOut = shape[1];
    }
    else if ([3, 4, 5].indexOf(shape.length) !== -1) {
        if (dataFormat === 'channelsFirst') {
            var receptiveFieldSize = math_utils_1.arrayProd(shape, 2);
            fanIn = shape[1] * receptiveFieldSize;
            fanOut = shape[0] * receptiveFieldSize;
        }
        else if (dataFormat === 'channelsLast') {
            var receptiveFieldSize = math_utils_1.arrayProd(shape, 0, shape.length - 2);
            fanIn = shape[shape.length - 2] * receptiveFieldSize;
            fanOut = shape[shape.length - 1] * receptiveFieldSize;
        }
    }
    else {
        var shapeProd = math_utils_1.arrayProd(shape);
        fanIn = Math.sqrt(shapeProd);
        fanOut = Math.sqrt(shapeProd);
    }
    return [fanIn, fanOut];
}
