'use strict';

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

var tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
// import mobilenet from '@tensorflow-models/mobilenet';
require('./lib/initializers.ts');

var cv = require('opencv4nodejs');

var _require = require('canvas'),
    ImageData = _require.ImageData,
    createCanvas = _require.createCanvas;

var contentLayers = ['block5_conv2'];

var styleLayers = ['input_1', 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'];

var loadLocalImage = async function loadLocalImage(filename) {
  var maxDim = 512;

  // load image and rescale
  var img = cv.imread(filename);
  var longDim = Math.max.apply(Math, _toConsumableArray(img.sizes));
  var scale = maxDim / longDim;
  img = img.rescale(scale);

  // convert your image to rgba color space
  var matRGBA = img.channels === 1 ? img.cvtColor(cv.COLOR_GRAY2RGBA) : img.cvtColor(cv.COLOR_BGR2RGBA);

  // create new ImageData from raw mat data
  var imgData = new ImageData(new Uint8ClampedArray(matRGBA.getData()), img.cols, img.rows);

  // set canvas dimensions and set image data
  var canvas = createCanvas(img.cols, img.rows);
  var ctx = canvas.getContext('2d');
  ctx.putImageData(imgData, 0, 0);

  return tf.browser.fromPixels(canvas);
};

var vggLayers = async function vggLayers(layerNames) {
  var vgg19 = await tf.loadLayersModel('file://' + __dirname + '/../vgg19-tensorflowjs-model/model/model.json');
  vgg19.trainable = false;

  var outputs = layerNames.map(function (name) {
    return vgg19.getLayer(name).output;
  });

  return tf.model({ inputs: vgg19.input, outputs: outputs });
};

var efficientNetLayers = async function efficientNetLayers(layerNames) {
  var efficientNet = await tf.loadGraphModel('file://${__dirname}/../tfjs_efficientnet3_imagenet/model.json');
  efficientNet.trainable = false;
};

(async function () {
  var contentImage = await loadLocalImage('./content.jpg');
  var styleImage = await loadLocalImage('./style.jpg');

  var efficientNet = await tf.loadLayersModel('file://${__dirname}/../../tfjs_efficientnet3_imagenet/model.json');

  console.log(efficientNet);

  // const styleExtractor = await vggLayers(styleLayers);
  // const data = tf.data.FileDataSource('./style.jpg');
  // console.log(styleImage);
  // const styleOutput = styleExtractor.predict(styleImage);
})();