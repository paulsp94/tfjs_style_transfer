'use strict';
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
// import mobilenet from '@tensorflow-models/mobilenet';
require('./lib/initializers.js/index.js');

const cv = require('opencv4nodejs');
const { ImageData, createCanvas } = require('canvas');

const contentLayers = ['block5_conv2'];

const styleLayers = [
  'input_1',
  'block1_conv1',
  'block2_conv1',
  'block3_conv1',
  'block4_conv1',
  'block5_conv1'
];

const loadLocalImage = async filename => {
  const maxDim = 512;

  // load image and rescale
  let img = cv.imread(filename);
  const longDim = Math.max(...img.sizes);
  const scale = maxDim / longDim;
  img = img.rescale(scale);

  // convert your image to rgba color space
  const matRGBA =
    img.channels === 1
      ? img.cvtColor(cv.COLOR_GRAY2RGBA)
      : img.cvtColor(cv.COLOR_BGR2RGBA);

  // create new ImageData from raw mat data
  const imgData = new ImageData(
    new Uint8ClampedArray(matRGBA.getData()),
    img.cols,
    img.rows
  );

  // set canvas dimensions and set image data
  const canvas = createCanvas(img.cols, img.rows);
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imgData, 0, 0);

  return tf.browser.fromPixels(canvas);
};

const vggLayers = async layerNames => {
  const vgg19 = await tf.loadLayersModel(
    `file://${__dirname}/../vgg19-tensorflowjs-model/model/model.json`
  );
  vgg19.trainable = false;

  const outputs = layerNames.map(name => vgg19.getLayer(name).output);

  return tf.model({ inputs: vgg19.input, outputs: outputs });
};

const efficientNetLayers = async layerNames => {
  const efficientNet = await tf.loadGraphModel(
    'file://${__dirname}/../tfjs_efficientnet3_imagenet/model.json'
  );
  efficientNet.trainable = false;
};

(async function() {
  const contentImage = await loadLocalImage('./content.jpg');
  const styleImage = await loadLocalImage('./style.jpg');

  const efficientNet = await tf.loadLayersModel(
    'file://${__dirname}/../../tfjs_efficientnet3_imagenet/model.json'
  );

  console.log(efficientNet);

  // const styleExtractor = await vggLayers(styleLayers);
  // const data = tf.data.FileDataSource('./style.jpg');
  // console.log(styleImage);
  // const styleOutput = styleExtractor.predict(styleImage);
})();
