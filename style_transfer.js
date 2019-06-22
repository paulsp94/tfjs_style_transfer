'use strict';
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const Jimp = require('jimp');

const contentLayer = 'block5_conv2';
const styleLayers = [
  'block1_conv1',
  'block2_conv1',
  'block3_conv1',
  'block4_conv1',
  'block5_conv1'
];

numContentLayers = contentLayers.length;
numStyleLayers = styleLayers.length;

const contentImg =
  'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg';
const styleImg =
  'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg';
const vgg19URL =
  'https://raw.githubusercontent.com/DavidCai1993/vgg19-tensorflowjs-model/master/model/model.json';

const MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3]);

const loadImageToTensor = async path => {
  let img = await Jimp.read(path);
  img.resize(224, 224);

  const p = [];

  img.scan(0, 0, img.bitmap.width, img.bitmap.height, function(x, y, idx) {
    p.push(this.bitmap.data[idx + 0]);
    p.push(this.bitmap.data[idx + 1]);
    p.push(this.bitmap.data[idx + 2]);
  });

  return tf
    .tensor3d(p, [224, 224, 3])
    .reshape([1, 224, 224, 3])
    .sub(MEANS);
};

const vggLayers = async layerNames => {
  const vgg19 = await tf.loadLayersModel(
    `file://${__dirname}/../vgg19-tensorflowjs-model/model/model.json`
  );
  vgg19.trainable = false;

  const outputs = layerNames.map(name => vgg19.getLayer(name).output);
  outputs.push(vgg19.getLayer(contentLayer).output);

  console.log(outputs.map(st => st.name));

  return tf.model({ inputs: vgg19.input, outputs: outputs });
};

const contentLoss = (baseContent, targetContent) => {
  return tf.mean(tf.square(baseContent - targetContent));
};

const gramMatrix = inputTensor => {
  channels = inputTensor.shape.pop();
  a = inputTensor.reshape([-1, channels]);
  n = tf.shape(a)[0];
  gram = tf.matMul(a, a, (transposeA = true));
  return gram / tf.cast(n, tf.float32);
};

const styleLoss = (baseStyle, gramTarget) => {
  gramStyle = gramMatrix(baseStyle);
  return tf.mean(tf.square(gramStyle - gramTarget));
};

const featureRepresentation = (model, contentPath, stylePath) => {
  contentImage = loadImageToTensor(contentPath);
  styleImage = loadImageToTensor(stylePath);

  styleOutputs = model(styleImage);
  contentOutputs = model(contentImage);

  styleFeatures = styleOutputs
    .slice(0, numStyleLayers)
    .map(styleLayer => styleLayer[0]);
  contentFeatures = contentOutputs
    .slice(numStyleLayers)
    .map(contentLayer => contentLayer[0]);
  return styleFeatures, contentFeatures;
};

const computeLoss = (
  model,
  lossWeights,
  initImage,
  gramStyleFeatures,
  contentFeatures
) => {
  const styleWeight, contentWeight = lossWeights;
  const modelOutputs = model(initImage)
  
  const styleOutputFeatures = modelOutputs.slice(0, numStyleLayers);
  const contentOutputFeatures = modelOutputs.slice(numStyleLayers);
  
  let styleScore = 0;
  let contentScore = 0;

  const weightPerStyleLayer = 1.0 / parseFloat(numStyleLayers);
  gramStyleFeatures.map((targetStyle, i) => {
    let combStyle = styleOutputFeatures[i];
    styleScore += weightPerStyleLayer * getStyleLoss(combStyle[0], targetStyle);
  })
    
  const weightPerContentLayer = 1.0 / parseFloat(numContentLayers);
  contentFeatures.map((targetContent, i) => {
    let combContent = contentOutputFeatures[i];
    contentScore += weightPerContentLayer* getContentLoss(combContent[0], targetContent)
  })
  
  styleScore *= styleWeight
  contentScore *= contentWeight
  const loss = styleScore + contentScore; 
  
  return loss, styleScore, contentScore
};

const computeGrads = cfg => {
}

(async function() {
  const model = await vggLayers(styleLayers);
  // console.log('Model: ', model);

  const tensor = await loadImageToTensor(contentImg);
  // console.log('Tensor: ', tensor);

  const result = model.predict(tensor);
  // console.log('Result: ', result);
})();
