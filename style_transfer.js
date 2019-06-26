'use strict';
// const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');

const Jimp = require('jimp');

const contentLayer = 'block5_conv2';
const styleLayers = [
  'block1_conv1',
  'block2_conv1',
  'block3_conv1',
  'block4_conv1',
  'block5_conv1'
];

const numContentLayers = contentLayer.length;
const numStyleLayers = styleLayers.length;

const contentImg =
  'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg';
const styleImg =
  'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg';
const vgg19URL =
  'https://raw.githubusercontent.com/DavidCai1993/vgg19-tensorflowjs-model/master/model/model.json';

const MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3]);
// const MEANS = [123.68, 116.779, 103.939];

const loadImageToTensor = async path => {
  console.info('Loading Image.');
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
  // TODO: Do I need to clip?
  // .clipByValue(-MEANS, 255 - MEANS);
};

// TODO: add dimension check for input
const deprocessImg = async processedImg => {
  return await processedImg
    .add(MEANS)
    .clipByValue(0, 255)
    .toInt().data;
};

const vggLayers = async layerNames => {
  console.info('Loading Model!');
  const vgg19 = await tf.loadLayersModel(vgg19URL);
  vgg19.trainable = false;

  const outputs = layerNames.map(name => vgg19.getLayer(name).output);
  outputs.push(vgg19.getLayer(contentLayer).output);

  return tf.model({ inputs: vgg19.input, outputs: outputs });
};

const contentLoss = (baseContent, targetContent) => {
  return tf.mean(tf.square(baseContent - targetContent));
};

const gramMatrix = inputTensor => {
  console.log(inputTensor);
  const channels = inputTensor.shape.pop();
  const a = inputTensor.reshape([-1, channels]);
  const n = a.shape[0];
  const gram = tf.matMul(a, a, true);
  return gram / tf.cast(n, 'float32');
};

const styleLoss = (baseStyle, gramTarget) => {
  const gramStyle = gramMatrix(baseStyle);
  return tf.mean(tf.square(gramStyle - gramTarget));
};

const featureRepresentation = async (model, contentPath, stylePath) => {
  const contentImage = await loadImageToTensor(contentPath);
  const styleImage = await loadImageToTensor(stylePath);

  const contentOutputs = await model.predict(contentImage);
  const styleOutputs = await model.predict(styleImage);

  const styleFeatures = styleOutputs.slice(0, numStyleLayers);
  const contentFeatures = contentOutputs.slice(numStyleLayers);

  return [styleFeatures, contentFeatures];
};

const computeLoss = async (
  model,
  styleWeight,
  contentWeight,
  initImage,
  gramStyleFeatures,
  contentFeatures
) => {
  console.log(model);
  const modelOutputs = await model.predict(initImage);

  const styleOutputFeatures = modelOutputs.slice(0, numStyleLayers);
  const contentOutputFeatures = modelOutputs.slice(numStyleLayers);

  let styleScore = 0;
  let contentScore = 0;

  const weightPerStyleLayer = 1.0 / parseFloat(numStyleLayers);
  gramStyleFeatures.map((targetStyle, i) => {
    let combStyle = styleOutputFeatures[i];
    styleScore += weightPerStyleLayer * getStyleLoss(combStyle[0], targetStyle);
  });

  const weightPerContentLayer = 1.0 / parseFloat(numContentLayers);
  contentFeatures.map((targetContent, i) => {
    let combContent = contentOutputFeatures[i];
    contentScore +=
      weightPerContentLayer * getContentLoss(combContent[0], targetContent);
  });

  styleScore *= styleWeight;
  contentScore *= contentWeight;
  const loss = styleScore + contentScore;

  return [loss, styleScore, contentScore];
};

const runStyleTransfer = async (
  contentPath,
  stylePath,
  numIterations = 1000,
  contentWeight = 1e3,
  styleWeight = 1e-2
) => {
  const model = await vggLayers(styleLayers);

  const [styleFeatures, contentFeatures] = await featureRepresentation(
    model,
    contentPath,
    stylePath
  );
  const gramStyleFeatures = styleFeatures.map(styleFeature =>
    gramMatrix(styleFeature)
  );

  const initImage = await loadImageToTensor(contentPath);
  const opt = tf.train.adam(5, 0.99, 1e-1);

  let bestLoss,
    bestImg = Infinity,
    None;

  const cfg = {
    model: model,
    styleWeight: styleWeight,
    contentWeight: contentWeight,
    initImage: initImage,
    gramStyleFeatures: gramStyleFeatures,
    contentFeatures: contentFeatures
  };

  const numRows = 2;
  const numCols = 5;
  const displayInterval = numIterations / (numRows * numCols);
  let startTime = new Date().getTime();
  const globalStart = new Date().getTime();

  let imgs = [];
  let plotImg;
  let grads, allLoss;
  for (let index = 0; index < numIterations; index++) {
    [grads, allLoss] = computeGrads(cfg);
    loss, styleScore, (contentScore = allLoss);
    opt.applyGradients([(grads, initImage)]);
    endTime = new Date().getTime();
    if (loss < bestLoss) {
      bestLoss = loss;
      bestImg = deprocess_img(initImage);
    }

    if (index % displayInterval === 0) {
      startTime = new Date().getTime();
      plotImg = deprocess_img(initImage);
      imgs.append(plotImg);
      console.info(`Iteration: ${index}`);
      console.info(
        `Total loss: ${loss}, style loss: ${styleScore}, content loss: ${contentScore}, time: ${new Date().getTime() -
          startTime}`
      );
    }
  }

  console.info(`Total time: ${new Date().getTime() - globalStart}s`);

  return [bestImg, bestLoss];
};

const computeGrads = cfg => {
  const allLoss = computeLoss(cfg);
  const totalLoss = allLoss[0];
  return [tf.grads(totalLoss, cfg['init_image']), allLoss];
};

const styleTransfer = async () => {
  const [best, bestLoss] = await runStyleTransfer(contentImg, styleImg, 1000);
};

styleTransfer();
// const model = await vggLayers(styleLayers);
// console.log('Model: ', model);

// const tensorContent = await loadImageToTensor(contentImg);
// const tensorStyle = await loadImageToTensor(styleImg);
// console.log('Tensor: ', tensor);

// const styleFeatures, contentFeatures = featureRepresentation(model, contentImg, styleImg);

// const result = model.predict(tensor);
// console.log('Result: ', result);
