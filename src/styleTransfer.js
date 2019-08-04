// const tf = require('@tensorflow/tfjs');
import * as tf from '@tensorflow/tfjs-node';

import { loadImageToTensor, saveImage } from './ioUtils';

const local = true;

const contentLayer = ['block5_conv2'];
const styleLayers = [
  'block1_conv1',
  'block2_conv1',
  'block3_conv1',
  'block4_conv1',
  'block5_conv1',
];

const numStyleLayers = styleLayers.length;

let contentImg;
let styleImg;
let vgg19URL;

if (local) {
  contentImg = `${__dirname}/../assets/content.jpg`;
  styleImg = `${__dirname}/../assets/style.jpg`;
  vgg19URL = `file:///${__dirname}/../../vgg19-tensorflowjs-model/model/model.json`;
} else {
  contentImg = 'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg';
  styleImg = 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg';
  vgg19URL = 'https://raw.githubusercontent.com/DavidCai1993/vgg19-tensorflowjs-model/master/model/model.json';
}

const MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3]);

// type ShapeMap = { [name: string]: [number, number, number, number] };

// Source: https://github.com/tensorflow/tfjs/issues/477#issuecomment-403523104
const mapInputShapes = (model, newShapeMap) => {
  const cfg = { ...model.getConfig() };
  cfg.layers = cfg.layers.map((l) => {
    if (l.name in newShapeMap) {
      return {
        ...l,
        config: {
          ...l.config,
          batchInputShape: newShapeMap[l.name],
        },
      };
    }
    return l;
  });

  const map = tf.serialization.SerializationMap.getMap().classNameMap;
  const [cls, fromConfig] = map[model.getClassName()];

  return fromConfig(cls, cfg);
};

const vggLayers = async (inputShape, layerNames) => {
  console.info('Loading Model!');
  const vgg19 = await tf.loadLayersModel(vgg19URL);
  vgg19.trainable = false;

  const newInputShape = [...inputShape];
  newInputShape[0] = null;
  const newConfig = mapInputShapes(vgg19, { input_1: inputShape });

  const outputs = layerNames.map(name => newConfig.getLayer(name).output);
  tf.dispose(vgg19);

  return tf.model({ inputs: newConfig.input, outputs });
};

const gramMatrix = inputTensor => tf.tidy(() => {
  const [i, h, w, c] = inputTensor.shape;
  const a = inputTensor.reshape([-1, c]);
  const gram = tf.div(tf.matMul(a, a, true), tf.scalar(h * w));
  return gram;
});

const featureRepresentation = async (contentPath, stylePath) => {
  const [contentOutputs, contentShape] = await loadImageToTensor(contentPath)
    .then(tensor => tf.tidy(() => tensor
      .cast('float32')
      .mul(255)
      .sub(MEANS)))
    .then(tensor => vggLayers(tensor.shape, contentLayer).then((model) => {
      const results = model.predict(tensor);
      tf.dispose(model);
      return [results, tensor.shape];
    }));
  const styleOutputs = await loadImageToTensor(stylePath)
    .then(tensor => tf.tidy(() => tensor
      .cast('float32')
      .mul(255)
      .sub(MEANS)))
    .then(tensor => vggLayers(tensor.shape, styleLayers).then((model) => {
      const results = model.predict(tensor);
      tf.dispose(model);
      return results;
    }));

  return [styleOutputs, contentOutputs, contentShape];
};

const generateNoiseImage = (image, noiseRatio = 0.6) => tf.tidy(() => {
  const noiseImage = tf.randomUniform(image.shape, -0.2, 0.2);
  return noiseImage.mul(noiseRatio).add(image.mul(1 - noiseRatio));
});

const computeStyleLoss = (generatedFeature, gramStyleFeature) => tf.tidy(() => gramMatrix(generatedFeature)
  .sub(gramStyleFeature)
  .pow(2)
  .mean());

const computeContentLoss = (generatedOutputs, contentFeatures) => tf.tidy(() => generatedOutputs
  .slice(-1)[0]
  .sub(contentFeatures)
  .pow(2)
  .mean());

const highPassXY = image => tf.tidy(() => {
  const xVar = image.sub(image.reverse(1));
  const yVar = image.sub(image.reverse(2));
  return [xVar, yVar];
});

const totalVariationLoss = image => tf.tidy(() => {
  const [xDeltas, yDeltas] = highPassXY(image);
  return tf.mean(xDeltas.pow(2)).add(tf.mean(yDeltas.pow(2)));
});

const runStyleTransfer = async (
  contentPath,
  stylePath,
  numIterations = 2000,
  contentWeight = 1e4,
  styleWeight = 1e-2,
  totalVariationWeight = 1e8,
) => {
  const [styleFeatures, contentFeatures, contentShape] = await featureRepresentation(
    contentPath,
    stylePath,
  );
  const model = await vggLayers(contentShape, [...styleLayers, ...contentLayer]);
  const gramStyleFeatures = styleFeatures.map(styleFeature => gramMatrix(styleFeature));
  const contentImage = await loadImageToTensor(contentPath);
  
  let bestLoss = Infinity;
  let bestImg;
  let initImage;
  initImage = generateNoiseImage(contentImage).variable();
  
  const computeLoss = () => tf.tidy(() => {
    const generatedOutputs = model.predict(initImage.mul(255).sub(MEANS));

    const styleScore = tf
      .addN(
        gramStyleFeatures.map((target, idx) => computeStyleLoss(generatedOutputs[idx], target)),
      )
      .mul(styleWeight / numStyleLayers);

    const contentScore = computeContentLoss(generatedOutputs, contentFeatures).mul(contentWeight);

    const tvScore = totalVariationLoss(initImage).mul(totalVariationWeight);

    return tf.addN([styleScore, contentScore, tvScore]);
  });

  let startTime = new Date().getTime();
  const globalStart = new Date().getTime();

  let loss;

  const step = (index, opt) => tf.tidy(() => {
    [loss] = opt.minimize(() => computeLoss(), true).dataSync();
    initImage = initImage.clipByValue(0, 1).variable();
    if (bestLoss > loss) {
      bestLoss = loss;
      bestImg = initImage.clone();
    }

    if (index % 100 === 0) {
      saveImage(`./results/${index}_${bestLoss}.jpg`, initImage);
      console.log(tf.memory());
      console.info(`Iteration: ${index}`);
      console.info(
        `Total loss: ${loss}, best loss: ${bestLoss}, time: ${new Date().getTime() - startTime}`,
      );
      startTime = new Date().getTime();
    }
  });

  for (let index = 0; index <= numIterations; index++) {
    const opt = tf.train.adam(0.02, 0.99, 0.999, 1e-1);
    step(index, opt); // +61 Tensors
    tf.dispose(opt);
    // one step +1 Tensor
  }

  console.info(`Total time: ${new Date().getTime() - globalStart}s`);

  return [bestImg, bestLoss];
};

const styleTransfer = async () => {
  const [best, bestLoss] = await runStyleTransfer(contentImg, styleImg, 3000);
};

styleTransfer();
