// const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');
const fs = require('fs');

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
  contentImg = `${__dirname}/assets/content.jpg`;
  styleImg = `${__dirname}/assets/style.jpg`;
  vgg19URL = `file:///${__dirname}/../../vgg19-tensorflowjs-model/model/model.json`;
} else {
  contentImg = 'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg';
  styleImg = 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg';
  vgg19URL = 'https://raw.githubusercontent.com/DavidCai1993/vgg19-tensorflowjs-model/master/model/model.json';
}

const MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3]);

const loadImageToTensor = async (path) => {
  console.info('Loading Image.');
  const img = await Jimp.read(path);
  const scalingFactor = 512 / Math.max(img.bitmap.height, img.bitmap.width);
  img.scale(scalingFactor);
  // img.resize(224, 224);

  const p = [];

  img.scan(0, 0, img.bitmap.width, img.bitmap.height, function (x, y, idx) {
    p.push(this.bitmap.data[idx + 0]);
    p.push(this.bitmap.data[idx + 1]);
    p.push(this.bitmap.data[idx + 2]);
  });

  return tf
    .tensor3d(p, [img.bitmap.width, img.bitmap.height, 3])
    .reshape([1, img.bitmap.width, img.bitmap.height, 3])
    .div(255);
};

const saveImage = (path, tensor) => {
  tf.tidy(() => {
    const newTensor = tensor
      .mul(255)
      .clipByValue(0, 255)
      .reshape(tensor.shape.slice(1));
    const newTensorArray = Array.from(newTensor.dataSync());
    const image = new Jimp(...tensor.shape.slice(1, 3));
    let i = 0;

    image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
      this.bitmap.data[idx + 0] = newTensorArray[i++];
      this.bitmap.data[idx + 1] = newTensorArray[i++];
      this.bitmap.data[idx + 2] = newTensorArray[i++];
      this.bitmap.data[idx + 3] = 255;
    });

    image.getBuffer(Jimp.MIME_PNG, (err, buffer) => {
      if (err) {
        console.error(err);
        return;
      }
      fs.writeFileSync(path, buffer);
    });
  });
};

const vggLayers = async (inputShape) => {
  console.info('Loading Model!');
  const vgg19 = await tf.loadLayersModel(vgg19URL);
  vgg19.trainable = false;

  // const outputs = layerNames.map(name => vgg19.getLayer(name).output);
  // outputs.push(vgg19.getLayer(contentLayer[0]).output);
  inputShape[0] = null;
  vgg19.input.shape = inputShape;

  return tf.model({ inputs: vgg19.input, outputs: vgg19.output });
};

const gramMatrix = inputTensor => tf.tidy(() => {
  const [i, h, w, c] = inputTensor.shape;
  const a = inputTensor.reshape([-1, c]);
  const gram = tf.div(tf.matMul(a, a, true), tf.scalar(h * w));
  return gram;
});

const getLayerResults = (model, image, layers) => {
  const layersCopy = [...layers];
  let currentResult = image;
  const results = [];
  let idx = 1;

  while (layersCopy.length !== 0) {
    const layer = model.getLayer(null, idx++);
    if (layersCopy.includes(layer.name)) {
      results.push(layer.apply(currentResult));
      layersCopy.splice(layersCopy.indexOf(layer.name), 1);
    }
    currentResult = layer.apply(currentResult);
  }

  return results;
};

const featureRepresentation = async (contentPath, stylePath) => {
  const [contentOutputs, contentModel] = await loadImageToTensor(contentPath)
    .then(tensor => tensor
      .cast('float32')
      .mul(255)
      .sub(MEANS))
    .then(tensor => vggLayers(tensor.shape).then(model => [getLayerResults(model, tensor, contentLayer), model]));
  const styleOutputs = await loadImageToTensor(stylePath)
    .then(tensor => tensor
      .cast('float32')
      .mul(255)
      .sub(MEANS))
    .then(tensor => vggLayers(tensor.shape).then(model => getLayerResults(model, tensor, styleLayers)));

  return [styleOutputs, contentOutputs, contentModel];
};

const generateNoiseImage = (image, noiseRatio = 0.6) => tf.tidy(() => {
  const noiseImage = tf.randomUniform(image.shape, -0.2, 0 - 2);
  return noiseImage.mul(noiseRatio).add(image.mul(1 - noiseRatio));
});

const computeStyleLoss = (generatedFeature, gramStyleFeature) => {
  const generatedGramStyle = gramMatrix(generatedFeature);

  return generatedGramStyle
    .sub(gramStyleFeature)
    .pow(2)
    .mean();
};

const computeContentLoss = (generatedOutputs, contentFeatures) => generatedOutputs
  .slice(-1)[0]
  .sub(contentFeatures.slice(-1)[0])
  .pow(2)
  .mean();

const highPassXY = (image) => {
  const xVar = image.sub(image.reverse(1));
  const yVar = image.sub(image.reverse(2));
  return [xVar, yVar];
};

const totalVariationLoss = (image) => {
  const [xDeltas, yDeltas] = highPassXY(image);
  return tf.mean(xDeltas.pow(2)).add(tf.mean(yDeltas.pow(2)));
};

const runStyleTransfer = async (
  contentPath,
  stylePath,
  numIterations = 2000,
  contentWeight = 1e4,
  styleWeight = 1e-2,
  totalVariationWeight = 1e8,
) => {
  // const model = await vggLayers(styleLayers);

  const [styleFeatures, contentFeatures, model] = await featureRepresentation(
    contentPath,
    stylePath,
  );

  const gramStyleFeatures = styleFeatures
    .slice(0, -1)
    .map(styleFeature => gramMatrix(styleFeature));

  const contentImage = await loadImageToTensor(contentPath);

  let bestLoss = Infinity;
  let bestImg;
  let initImage;
  initImage = generateNoiseImage(contentImage, 0).variable();

  const computeLoss = () => tf.tidy(() => {
    const generatedOutputs = getLayerResults(model, initImage.mul(255).sub(MEANS), [
      ...styleLayers,
      ...contentLayer,
    ]);

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
  for (let index = 0; index <= numIterations; index++) {
    const opt = tf.train.adam(0.02, 0.99, 0.999, 1e-1);
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
  }

  console.info(`Total time: ${new Date().getTime() - globalStart}s`);

  return [bestImg, bestLoss];
};

const styleTransfer = async () => {
  const [best, bestLoss] = await runStyleTransfer(contentImg, styleImg, 3000);
};

styleTransfer();
