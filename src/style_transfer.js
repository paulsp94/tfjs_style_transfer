// const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');
const savePixels = require('save-pixels');
const ndarray = require('ndarray');
const fs = require('fs');

const local = true;

const contentLayer = 'block5_conv2';
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
  img.resize(224, 224);

  const p = [];

  img.scan(0, 0, img.bitmap.width, img.bitmap.height, function (x, y, idx) {
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

const saveImage = (path, tensor) => {
  const newTensor = tensor.add(MEANS).reshape([224, 224, 3]);
  const newTensorArray = Array.from(newTensor.dataSync());
  // const imgFile = fs.createWriteStream(path);
  const i = 0;

  // newTensorArray = ndarray(newTensorArray, [224, 224, 3]);

  // savePixels(newTensorArray, 'jpg')
  //   .on('data', chunk => imgFile.write(chunk))
  //   .on('end', () => console.log('end pixel stream'))
  //   .on('finish', () => console.log('finish pixel stream'));

  // eslint-disable-next-line no-new
  new Jimp({ data: Buffer.from(newTensorArray), width: 224, height: 224 }, (err, image) => {
    console.log(err, image);
    // image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
    //   this.bitmap.data[idx + 0] = newTensorArray[i++];
    //   this.bitmap.data[idx + 1] = newTensorArray[i++];
    //   this.bitmap.data[idx + 2] = newTensorArray[i++];
    //   this.bitmap.data[idx + 3] = 255;
    // });

    image.write(path, () => console.log('Finished writing Image: ', path));
  });
};

const vggLayers = async (layerNames) => {
  console.info('Loading Model!');
  const vgg19 = await tf.loadLayersModel(vgg19URL);
  vgg19.trainable = false;

  const outputs = layerNames.map(name => vgg19.getLayer(name).output);
  outputs.push(vgg19.getLayer(contentLayer).output);

  return tf.model({ inputs: vgg19.input, outputs });
};

const gramMatrix = inputTensor => tf.tidy(() => {
  const channels = inputTensor.shape.slice(-1);
  const a = inputTensor.reshape([-1, channels]);
  const gram = tf.div(tf.matMul(a, a, true), inputTensor.size);
  return gram;
});

const styleLoss = (baseStyle, gramTarget) => tf.tidy(() => tf.sum(tf.square(tf.sub(baseStyle, gramTarget))));
const contentLoss = (baseContent, targetContent) => tf.tidy(() => tf.div(tf.sum(tf.square(tf.sub(baseContent, targetContent))), tf.scalar(2)));

const featureRepresentation = async (model, contentPath, stylePath) => {
  const contentImage = await loadImageToTensor(contentPath);
  const styleImage = await loadImageToTensor(stylePath);

  const contentOutputs = await model.predict(contentImage);
  const styleOutputs = await model.predict(styleImage);

  return [styleOutputs, contentOutputs];
};

const generateNoiseImage = (image, noiseRatio = 0.6) => tf.tidy(() => {
  const noiseImage = tf.randomUniform([1, 224, 224, 3], -20, 20);
  return noiseImage.mul(noiseRatio).add(image.mul(1 - noiseRatio));
});

const runStyleTransfer = async (
  contentPath,
  stylePath,
  numIterations = 2000,
  contentWeight = 1,
  styleWeight = 4,
) => {
  const model = await vggLayers(styleLayers);

  const [styleFeatures, contentFeatures] = await featureRepresentation(
    model,
    contentPath,
    stylePath,
  );

  const gramStyleFeatures = styleFeatures.map(styleFeature => gramMatrix(styleFeature));

  const contentImage = await loadImageToTensor(contentPath);

  let bestLoss = Infinity;
  let bestImg;
  let initImage;
  initImage = generateNoiseImage(contentImage, 1).variable();

  const computeLoss = () => tf.tidy(() => {
    const generatedOutputs = model.predict(initImage);
    const gramGeneratedFeatures = generatedOutputs.map(generatedFeature => gramMatrix(generatedFeature));

    let styleScore = tf.scalar(0);
    const weightPerStyleLayer = 1.0 / parseFloat(numStyleLayers);
    gramStyleFeatures.forEach((targetStyle, i) => {
      styleScore = tf.add(
        styleScore,
        tf.mul(weightPerStyleLayer, styleLoss(gramGeneratedFeatures[i], targetStyle)),
      );
    });

    let contentScore = contentLoss(generatedOutputs.slice(-1)[0], contentFeatures.slice(-1)[0]);

    styleScore = styleScore.mul(tf.scalar(styleWeight));
    contentScore = contentScore.mul(tf.scalar(contentWeight));
    const loss = tf.add(styleScore, contentScore);

    return loss;
  });

  let startTime = new Date().getTime();
  const globalStart = new Date().getTime();

  let loss;
  for (let index = 0; index <= numIterations; index++) {
    const opt = tf.train.adam(0.02, 0.99, 1e-1);
    opt.minimize(() => computeLoss(), true, [initImage]);
    loss = await computeLoss().array();
    if (bestLoss > loss) {
      bestLoss = loss;
      bestImg = initImage.clone();
    }

    if (index % 100 === 0) {
      await saveImage(`./results/${index}_${bestLoss}.jpg`, bestImg);
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
