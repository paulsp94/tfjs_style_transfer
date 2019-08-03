import * as tf from '@tensorflow/tfjs-node';
import Jimp from 'jimp';
import { writeFileSync } from 'fs';

export const loadImageToTensor = async (path) => {
  console.info('Loading Image.');
  const img = await Jimp.read(path);
  const scalingFactor = 512 / Math.max(img.bitmap.height, img.bitmap.width);
  img.scale(scalingFactor);
  // img.resize(224, 224);

  const p = [];

  img.scan(0, 0, img.bitmap.width, img.bitmap.height, function test(x, y, idx) {
    p.push(this.bitmap.data[idx + 0]);
    p.push(this.bitmap.data[idx + 1]);
    p.push(this.bitmap.data[idx + 2]);
  });

  return tf
    .tensor3d(p, [img.bitmap.width, img.bitmap.height, 3])
    .reshape([1, img.bitmap.width, img.bitmap.height, 3])
    .div(255);
};

export const saveImage = (path, tensor) => tf.tidy(() => {
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
    writeFileSync(path, buffer);
  });
});
