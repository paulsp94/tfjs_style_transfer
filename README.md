# Style Transfer in tfjs

This repository tries to implement the neural style transfer.

## Setup

To setup the repo run:

`npm run setup`

This will clone the required model into the directory.

## Run

To run the style transfer call:

`npm start`

## Extracted feature samples

Here are some samples of extracted features from style (first 5) and content (last one) from different layers.

![alt text](./featureSample/styleTransfer/500_39807.625_style1.png "block1_conv1")
Style extraced from the first convolutional layer in the first convolutional block.

![alt text](./featureSample/styleTransfer/500_58900656_style2.png "block2_conv1")
Style extraced from the first convolutional layer in the second convolutional block.

![alt text](./featureSample/styleTransfer/500_29894260_style3.png "block3_conv1")
Style extraced from the first convolutional layer in the third convolutional block.

![alt text](./featureSample/styleTransfer/500_2514929.5_style4.png "block4_conv1")
Style extraced from the first convolutional layer in the fourth convolutional block.

![alt text](./featureSample/styleTransfer/500_82911.734375_style5.png "block5_conv1")
Style extraced from the first convolutional layer in the fifth convolutional block.
