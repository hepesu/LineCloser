# LineCloser
Unofficial Keras implementation of Joint Gap Detection and Inpainting of Line Drawings.

## Overview
Joint gap for line-drawings. Model1 uses network from the paper. For stable training, BN was added for all Conv2D. Model2 uses common network for inpaint.

## Dependencies
* Keras2 (Tensorflow backend)
* OpenCV3
* CairoSVG

## Usage
1. Set up directories.

2. Download the model from release and put it in the same folder with code.

3. Run `predict.py` for prediction. Run `model{NUM}.py` for train.

## Data Preparation
There are 3 methods for data generation, `DATA_GEN`, `DATA_GAP` and `DATA_THIN`.

0. Use `DATA_GEN` for training, the data is generated online.

1. Collect line-drawings with [LineDistiller](https://github.com/hepesu/LineDistiller).

2. Put line-drawings into `data/line`, using `DATA_GAP` for training.

3. Thin(normalize) the line-drawings with [LineNormalizer](https://github.com/hepesu/LineNormalizer) or tranditional thinning method. 

4. Manually processe line-drawings and thinning results(threshold etc.), then crop them into pieces.

5. Put line-drawings into `data/line` and put thinning results into `data/thin`, using `DATA_THIN` for training.

## Models
Models are licensed under a CC-BY-NC-SA 4.0 international license.
* [LineCloser Release Page](https://github.com/hepesu/LineCloser/releases)



From **Project HAT** by Hepesu With :heart:
