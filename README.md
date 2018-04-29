# Dogs vs Cats

## Data
[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

## Projects Requirements
https://github.com/nd009/capstone/tree/master/dog_vs_cat

## File Catalog
### Dir: classification
- data_preprocess.ipynb: find outliers; clipping&padding.
- ref_data.ipynb: generate data directory for ImageDataGenerator.
- gen_feature.ipynb: generate feature vector.
- single_model.ipynb: single model logistic regression experiments.
- single_model_fine_tune.ipynb: Xception last block fine-tune.
- multi_model.ipynb: ensemble models logistic regression.
- class_activation_map.ipynb: generate CAM.
- outliers_total.json: outliers.
  
### Dir: docs
- proposal.pdf
- proposal_data.ipynb: for plot data.
- proposal_sketch.docx
- plot_model.ipynb: for plot model structure.
- report_sketch.docx
- report.pdf
- data_preprocess.html
- ref_data.html
- gen_feature.html
- single_model.html
- single_model_fine_tune.html
- multi_model.html
- class_activation_map.html

## Enviroment
- Ubuntu 16.04 LTS
- NVIDIA GTX 1080
- CUDA 9.0
- cuDNN 7.0
- Anaconda 3.5 (Python 3.6)
- Tensorflow 1.6
- Keras 2.1.5

## Procedure
1. ref_data;
2. data_preprocess: using 3 pretrained models to find outliers_total, about 15mins for each model.
3. ref_data: regenerate data references with out outliers; split data.
4. gen_feature: generate features, about 10mins for each model.
5. single_model: experiment with InceptionV3，Xception，InceptionResNetV2 respectively.
6. single_model_fine_tune: fine-tune Xception's last conv block, about 460s for each epoch, 20 epochs in total(2.5 hours).
7. multi_model: ensemble models.
8. class_activation_map: generate CAM.

![cam1](https://github.com/crj0322/dog_vs_cat/raw/master/docs/cam1.png)
![cam2](https://github.com/crj0322/dog_vs_cat/raw/master/docs/cam2.png)
![cam3](https://github.com/crj0322/dog_vs_cat/raw/master/docs/cam3.png)

## Detection
### Dir: yolov3
- yolo_model.ipynb: build darknet53 and yolov3 output layers with keras.
- yolo_predict.ipynb: implemented prediction with numpy for easily understandin.
![yolov3](https://github.com/crj0322/dog_vs_cat/raw/master/docs/fo4yolo.png)

- the model weights and draw bbox codes are stolen from [HERE](https://github.com/qqwweee/keras-yolo3).
- todo: train yolo with dog cat data.
