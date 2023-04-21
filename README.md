<h1 align="center">
  Facial Emotion Recognition
</h1>
</br>
<p align="center">
  By Nigel Mun (1005031), Xu Muzi (1005641), Emmanuel Lopez (1005407)
</p>

**Usage of code:**

1. visualisation
- *data-visualisation.ipynb* displays all the different distributions of the csv dataset
- *face-recognition-analysis.ipynb* shows a plot of the images on a graph, showing their similarities/differences to each other

2. Preprocessing
- *CSV_to_Dataset.ipynb* converts all the csv data into a Tensor object, one for training, one for testing
- *DataAug.ipynb* performs data augmentation to the Tensor object, and saving it as a new Tensor object

3. Models
- All the files in this folder are the models we have tested during our initial trial phase, to pick out the best 2 models that we are going to employ in our project.

4. Final models
- The models in here are the models we will use to train and test with live data from the web cam.

5. VideoCapture
- You can run the video capture file using:
- ```cd /VideoCapture```
- ```python VideoCapture.py```
- This will cause a popup whereby your computer camera will be activated, and you can test what emotion is predicted from your face.




