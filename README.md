# face-recognition-model-development
Model development part of interview project


download the dataset from https://www.kaggle.com/datasets/atulanandjha/lfwpeople


How to train / finetune the model and deploy in app? 
app: https://github.com/haidiazaman/face-data-matching-app/tree/main

1. clone repo.
2. create dataset folder and add your images.
3. preprocess your data (run MTCNN face detector to get all the bounding boxes and prepare csv. follow format in lfw_dataset/lfw_face_processed.csv) Please ensure to plot and check if these faces are correct.
4. create a config file in train/configs. you can change the parameters as you require.
5. edit the config file name in train/runs.sh.
6. start training by navigating to train/ , followed by ./runs.sh. You may need to do chmod +x runs.sh first
   - in terminal do:
   - cd train
   - chmod +x runs.sh
   - ./runs.sh
7. check results in evaluate/ . Open evaluate_model.ipynb to check your model results inclusive of linear classifier head and check_embeddings_similarity_score.ipynb to check similarity scores for embeddings of image output vectors.
8. strip the last layer from the trained model and convert to tflite using evaluate/pytorch2tflite.py
9. to use the tflite model in the Android app, add this model to the assets/ folder in the app and change the embedding dim in ScannerActivity.kt and CheckRecords.kt to 512.

![alt text](https://github.com/haidiazaman/face-recognition-model-development/blob/main/imgs/photo_2024-01-26_00-52-05.jpg)

This diagram is a high level summary of the training pipeline and how the model is used in an Android app for real-time facial recognition. First, as a data preprocessing step, all images are passed through the MTCNN face detector to get the boundings boxes of the faces. These faces are then cropped using the generated bounding boxes. Next for training, an InceptionResnetV1 model is initialised. An additional linear classifier is added to the model as the classifier head for training. The classifier output number of neurons is equal to the number of unique persons in the dataset. As such the output num_classes is set dynamically. It is inferred from the dataset only during training. The dataset should contain a substantial number of images for each individual (>=20) to get good results. The model learns by standard multi-class image classification technique. With the classifier head, the model learns just like a normal multi-class classification model, except the number of classes are exactly equal to number of unique individuals in the dataset (typical tasks usually have preset number of classes, e.g. cats-vs-dogs or fashion-products). 


After training, the linear classifier can be used to do evaluation on the test dataset (same as normal multi-class classification tasks). Once performance is acceptable, the linear classifier is discarded. To implement in an Android app or other face recognition tasks, the model embeddings output is used. This is a 512-embedding vector. If 2 images are of the same person, then the embedding vectors of each image should be similar. This is checked by implementing a similarity check logic / function to calculate a similarity score. By setting an appropriate threshold, you will consider a face match when the sim_score(v1,v2)>=threshold, else -> no face match.

Question: Why are we not directly using the entire model together with the classifier head? In an ideal scenario where your target audience has been included in your train set, then yes, you can just use the entire model since this becomes purely an image classification task. But the main idea here is that you train the model on tonnes of images of people of different ethnicity, colour, etc. This will train the model to be able to generate embeddings that will cluster similar faces together, and dissimilar faces far apart. The idea is that for a new user, if the model is trained well (check test / unseen set), then the model will be able to generate similar embeddings for different images of this same person despite not having seen this person before (person not in training set)
