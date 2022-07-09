# Music Genre Classification using CNN
Classifying music into genres using CNN.

## Preprocessing
Music files are saved with their name in format, 'name' - 'genre'. So I can open each file and extract the label from its name. I use ***wavfile*** to open the files and use ***scipy.signal.sepctrogram()*** to convert the files to spectrogram (see image below). I feed these spectrogram to the CNN.

![image](https://user-images.githubusercontent.com/106041952/178120953-7ffe2203-220c-4286-8034-5b08778da4b6.png)

## Classifier
Classifier is a ***CNN*** with 4 convolutional layers and pooling layers and batch normalization in first 3 layers. Then it flattens it and uses 3 ***relu*** layers and at last it uses softmax to get probabilities for each class.

![Untitled1](https://user-images.githubusercontent.com/106041952/178121777-24c4b5a9-4b1d-4674-8ff9-1db8a6af0f31.png)
