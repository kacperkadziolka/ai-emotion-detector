# AI/ML Emotion Detector
The project was created for the university course "Intelligent Systems Programming". <br/>
AI/ML Facial emotion detection model based on 7 types of facial expressions:
1. 'Angry'
2. 'Disgusted'
3. 'Fearful'
4. 'Happy'
5. 'Neutral'
6. 'Sad'
7. 'Surprised'

![AI/ML Emotion Detector](https://github.com/kacperkadziolka/kacperkadziolka/blob/main/emoition_detector.png)

# Dataset and training
Dataset used to train the model: <br/>
https://www.kaggle.com/datasets/msambare/fer2013 <br/>
<br/>
The training of the model was performed with 50 epochs and took about 2 hours. <br/>
It was computed with an i5-1135G7 processor, 16 GB RAM and a GeForce MX330.

# Run the model
To run the model copy the remote repository to your local storage and run it with the command:
```
python EmotionDetector.py
```

# Python packages used 
- numpy
- opencv
- keras
- tensorflow
