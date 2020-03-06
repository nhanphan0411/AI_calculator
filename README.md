# AI CALCULATOR

![](https://i.imgur.com/trQORWG.png)

The project is to build a web application that can recognize and perform mathematics on handwritten regression.

The training uses CNN model with image data preprocessed by OpenCV. The entire code can be found at ```model/main_nb.ipynb```

To try web demo, run:
```
python3 app/main.py
```

Image data is received by using Javascript canvas, then processed with OpenCV before fit into pre-trained model for prediction.