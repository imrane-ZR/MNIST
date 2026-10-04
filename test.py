import cv2
import tensorflow as tf
import numpy as np

# Test sur des images de chiffres
path = "6.png"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = cv2.resize(image, (28, 28))
image = np.array(image) / 255.0
new_model = tf.keras.models.load_model("mnist_mlp_model.keras")
predict=new_model.predict(np.array([image]))
per=np.argmax(predict,axis=1)

print(per)
print(predict)