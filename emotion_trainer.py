import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Lista de emoções reconhecidas pelo modelo
EMOCOES = ["feliz", "triste", "surpreso", "neutro"]

# Parâmetros do modelo
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Carregar imagens e aplicar pré-processamento
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Criar modelo de rede neural
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(EMOCOES), activation="softmax")  # Saída para cada emoção
])

# Compilar o modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Treinar o modelo
model.fit(train_data, validation_data=val_data, epochs=10)

# Salvar o modelo treinado
model.save("emotion_model.h5")
print("Modelo treinado e salvo como emotion_model.h5")
