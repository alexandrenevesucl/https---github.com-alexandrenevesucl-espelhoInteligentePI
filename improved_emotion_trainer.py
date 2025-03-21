import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os

# Definição das emoções
EMOCOES = ["feliz", "triste", "surpreso", "neutro"]

# Diretórios de treino e validação
dataset_dir = "dataset"
treino_dir = os.path.join(dataset_dir, "train")
validacao_dir = os.path.join(dataset_dir, "validation")

# Data Augmentation para aumentar a variabilidade dos dados
datagen_treino = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

datagen_validacao = ImageDataGenerator(rescale=1.0 / 255)

# Criando os datasets de treino e validação
batch_size = 32
treino_generator = datagen_treino.flow_from_directory(
    treino_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode="categorical",
)

validacao_generator = datagen_validacao.flow_from_directory(
    validacao_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode="categorical",
)

# Construção do modelo aprimorado
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation="relu", input_shape=(48, 48, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(EMOCOES), activation="softmax"),
])

# Compilação do modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Treinamento do modelo
epochs = 30
history = model.fit(
    treino_generator,
    epochs=epochs,
    validation_data=validacao_generator,
)

# Salvando o modelo aprimorado
model.save("improved_emotion_model.h5")
print("Modelo aprimorado treinado e salvo como improved_emotion_model.h5")
