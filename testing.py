import numpy as np
import tensorflow as tf
import normalization_modified
import gesture_classification_utils
from sklearn.model_selection import train_test_split

v = np.load("2026-10-17_14-39-27_003_raw_MANUAL.npz")
labels = v['labels']

LABELS = {
    -1: "UNLABELED",
    0: "NONE",
    1: "FOLLOW",
    2: "STAY",
    3: "FETCH",
    4: "COME",
    5: "GIVE",
    6: "INVALID",
}

import numpy as np

### NORMALIZATION ###

raw_keypoints = v['raw_keypoints']
labels[labels == -1] = 0

normalized_figure = np.array([normalization_modified.normalize_figure(kp).ravel() for kp in raw_keypoints] )
joint_angles = normalization_modified.compute_joint_angles(normalized_figure)
print(normalized_figure[0].shape)

# test out this point
print(labels[668])
norm_668 = normalization_modified.normalize_figure(raw_keypoints[668]).ravel()
norm_668 = np.expand_dims(norm_668, axis=0)
model = tf.keras.models.load_model("./my_checkpoint.keras")
prediction = model.predict(norm_668)
print(np.argmax(prediction, axis=-1))

### TRAINING ###

# X_train, X_test, y_train, y_test = train_test_split(
#     normalized_figure,
#     labels, 
#     train_size=0.75, 
#     random_state=3485789347
# )

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input((36, )),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(20, activation='relu'),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(8, activation='softmax')
# ])

# # Model checkpoint callback
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     "./my_checkpoint.keras", verbose=1, save_weights_only=False)
# # Callback for early stopping
# es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(
#     X_train,
#     y_train,
#     epochs=1000,
#     batch_size=128,
#     validation_data=(X_test, y_test),
#     callbacks=[cp_callback, es_callback]
# )