import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Layer
from tensorflow.keras.callbacks import EarlyStopping

import os

#train model- True. Load model-False.
TRAIN_MODEL = False
TIME_STEPS = 168

#load dataset
files = [
    "USA_GA_Albany-Dougherty.County.AP.722160_TMY3_LOW.csv",
    "USA_GA_Albany-Dougherty.County.AP.722160_TMY3_BASE.csv",
    "USA_GA_Albany-Dougherty.County.AP.722160_TMY3_HIGH.csv"
]

dfs = []

for i, file in enumerate(files):
    temp_df = pd.read_csv(file)

    year = 2023 + i
    temp_df["Date/Time"] = f"{year} " + temp_df["Date/Time"]

    mask_24 = temp_df["Date/Time"].str.contains("24:00:00")

    temp_df.loc[mask_24, "Date/Time"] = (
        temp_df.loc[mask_24, "Date/Time"]
        .str.replace("24:00:00", "00:00:00", regex=False)
    )

    temp_df["Date/Time"] = pd.to_datetime(
        temp_df["Date/Time"], format="%Y %m/%d  %H:%M:%S"
    )

    temp_df.loc[mask_24, "Date/Time"] += pd.Timedelta(days=1)

    dfs.append(temp_df)

df = pd.concat(dfs)
df = df.sort_values("Date/Time").set_index("Date/Time")

print("Consolidated shape:", df.shape)

#target column
target_col = "Electricity:Facility [kW](Hourly)"
series = df[target_col]

#tain test split
split_ratio = 0.8
split_index = int(len(series) * split_ratio)

train_series = series.iloc[:split_index]
test_series  = series.iloc[split_index:]

#scaling
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
test_scaled  = scaler.transform(test_series.values.reshape(-1, 1))

#creating time sequences
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, TIME_STEPS)
X_test, y_test   = create_sequences(test_scaled, TIME_STEPS)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

#positional encoding layer
class PositionalEncoding(Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.constant(angle_rads, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding

#transformer block
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)

    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(inputs.shape[-1])(ff)

    ff = Dropout(dropout)(ff)
    return LayerNormalization(epsilon=1e-6)(ff + x)

#model
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Dense(32)(inputs)  # 🔥 projection
    x = PositionalEncoding(TIME_STEPS, 32)(x)

    x = transformer_block(x, 32, 2, 64)
    x = transformer_block(x, 32, 2, 64)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="mse"
    )

    return model

#train or load
MODEL_PATH = "electricity_transformer_model.keras"

if TRAIN_MODEL or not os.path.exists(MODEL_PATH):

    print("\nTraining Transformer...")

    model = build_transformer_model((TIME_STEPS, 1))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(MODEL_PATH)

else:
    print("\nLoading Saved Transformer Model...")

    model = load_model(
        MODEL_PATH,
        custom_objects={
            "PositionalEncoding": PositionalEncoding,
            "transformer_block": transformer_block
        }
    )

#predictions
y_pred = model.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

np.save("transformer_pred.npy", y_pred_inv)
np.save("transformer_test.npy", y_test_inv)
#evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)

epsilon = 1e-10
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + epsilon)))

print("\nTransformer Evaluation:")
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape, "%")

#random 168hr window for prediction
start_idx = np.random.randint(0, len(series) - TIME_STEPS - 1)

random_168 = series.values[start_idx : start_idx + TIME_STEPS]
actual_next = series.values[start_idx + TIME_STEPS]

random_scaled = scaler.transform(random_168.reshape(-1, 1))
random_scaled = random_scaled.reshape(1, TIME_STEPS, 1)

pred_scaled = model.predict(random_scaled)
pred = scaler.inverse_transform(pred_scaled)

print("\n📊 RANDOM WINDOW PREDICTION")
print(f"Predicted: {pred[0][0]:.2f} kW")
print(f"Actual   : {actual_next:.2f} kW")

print("\n✅ Transformer model completed successfully.")