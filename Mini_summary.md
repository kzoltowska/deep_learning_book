# Keras Neural Network — Setup Guide

A step-by-step reference for building, training, and evaluating a Keras model for regression or classification.

---

## 1. Set Random Seeds

Do this **before anything else** — before imports, before data loading.

```python
import tensorflow as tf
import numpy as np
import random

def reset_random_seeds(nseed, enable_determinism=False):
    tf.keras.utils.set_random_seed(nseed)  # covers TF, NumPy, and Python random
    if enable_determinism:
        tf.config.experimental.enable_op_determinism()  # slower but fully reproducible

reset_random_seeds(42)
```

> Re-call `reset_random_seeds()` every time you define a new model graph.

---

## 2. Prepare Data

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit on train only
X_test  = scaler.transform(X_test)       # transform test with same scaler
```

> Always scale **after** splitting. Fitting the scaler on the full dataset leaks information.

---

## 3. Build the Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential([
    Input(shape=(n_features,)),
    Dense(64, activation='relu', kernel_initializer='he_uniform'),
    Dense(32, activation='relu', kernel_initializer='he_uniform'),
    Dense(1)  # no activation = linear output for regression
              # use 'sigmoid' for binary classification
              # use 'softmax' for multi-class classification
])
```

**Key decisions per layer:**

| Setting | Regression | Binary Classification | Multi-class |
|---|---|---|---|
| Output units | 1 | 1 | num_classes |
| Output activation | none (linear) | sigmoid | softmax |
| Loss function | mse / mae | binary_crossentropy | categorical_crossentropy |
| Metric | mae | accuracy | accuracy |

---

## 4. Compile the Model

```python
model.compile(
    optimizer='adam',       # see hyperparameter tuning below
    loss='mse',
    metrics=['mae']
)
```

---

## 5. Add Callbacks

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,            # stop after N epochs with no improvement
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,             # multiply LR by this on plateau
    patience=10,
    min_lr=1e-6
)
```

---

## 6. Train the Model

```python
history = model.fit(
    X_train, y_train,
    validation_split=0.2,   # or pass validation_data=(X_val, y_val)
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

---

## 7. Plot Loss History

```python
import matplotlib.pyplot as plt

def plot_loss_history(h, title):
    plt.plot(h.history['loss'], label='Train loss')
    plt.plot(h.history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.legend()
    plt.show()

plot_loss_history(history, 'Training History')
```

---

## 8. Evaluate

```python
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {test_mae:.4f}")

y_pred = model.predict(X_test)
```

---

## 9. Hyperparameters to Tune

These are the main levers to adjust when your model is underfitting or overfitting:

| Hyperparameter | What to try | Notes |
|---|---|---|
| **Number of layers** | 1–5 | More layers = more capacity |
| **Units per layer** | 8, 16, 32, 64, 128, 256 | Start small, scale up |
| **Activation** | relu, leaky_relu, elu | ReLU is a safe default |
| **Kernel initializer** | he_uniform, he_normal | Use he_* with ReLU |
| **Learning rate** | 1e-2, 1e-3, 1e-4 | Most impactful single parameter |
| **Optimizer** | adam, rmsprop, sgd | Adam works well by default |
| **Batch size** | 16, 32, 64, 128 | Smaller = noisier but may generalise better |
| **Epochs + patience** | 100–500, patience 10–30 | Use EarlyStopping; don't tune epochs manually |
| **Dropout rate** | 0.1–0.5 | Regularisation; add `Dropout(rate)` layers |
| **L2 regularisation** | 1e-4 to 1e-2 | `kernel_regularizer=tf.keras.regularizers.l2(1e-4)` |

---

## 10. Feature Importance

Keras does not have built-in feature importance. Common approaches:

**Permutation importance** — shuffle one feature at a time and measure performance drop:

```python
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

# Wrap Keras model for sklearn
result = permutation_importance(wrapped_model, X_test, y_test, n_repeats=10)
```

**Sanity-check the importance algorithm with a random feature:**

> Add a column of random noise to your dataset before running importance:
>
> ```python
> import numpy as np
> X['random_noise'] = np.random.randn(len(X))
> ```
>
> After computing importances, any real feature ranked **below the random noise feature**
> is contributing less than chance and can be considered uninformative. This is a simple
> but effective way to validate that the importance algorithm is working correctly.

---

## Quick Checklist

- [ ] Seed set before model definition
- [ ] Data split before scaling
- [ ] Input shape matches features
- [ ] Output layer matches task (linear / sigmoid / softmax)
- [ ] Loss function matches task
- [ ] EarlyStopping added to avoid overfitting
- [ ] Loss curves plotted and inspected
- [ ] Random noise feature added to validate feature importance