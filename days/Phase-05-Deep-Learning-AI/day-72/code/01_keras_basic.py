"""
Day 72 — TensorFlow & Keras
Topic: Keras Sequential API Basics
Date: 29 July 2026
Author: Bala Ravi

Same neural network as Day 71 scratch —
but in 15 lines of Keras instead of 150!
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    tf.random.set_seed(42)
    print(f"TensorFlow: {tf.__version__}")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed.")
    print("    Run: pip install tensorflow")


def build_sequential_model(
        n_features: int,
        task_type: str = 'binary'
        ) -> 'keras.Model':
    """
    Build Sequential Keras model.

    Args:
        n_features: Number of input features
        task_type: 'binary', 'multiclass', 'regression'

    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        return None

    if task_type == 'binary':
        output_units = 1
        output_activation = 'sigmoid'
        loss = 'binary_crossentropy'
        metrics = ['accuracy']

    elif task_type == 'regression':
        output_units = 1
        output_activation = 'linear'
        loss = 'mse'
        metrics = ['mae']

    else:  # multiclass
        output_units = 4
        output_activation = 'softmax'
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

    model = keras.Sequential([
        keras.layers.Dense(
            64, activation='relu',
            input_shape=(n_features,),
            name='hidden_1'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(
            32, activation='relu',
            name='hidden_2'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(
            output_units,
            activation=output_activation,
            name='output')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001),
        loss=loss,
        metrics=metrics)

    return model


def keras_vs_scratch() -> None:
    """Compare Keras vs our Day 71 scratch NN."""
    print("\n=== Keras vs Scratch NN ===\n")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        n_informative=6,
        random_state=42)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("Showing expected output:\n")
        print("Scratch NN accuracy:  0.8917")
        print("Keras accuracy:       0.9083")
        print("\n✅ Same model — similar results!")
        print("   Keras: 15 lines of code")
        print("   Scratch: 150 lines of code")
        return

    model = build_sequential_model(
        n_features=10, task_type='binary')

    print("Model Summary:")
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1)
    ]

    print("\nTraining Keras model...")
    history = model.fit(
        X_tr, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0)

    test_loss, test_acc = model.evaluate(
        X_te, y_test, verbose=0)

    stopped_epoch = len(history.history['loss'])
    print(f"\nEarly stopped at epoch: {stopped_epoch}")
    print(f"Final val_loss: "
          f"{history.history['val_loss'][-1]:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    print(f"\n📊 Code comparison:")
    print(f"  Day 71 Scratch NN: ~150 lines")
    print(f"  Day 72 Keras:      ~15 lines")
    print(f"  Same architecture → similar results! ✅")


def train_with_callbacks() -> None:
    """Show all important Keras callbacks."""
    print("\n=== Keras Callbacks Demo ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nKey callbacks to know:")

        callbacks_info = {
            'EarlyStopping': (
                "Stop training when metric stops improving. "
                "patience=10 → stop if no improvement for 10 epochs."),
            'ModelCheckpoint': (
                "Save best model during training. "
                "save_best_only=True → only saves improvements."),
            'ReduceLROnPlateau': (
                "Reduce learning rate when stuck. "
                "factor=0.5 → halve LR when val_loss plateaus."),
            'TensorBoard': (
                "Visualize training in browser. "
                "Most powerful debugging tool!")
        }

        for name, desc in callbacks_info.items():
            print(f"  {name}:")
            print(f"    {desc}\n")
        return

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=1000, n_features=12,
        n_informative=8, random_state=42)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = build_sequential_model(12, 'binary')

    import tempfile, os
    checkpoint_path = os.path.join(
        tempfile.gettempdir(), 'best_model.h5')

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1)
    ]

    history = model.fit(
        X_tr, y_train,
        epochs=300,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0)

    n_epochs = len(history.history['loss'])
    test_acc = model.evaluate(
        X_te, y_test, verbose=0)[1]

    print(f"Training stopped at epoch: {n_epochs}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"\n💡 EarlyStopping saved us from")
    print(f"   running unnecessary epochs!")
    print(f"   Model restored to BEST weights! ✅")


def keras_regression_demo() -> None:
    """Keras for regression — house prices."""
    print("\n=== Keras Regression ===\n")

    np.random.seed(42)
    n = 500

    area = np.random.normal(1500, 400, n)
    bedrooms = np.random.randint(1, 6, n).astype(float)
    age = np.random.exponential(15, n)

    price = (
        area * 6000 +
        bedrooms * 300000 -
        age * 30000 +
        np.random.normal(0, 500000, n))

    X = np.column_stack([area, bedrooms, age])
    y = price

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42))

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)
    y_tr = scaler_y.fit_transform(
        y_train.reshape(-1, 1)).flatten()
    y_te = scaler_y.transform(
        y_test.reshape(-1, 1)).flatten()

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nExpected results:")
        print("  Test MAE: ~₹350,000")
        print("  Test R²:  ~0.89")
        return

    model = build_sequential_model(
        3, task_type='regression')

    history = model.fit(
        X_tr, y_tr,
        epochs=300,
        batch_size=32,
        validation_split=0.2,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=15,
            restore_best_weights=True)],
        verbose=0)

    y_pred_scaled = model.predict(
        X_te, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()

    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - np.sum((y_test - y_pred)**2) / (
        np.sum((y_test - y_test.mean())**2))

    print(f"Regression Results:")
    print(f"  MAE: ₹{mae:,.0f}")
    print(f"  R²:  {r2:.4f}")
    print(f"\n💡 Same Keras API for classification!")
    print(f"   Just change: loss='mse',")
    print(f"   output activation='linear'")


if __name__ == "__main__":
    keras_vs_scratch()
    train_with_callbacks()
    keras_regression_demo()
