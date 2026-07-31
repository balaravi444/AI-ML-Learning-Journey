"""
Day 72 — TensorFlow & Keras
Topic: Functional API + Model Architecture Patterns
Date: 29 July 2026
Author: Bala Ravi

Sequential API → simple models
Functional API → complex architectures

Multi-input, skip connections, shared layers.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def functional_api_demo() -> None:
    """Show Functional API for complex models."""
    print("=== Keras Functional API ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nFunctional API concept:\n")
        print("""
# Sequential (simple, linear):
model = keras.Sequential([
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Functional (flexible, any architecture):
inputs = keras.Input(shape=(10,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

# Multi-input (text + numeric):
text_input = keras.Input(shape=(500,))
num_input  = keras.Input(shape=(10,))

text_branch = Dense(64, activation='relu')(text_input)
num_branch  = Dense(32, activation='relu')(num_input)

merged = keras.layers.Concatenate()(
    [text_branch, num_branch])
output = Dense(1, activation='sigmoid')(merged)

model = keras.Model(
    inputs=[text_input, num_input],
    outputs=output)
        """)
        return

    print("1. Standard Sequential:\n")
    seq_model = keras.Sequential([
        keras.layers.Dense(
            64, activation='relu',
            input_shape=(10,)),
        keras.layers.Dense(
            32, activation='relu'),
        keras.layers.Dense(
            1, activation='sigmoid')
    ])
    print(f"   Params: {seq_model.count_params():,}")

    print("\n2. Functional API (same architecture):\n")
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(
        64, activation='relu')(inputs)
    x = keras.layers.Dense(
        32, activation='relu')(x)
    outputs = keras.layers.Dense(
        1, activation='sigmoid')(x)
    func_model = keras.Model(
        inputs=inputs,
        outputs=outputs)
    print(f"   Params: {func_model.count_params():,}")

    print("\n3. Multi-input (numeric + text features):\n")
    num_input = keras.Input(
        shape=(10,), name='numeric')
    text_input = keras.Input(
        shape=(500,), name='text_tfidf')

    num_branch = keras.layers.Dense(
        32, activation='relu')(num_input)
    text_branch = keras.layers.Dense(
        64, activation='relu')(text_input)

    merged = keras.layers.Concatenate()(
        [num_branch, text_branch])
    hidden = keras.layers.Dense(
        32, activation='relu')(merged)
    output = keras.layers.Dense(
        1, activation='sigmoid')(hidden)

    multi_model = keras.Model(
        inputs=[num_input, text_input],
        outputs=output)

    print(f"   Inputs:  numeric(10,) + text(500,)")
    print(f"   Params:  {multi_model.count_params():,}")
    print(f"\n   This is how Bug Predictor v3 would")
    print(f"   combine text + metadata in one model! 🔥")

    print("\n✅ Functional API enables:")
    print("   Multi-input models (text + metadata)")
    print("   Skip connections (ResNet style)")
    print("   Shared layers")
    print("   Multiple outputs")


def regularization_demo() -> None:
    """Show dropout and batch normalization."""
    print("\n=== Regularization in Keras ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nRegularization techniques:\n")

        techniques = {
            'Dropout(0.3)': (
                "Randomly zeroes 30% of neurons during training.\n"
                "    Forces network to learn redundant representations.\n"
                "    Reduces overfitting significantly!"),
            'BatchNormalization()': (
                "Normalizes layer inputs during training.\n"
                "    Stabilizes training, allows higher LR.\n"
                "    Apply before activation."),
            'L2 kernel_regularizer': (
                "Penalizes large weights.\n"
                "    keras.regularizers.l2(0.001)\n"
                "    Same as Ridge regression but for NNs!"),
        }

        for name, desc in techniques.items():
            print(f"  {name}:")
            print(f"    {desc}\n")
        return

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=500, n_features=20,
        n_informative=10, random_state=42)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    def build_model(with_regularization: bool):
        model = keras.Sequential()
        model.add(keras.layers.Dense(
            128, input_shape=(20,)))

        if with_regularization:
            model.add(
                keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        if with_regularization:
            model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Dense(64))
        if with_regularization:
            model.add(
                keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        if with_regularization:
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(
            1, activation='sigmoid'))
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
        return model

    results = {}
    for reg in [False, True]:
        model = build_model(reg)
        history = model.fit(
            X_tr, y_train,
            epochs=150, batch_size=32,
            validation_split=0.2,
            verbose=0)

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history[
            'val_accuracy'][-1]
        test_acc = model.evaluate(
            X_te, y_test, verbose=0)[1]
        gap = train_acc - val_acc

        label = ("With regularization"
                  if reg else "No regularization")
        results[label] = {
            'train': train_acc,
            'val': val_acc,
            'test': test_acc,
            'gap': gap}

    print(f"{'Model':<25} | "
          f"{'Train':>7} | "
          f"{'Val':>7} | "
          f"{'Test':>7} | "
          f"{'Gap':>7}")
    print("-" * 60)

    for name, r in results.items():
        flag = ("✅" if r['gap'] < 0.05
                 else "⚠️  overfit")
        print(f"{name:<25} | "
              f"{r['train']:>7.4f} | "
              f"{r['val']:>7.4f} | "
              f"{r['test']:>7.4f} | "
              f"{r['gap']:>7.4f} {flag}")

    print(f"\n💡 Dropout + BatchNorm:")
    print(f"   Reduces train-val gap significantly!")
    print(f"   Better generalization to test set! 🔥")


if __name__ == "__main__":
    functional_api_demo()
    regularization_demo()
