"""
Day 76 — Model Optimization + Callbacks
Topic: Learning Rate Scheduling
Date: 02 August 2026
Author: Bala Ravi

LR is the most important hyperparameter!
Fixed LR is almost always suboptimal.
Adaptive LR = better accuracy every time.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    tf.random.set_seed(42)
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed.")
    print("    Run: pip install tensorflow\n")


def demonstrate_lr_schedules() -> None:
    """Show different LR schedules visually."""
    print("=== Learning Rate Schedules ===\n")

    n_epochs = 50
    initial_lr = 1e-3

    # Schedule 1: Fixed
    fixed_lrs = [initial_lr] * n_epochs

    # Schedule 2: Step decay
    step_lrs = [
        initial_lr * (0.5 ** (epoch // 10))
        for epoch in range(n_epochs)]

    # Schedule 3: Cosine annealing
    cosine_lrs = [
        initial_lr * 0.5 * (
            1 + np.cos(np.pi * epoch / n_epochs))
        for epoch in range(n_epochs)]

    # Schedule 4: Warmup + cosine
    warmup = 5
    warmup_cosine_lrs = []
    for epoch in range(n_epochs):
        if epoch < warmup:
            lr = initial_lr * (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / (
                n_epochs - warmup)
            lr = initial_lr * 0.5 * (
                1 + np.cos(np.pi * progress))
        warmup_cosine_lrs.append(lr)

    print(f"{'Epoch':>6} | "
          f"{'Fixed':>10} | "
          f"{'Step Decay':>11} | "
          f"{'Cosine':>8} | "
          f"{'Warmup+Cos':>11}")
    print("-" * 55)

    for epoch in [0, 5, 10, 20, 30, 40, 49]:
        print(f"{epoch:>6} | "
              f"{fixed_lrs[epoch]:>10.2e} | "
              f"{step_lrs[epoch]:>11.2e} | "
              f"{cosine_lrs[epoch]:>8.2e} | "
              f"{warmup_cosine_lrs[epoch]:>11.2e}")

    print(f"\n💡 Cosine annealing usually wins!")
    print(f"   High LR early → explore loss surface")
    print(f"   Low LR later  → fine-tune minimum")
    print(f"\n   Warmup: prevents early instability")
    print(f"   Especially important for fine-tuning!")

    if TF_AVAILABLE:
        # Keras cosine decay schedule
        cosine_schedule = (
            keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-3,
                decay_steps=50 * 100,  # epochs × steps
                alpha=1e-6))

        print(f"\n  Keras CosineDecay schedule:")
        print(f"  Initial LR: 1e-3")
        print(f"  Final LR:   1e-6")
        print(f"  Follows cosine curve! ✅")


def reduce_lr_on_plateau_demo() -> None:
    """Show ReduceLROnPlateau in action."""
    print("\n=== ReduceLROnPlateau Demo ===\n")

    if not TF_AVAILABLE:
        print("Simulated ReduceLROnPlateau behavior:\n")

        val_losses = [
            0.95, 0.87, 0.81, 0.78, 0.77,
            0.76, 0.77, 0.78, 0.77, 0.76,
            0.75, 0.75, 0.75, 0.75, 0.75,  # plateau!
            0.73, 0.71, 0.70, 0.70, 0.70
        ]

        lr = 1e-3
        patience = 5
        factor = 0.3
        best_loss = float('inf')
        epochs_no_improve = 0

        print(f"{'Epoch':>5} | "
              f"{'Val Loss':>9} | "
              f"{'LR':>10} | "
              f"{'Event'}")
        print("-" * 45)

        for epoch, loss in enumerate(val_losses):
            event = ""
            if loss < best_loss - 0.001:
                best_loss = loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    lr *= factor
                    epochs_no_improve = 0
                    event = f"⚡ LR → {lr:.2e}"

            print(f"{epoch+1:>5} | "
                  f"{loss:>9.4f} | "
                  f"{lr:>10.2e} | "
                  f"{event}")

        print(f"\n✅ LR reduced automatically at plateau!")
        print(f"   Model continues improving! 🔥")
        return

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=1000, n_features=20,
        n_informative=12, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = keras.Sequential([
        keras.layers.Dense(
            128, activation='relu',
            input_shape=(20,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    lr_history = []

    class LRTracker(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = float(
                keras.backend.get_value(
                    self.model.optimizer.lr))
            lr_history.append(lr)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            verbose=1),
        LRTracker()
    ]

    print("Training with ReduceLROnPlateau...")
    history = model.fit(
        X_tr, y_tr,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0)

    final_acc = history.history['val_accuracy'][-1]
    lr_reductions = sum(
        1 for i in range(1, len(lr_history))
        if lr_history[i] < lr_history[i-1])

    print(f"\nFinal val accuracy: {final_acc:.4f}")
    print(f"LR reductions:      {lr_reductions}")
    print(f"Initial LR:         {lr_history[0]:.2e}")
    print(f"Final LR:           {lr_history[-1]:.2e}")
    print(f"\n✅ ReduceLROnPlateau found optimal LR!")


def cosine_annealing_demo() -> None:
    """Implement cosine annealing from scratch."""
    print("\n=== Cosine Annealing ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nCosine Annealing formula:")
        print("lr(t) = lr_min + 0.5 × (lr_max - lr_min)")
        print("        × (1 + cos(π × t / T))")
        print("\nwhere t=current epoch, T=total epochs")
        return

    class CosineAnnealingScheduler(
            keras.callbacks.Callback):
        """Custom cosine annealing scheduler."""

        def __init__(self,
                      n_epochs: int,
                      lr_max: float = 1e-3,
                      lr_min: float = 1e-6) -> None:
            super().__init__()
            self.n_epochs = n_epochs
            self.lr_max = lr_max
            self.lr_min = lr_min

        def on_epoch_begin(
                self, epoch: int,
                logs=None) -> None:
            lr = (self.lr_min + 0.5 *
                  (self.lr_max - self.lr_min) *
                  (1 + np.cos(
                      np.pi * epoch / self.n_epochs)))
            keras.backend.set_value(
                self.model.optimizer.lr, lr)

    print("Cosine LR schedule over 30 epochs:")
    for epoch in [0, 5, 10, 15, 20, 25, 29]:
        lr = (1e-6 + 0.5 * (1e-3 - 1e-6) *
               (1 + np.cos(np.pi * epoch / 30)))
        bar = '█' * int(lr / 1e-3 * 30)
        print(f"  Epoch {epoch:>2}: "
              f"{lr:.2e} {bar}")

    print(f"\n💡 Cosine schedule:")
    print(f"   High LR early → explore space")
    print(f"   Low LR late → converge precisely")
    print(f"   Often finds 1-3% better accuracy! 🔥")


if __name__ == "__main__":
    demonstrate_lr_schedules()
    reduce_lr_on_plateau_demo()
    cosine_annealing_demo()
