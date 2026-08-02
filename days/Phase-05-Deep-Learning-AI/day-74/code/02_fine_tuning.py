"""
Day 74 — Transfer Learning
Topic: Fine-tuning Pretrained Models
Date: 31 July 2026
Author: Bala Ravi

Phase 2: Unfreeze top layers, very low LR.
Foundation for Skin Disease Detector!
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


def build_finetune_model(
        n_classes: int = 7,
        input_shape: tuple = (224, 224, 3),
        unfreeze_from: int = 100
        ) -> tuple:
    """
    Build model ready for fine-tuning.

    Args:
        n_classes: Number of disease classes
        input_shape: Image input shape
        unfreeze_from: Layer index to unfreeze from

    Returns:
        (model, base_model) tuple
    """
    if not TF_AVAILABLE:
        return None, None

    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet')

    # Phase 1: Freeze all
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(
        inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        256, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(
        n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model, base


def unfreeze_for_finetuning(
        model,
        base_model,
        unfreeze_from: int = 100,
        learning_rate: float = 1e-5
        ) -> None:
    """
    Unfreeze top layers for fine-tuning.

    MUST be called AFTER Phase 1 training!

    Args:
        model: Full model
        base_model: Base pretrained model
        unfreeze_from: Layer to start unfreezing
        learning_rate: Very low LR for fine-tuning
    """
    if not TF_AVAILABLE:
        return

    base_model.trainable = True

    # Freeze bottom layers, unfreeze top
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    n_trainable = sum([
        1 for l in base_model.layers
        if l.trainable])

    print(f"Unfroze layers {unfreeze_from}+ "
          f"({n_trainable} layers)")
    print(f"Fine-tuning LR: {learning_rate}")

    # Recompile with very low LR!
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])


def data_augmentation_pipeline() -> None:
    """Show data augmentation for skin images."""
    print("=== Data Augmentation Pipeline ===\n")
    print("Skin disease image augmentations:\n")

    if not TF_AVAILABLE:
        augmentations = {
            'RandomFlip(horizontal)': (
                'Flip left-right (skin lesions symmetric)'),
            'RandomRotation(0.2)': (
                'Rotate ±20% (lesions any angle)'),
            'RandomZoom(0.15)': (
                'Zoom in/out ±15% (different distances)'),
            'RandomContrast(0.2)': (
                'Vary contrast ±20% (lighting variation)'),
            'RandomBrightness(0.15)': (
                'Vary brightness (different skin tones)'),
            'RandomTranslation(0.1, 0.1)': (
                'Shift image ±10% (lesion position)')
        }

        print(f"{'Augmentation':<35} | {'Purpose'}")
        print("-" * 70)

        for aug, purpose in augmentations.items():
            print(f"{aug:<35} | {purpose}")

        print(f"\n💡 Applied randomly during TRAINING only!")
        print(f"   Val/test: no augmentation!")
        print(f"   Artificially increases dataset size!")
        print(f"   1000 images × augmentation = effectively 10K+")
        return

    augmentation_layer = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomContrast(0.2),
        keras.layers.RandomBrightness(0.15)
    ], name='augmentation')

    print("Augmentation layers:")
    for layer in augmentation_layer.layers:
        print(f"  → {layer.name}")

    # Demo on random image
    dummy_image = tf.random.uniform(
        (1, 224, 224, 3), 0, 255)

    print(f"\nOriginal image shape: "
          f"{dummy_image.shape}")
    print(f"After augmentation:   "
          f"{augmentation_layer(dummy_image).shape}")
    print(f"\n💡 Same shape — just different appearance!")
    print(f"   Model sees 'new' image each epoch!")


def full_transfer_workflow() -> None:
    """Show complete transfer learning workflow."""
    print("\n=== Complete Transfer Learning Workflow ===\n")

    print("This is what Day 77-80 will build:")
    print("(Skin Disease Detector)\n")

    steps = [
        ("1. Load pretrained MobileNetV2",
         "weights='imagenet', include_top=False",
         "Get 3.4M pretrained weights"),

        ("2. Add custom classification head",
         "GAP → Dense(256) → Dense(7)",
         "7 skin disease classes"),

        ("3. Freeze base model",
         "base.trainable = False",
         "Protect pretrained features"),

        ("4. Phase 1: Train head only",
         "optimizer=Adam(lr=1e-3), epochs=20",
         "Learn skin disease mapping"),

        ("5. Evaluate Phase 1",
         "model.evaluate(val_data)",
         "Should see ~82% accuracy"),

        ("6. Unfreeze top layers",
         "layers[100:].trainable = True",
         "Adapt top features to skin domain"),

        ("7. Phase 2: Fine-tune",
         "optimizer=Adam(lr=1e-5), epochs=20",
         "Very low LR → avoid forgetting"),

        ("8. Evaluate Phase 2",
         "model.evaluate(val_data)",
         "Should see ~91% accuracy!"),

        ("9. Save model",
         "model.save('skin_model.h5')",
         "Ready for deployment!"),

        ("10. Deploy FastAPI",
         "POST /predict → image → diagnosis",
         "Live skin disease detection!")
    ]

    for step, code, outcome in steps:
        print(f"  {step}")
        print(f"    Code:    {code}")
        print(f"    Result:  {outcome}")
        print()

    print("✅ From 1000 images to 91% accuracy!")
    print("   Transfer learning makes this possible!")
    print("   Day 77: Skin Disease Detector builds this! 🔥")


if __name__ == "__main__":
    demonstrate = True

    if TF_AVAILABLE and demonstrate:
        print("=== Fine-tuning Demo ===\n")
        model, base = build_finetune_model(
            n_classes=7,
            input_shape=(96, 96, 3))

        if model:
            total = model.count_params()
            trainable = sum([
                tf.size(v).numpy()
                for v in model.trainable_variables])
            print(f"Phase 1 — Feature Extraction:")
            print(f"  Total params:     {total:,}")
            print(f"  Trainable:        {trainable:,}")
            print(f"  Trainable %:      "
                  f"{trainable/total*100:.1f}%")

            print(f"\nPhase 2 — After unfreezing top layers:")
            unfreeze_for_finetuning(
                model, base,
                unfreeze_from=100,
                learning_rate=1e-5)
            trainable2 = sum([
                tf.size(v).numpy()
                for v in model.trainable_variables])
            print(f"  Trainable:        {trainable2:,}")
            print(f"  Trainable %:      "
                  f"{trainable2/total*100:.1f}%")

    data_augmentation_pipeline()
    full_transfer_workflow()
