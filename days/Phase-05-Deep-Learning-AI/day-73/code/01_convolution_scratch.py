"""
Day 73 — CNN: Convolutional Neural Networks
Topic: Convolution Operation from Scratch
Date: 30 July 2026
Author: Bala Ravi

Understanding what convolution actually does
before using Keras to do it automatically!
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def convolve2d(
        image: np.ndarray,
        kernel: np.ndarray,
        padding: str = 'valid') -> np.ndarray:
    """
    2D Convolution from scratch.

    Slides kernel over image, computes dot product
    at each position = feature map!

    Args:
        image: 2D array (H, W)
        kernel: 2D filter (kH, kW)
        padding: 'valid' shrinks, 'same' keeps size

    Returns:
        Feature map (output of convolution)
    """
    H, W = image.shape
    kH, kW = kernel.shape

    if padding == 'same':
        pH = kH // 2
        pW = kW // 2
        image = np.pad(
            image,
            ((pH, pH), (pW, pW)),
            mode='constant')
        H_out, W_out = H, W
    else:
        H_out = H - kH + 1
        W_out = W - kW + 1

    feature_map = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            patch = image[i:i+kH, j:j+kW]
            feature_map[i, j] = np.sum(
                patch * kernel)

    return feature_map


def max_pool2d(
        feature_map: np.ndarray,
        pool_size: int = 2) -> np.ndarray:
    """
    MaxPooling from scratch.

    Takes maximum value in each pool_size × pool_size window.
    Reduces spatial dimensions by pool_size factor.

    Args:
        feature_map: 2D array
        pool_size: Window size

    Returns:
        Pooled feature map (halved dimensions)
    """
    H, W = feature_map.shape
    H_out = H // pool_size
    W_out = W // pool_size

    pooled = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            window = feature_map[
                i*pool_size:(i+1)*pool_size,
                j*pool_size:(j+1)*pool_size]
            pooled[i, j] = window.max()

    return pooled


def demonstrate_convolution() -> None:
    """Show what different filters detect."""
    print("=== Convolution Demo ===\n")

    # Simulated 8×8 grayscale image
    image = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ], dtype=float)

    # Different filters
    filters = {
        'Vertical Edge Detector': np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]),
        'Horizontal Edge Detector': np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]]),
        'Sharpen': np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]]),
        'Blur': np.array([
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]])
    }

    print("Image (8×8 — vertical edge at column 4):")
    for row in image:
        print('  ' + ' '.join(
            ['█' if v > 0 else '·' for v in row]))

    print()
    for filter_name, kernel in filters.items():
        feature_map = convolve2d(
            image, kernel, padding='valid')
        feature_map = np.clip(feature_map, 0, 8)

        print(f"{filter_name}:")
        print(f"  Feature map ({feature_map.shape}):")
        for row in feature_map:
            bar = '  '
            for v in row:
                intensity = int(v / 8 * 5)
                bar += '█' * intensity + '·' * (
                    5 - intensity) + ' '
            print(bar)
        print()

    print("💡 Vertical Edge Detector:")
    print("   High activation at column 3-4!")
    print("   CNN LEARNS these filters automatically! 🔥")


def demonstrate_pooling() -> None:
    """Show MaxPooling effect."""
    print("\n=== MaxPooling Demo ===\n")

    feature_map = np.array([
        [1, 3, 2, 4, 1, 2],
        [5, 6, 7, 8, 3, 1],
        [1, 2, 3, 4, 5, 6],
        [9, 8, 7, 6, 5, 4],
        [2, 3, 4, 5, 6, 7],
        [8, 7, 6, 5, 4, 3]
    ], dtype=float)

    pooled = max_pool2d(feature_map, pool_size=2)

    print(f"Feature map: {feature_map.shape}")
    print(feature_map.astype(int))

    print(f"\nAfter MaxPool(2×2): {pooled.shape}")
    print(pooled.astype(int))

    print(f"\n  Size reduced: "
          f"{feature_map.shape} → {pooled.shape}")
    print(f"  Parameters reduced by 4×!")
    print(f"  Most important features preserved! ✅")


def parameter_comparison() -> None:
    """Compare Dense vs CNN parameters."""
    print("\n=== Dense vs CNN Parameters ===\n")

    img_size = 64
    n_classes = 10

    # Dense approach
    n_inputs = img_size * img_size * 3
    dense_params = (
        n_inputs * 512 +    # Input → Dense(512)
        512 +               # bias
        512 * 256 +         # Dense(512) → Dense(256)
        256 +               # bias
        256 * n_classes +   # Dense(256) → output
        n_classes)

    # CNN approach (simplified)
    conv1_params = (3 * 3 * 3 * 32) + 32    # 3×3, 3 ch → 32 filters
    conv2_params = (3 * 3 * 32 * 64) + 64   # 3×3, 32 ch → 64 filters
    conv3_params = (3 * 3 * 64 * 128) + 128 # 3×3, 64 ch → 128 filters
    # After 3 MaxPools: 64/8 × 64/8 × 128 = 8×8×128
    gap_to_dense = 128 * 256 + 256
    dense_out = 256 * n_classes + n_classes
    cnn_params = (conv1_params + conv2_params +
                  conv3_params + gap_to_dense +
                  dense_out)

    print(f"Image size: {img_size}×{img_size}×3")
    print(f"Classes: {n_classes}\n")

    print(f"Dense NN parameters:  {dense_params:>12,}")
    print(f"CNN parameters:       {cnn_params:>12,}")
    ratio = dense_params / cnn_params
    print(f"\nCNN uses {ratio:.0f}× fewer parameters!")
    print(f"→ Less overfit, faster training!")
    print(f"→ Better generalization! 🔥")


if __name__ == "__main__":
    demonstrate_convolution()
    demonstrate_pooling()
    parameter_comparison()
