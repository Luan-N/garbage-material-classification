from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _get_image_paths(data_dir):
    data_path = Path(data_dir)
    class_dirs = sorted([path for path in data_path.iterdir() if path.is_dir()])

    class_names = [path.name for path in class_dirs]
    class_to_index = {name: index for index, name in enumerate(class_names)}

    image_paths = []
    labels = []
    class_counts = {}

    # Read images from each class folder and count them.
    for class_dir in class_dirs:
        files = sorted(
            [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        )
        class_counts[class_dir.name] = len(files)

        for file_path in files:
            image_paths.append(str(file_path))
            labels.append(class_to_index[class_dir.name])

    return image_paths, labels, class_names, class_counts


def _filter_unreadable_images(image_paths, labels):
    valid_paths = []
    valid_labels = []
    unreadable_files = []

    # Keep only images Pillow can open and verify.
    for path, label in zip(image_paths, labels):
        try:
            with Image.open(path) as image:
                image.verify()
            valid_paths.append(path)
            valid_labels.append(label)
        except Exception:
            unreadable_files.append(path)

    return valid_paths, valid_labels, unreadable_files


def _build_dataset(paths, labels, image_size, batch_size, training, preprocess_fn, augmenter):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        # Load image, force RGB, resize, and apply the model-specific preprocessing.
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)# tensorflow models expects float32 image data type
        image = preprocess_fn(image)
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Camera blurring simulation
    def random_blur(image, label):
        if tf.random.uniform(()) > 0.5:
            image = tf.expand_dims(image, 0)
            image = tf.nn.avg_pool2d(image, ksize=3, strides=1, padding="SAME")
            image = tf.squeeze(image, 0)
        return image, label

    # Shuffle and augment only the training dataset to improve generalization and prevent overfitting.
    if training:
        dataset = dataset.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
        if augmenter is not None:
            dataset = dataset.map(
                lambda image, label: (augmenter(image, training=True), label),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        dataset = dataset.map(random_blur, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def prepare_datasets(
    data_dir="data/RealWaste",
    image_size=(224, 224), #224/244 is recommended/standard for mobilenetv2, and also works well for cnn.
    batch_size=32,
    model_type="cnn",
    val_size=0.1,
    test_size=0.1,
    random_state=42,
):
    # Read the dataset folders and class counts.
    image_paths, labels, class_names, class_counts = _get_image_paths(data_dir)

    # Flag unreadable images and drop them from the pipeline.
    image_paths, labels, unreadable_files = _filter_unreadable_images(image_paths, labels)

    if not image_paths:
        raise ValueError(f"No valid images found in {data_dir}")

    labels = np.array(labels)

    # Split once for test, then split the rest for validation.
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    # Calculate the validation ratio relative to the remaining training data after the test split.
    val_ratio = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=random_state,
    )

    # PIXEL NORMALIZATION: For cnn, use range [0, 1]; for mobilenetv2, use the specific mobilenet_preprocess function.
    if model_type.lower() == "mobilenetv2":
        preprocess_fn = mobilenet_preprocess
    elif model_type.lower() == "cnn":
        preprocess_fn = lambda image: image / 255.0
    else:
        raise ValueError("model_type must be 'cnn' or 'mobilenetv2'")

    # AUGMENTATION: Apply augmentation(flip, rotation, zoom, translation) only to the training set to prevent overfitting and improve generalization.
    augmenter = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ]
    )

    # BUILD: Build TensorFlow datasets for training, validation, and testing.
    train_ds = _build_dataset(
        train_paths,
        train_labels,
        image_size,
        batch_size,
        training=True,
        preprocess_fn=preprocess_fn,
        augmenter=augmenter,
    )
    val_ds = _build_dataset(
        val_paths,
        val_labels,
        image_size,
        batch_size,
        training=False,
        preprocess_fn=preprocess_fn,
        augmenter=None,
    )
    test_ds = _build_dataset(
        test_paths,
        test_labels,
        image_size,
        batch_size,
        training=False,
        preprocess_fn=preprocess_fn,
        augmenter=None,
    )

    # COMPUTE CLASS WEIGHTS: Compute class weights from the training labels.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = {int(index): float(weight) for index, weight in enumerate(class_weights)}

    # Print the basic dataset summary.
    print("Classes:", class_names)
    print("Class counts:", class_counts)
    print("Unreadable files:", len(unreadable_files))
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "class_names": class_names,
        "class_counts": class_counts,
        "class_weights": class_weights,
        "unreadable_files": unreadable_files,
    }