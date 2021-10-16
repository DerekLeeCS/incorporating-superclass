import tensorflow as tf

from sklearn.model_selection import train_test_split

from datasets.cifar100 import CIFAR100
from models.baseline import ResNet50

# From:
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# Constants
IMG_SIZE = 32
BATCH_SIZE = 64
VALID_SIZE = 0.2
NUM_EPOCHS = 160
AUTOTUNE = tf.data.experimental.AUTOTUNE
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
METRIC = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
IS_TRAINING = True

if __name__ == '__main__':
    # Get data
    dataset = CIFAR100()
    data_train = dataset.get_data(True)
    data_test = dataset.get_data(False)
    num_classes = dataset.get_num_classes()

    # Extract data
    train_img = data_train[b'data']
    train_label = data_train[b'fine_labels']
    test_img = data_test[b'data']
    test_label = data_test[b'fine_labels']

    # Train / Valid Split
    train_img, valid_img, train_label, valid_label = train_test_split(train_img, train_label, test_size=VALID_SIZE)

    # Calculate number of steps per epoch
    steps_per_epoch = int(tf.shape(train_img)[0] / BATCH_SIZE)

    # Data Augmentation
    data_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                                                     height_shift_range=0.1, horizontal_flip=True)
    data_gen_train.fit(train_img)

    # Convert to Dataset
    train_dataset = (
        tf.data.Dataset.from_generator(lambda: data_gen_train.flow(train_img, train_label, batch_size=BATCH_SIZE),
                                       output_signature=(
                                           tf.TensorSpec(shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                           tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32)))
        .prefetch(AUTOTUNE)
    )
    valid_dataset = (
        tf.data.Dataset.from_tensor_slices((valid_img, valid_label))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_img, test_label))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    # Run model
    model = ResNet50(num_classes, IMG_SIZE, LOSS, OPTIMIZER, METRIC)
    if IS_TRAINING:
        model.train(train_dataset, valid_dataset, NUM_EPOCHS, steps_per_epoch)
        model.plot_accuracy()
        model.save()
    else:
        model.load_weights()

    model.test(test_dataset)
