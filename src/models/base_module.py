import datetime

import tensorflow as tf


class BaseModule(tf.Module):
    checkpoint_path = 'checkpoints/'
    saved_model_path = 'saved_model/'

    _output_coarse_name = 'output_coarse'
    _output_fine_name = 'output_fine'

    @staticmethod
    def get_output_coarse_name():
        return BaseModule._output_coarse_name

    @staticmethod
    def get_output_fine_name():
        return BaseModule._output_fine_name

    def train(self, train_dataset: tf.data.Dataset, valid_dataset: tf.data.Dataset, num_epochs: int,
              steps_per_epoch: int):
        lr_decay = tf.keras.callbacks.LearningRateScheduler(self._lr_decay)

        # Used for TensorBoard
        log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=20,  # Saves every 20 epochs
            save_best_only=True)

        self.model.fit(train_dataset, epochs=num_epochs, validation_data=valid_dataset,
                       steps_per_epoch=steps_per_epoch,
                       callbacks=[lr_decay, tensorboard_callback, cp_callback])

    def test(self, test_dataset: tf.data.Dataset):
        self.model.evaluate(test_dataset)

    def load_weights(self):
        self.model.load_weights(self.checkpoint_path)

    def save(self):
        """Save the model in the saved_model_path directory under a directory named after the class.
        E.g. For the ResNet50 class, <saved_model_path>/ResNet50/
        """
        self.model.save(self.saved_model_path + type(self).__name__ + '/')

    @staticmethod
    def _lr_decay(epoch: int) -> float:
        lr = 1e-3
        if epoch > 80:
            lr *= 1e-2
        elif epoch > 60:
            lr *= 1e-1

        return lr
