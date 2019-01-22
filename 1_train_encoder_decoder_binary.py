import numpy as np
import cv2

import keras
from keras import callbacks
from keras.models import Sequential, Model
from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, LSTM, Reshape, Conv2DTranspose, Input, LeakyReLU
from keras.optimizers import SGD
from keras import backend as K
import tensorflow as tf

import time
import glob
from PIL import Image


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour : {}min : {}sec'.format(t_hour, t_min, t_sec))


def cascade_all_models():

    T = 20
    Ch = 1
    Row = 128
    Col = 128
    input_shape_cnn_with_time_distribution = (T, Ch, Row, Col)  # num_steps, img_chs, img_rows, img_cols
    hidden_layer_size = 256
    dropout_rate = 0.2

    def cnn_with_time_distribution():
        # TimeDistributed wrapper applies a layer to every step.
        # The input should be at least 3D, and the dimension of index zero will be the temporal dimension.
        # E.g., using LSTM after cnn: https://github.com/keras-team/keras/issues/5527
        cnn = Sequential()
        cnn.add(  TimeDistributed( Conv2D(4, (3, 3), activation='relu'), input_shape=input_shape_cnn_with_time_distribution )  )
        cnn.add(TimeDistributed(Conv2D(4, (3, 3), activation='relu')))
        cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        cnn.add(TimeDistributed(Conv2D(8, (3, 3), activation='relu')))
        cnn.add(TimeDistributed(Conv2D(8, (3, 3), activation='relu')))
        cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        cnn.add(TimeDistributed(Conv2D(8, (3, 3), activation='relu')))
        cnn.add(TimeDistributed(Conv2D(8, (3, 3), activation='relu')))
        cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        cnn.add(TimeDistributed(Conv2D(8, (3, 3), activation='relu')))
        cnn.add(TimeDistributed(Conv2D(8, (3, 3), activation='relu')))
        cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        cnn.add(TimeDistributed(Flatten()))
        cnn.add(TimeDistributed(Dense(units=hidden_layer_size, activation='relu')))
        cnn.add(TimeDistributed(Dropout(dropout_rate)))
        print('output shape of cnn: {}'.format(cnn.output_shape))  # (None, 20, 256)

        return cnn

    def lstm():
        # example: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
        # example: https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
        #
        # The input to every LSTM layer must be three-dimensional. The three dimensions of this input are:
        # (i) Samples. One sequence is one sample. A batch is comprised of one or more samples. (this dim is omitted in input_shape of LSTM)
        # (ii) Time Steps. One time step is one point of observation in the sample.
        # (iii) Features. One feature is one observation at a time step.
        # This means that the input layer expects a 3D array of data when fitting the model and when making predictions,
        # even if specific dimensions of the array contain a single value, e.g. one sample or one feature.
        # Example: https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        #
        # (return_sequences=False) and (return_state=False) ==> output of LSTM is the hidden state at the last step
        lstm = Sequential()
        lstm.add(LSTM(units=hidden_layer_size, input_shape=(T, hidden_layer_size), return_sequences=False, return_state=False))
        print('output shape of lstm: {}'.format(lstm.output_shape))  # (None, 256)

        return lstm

    def encoder():
        enc = Sequential()
        enc.add(Dense(units=256, activation='relu', input_shape=(hidden_layer_size, )))  # Important difference: (hidden_layer_size, ) is 1D array while (hidden_layer_size, 1) is 2D array!!
        enc.add(Dropout(dropout_rate))

        enc.add(Dense(units=256, activation='relu'))
        enc.add(Dropout(dropout_rate))

        enc.add(Dense(units=hidden_layer_size, activation='relu'))
        enc.add(Dropout(dropout_rate))
        print('output shape of enc: {}'.format(enc.output_shape))  # (None, 256, 256)

        return enc

    def decoder():  # output of decoder would be of shape (chs=1, rows=128, cols=128)
        dec = Sequential()
        dec.add(Dense(units=256, activation='relu', input_shape=(hidden_layer_size, )))  # H1. Important difference: (hidden_layer_size, ) is 1D array while (hidden_layer_size, 1) is 2D array!!
        dec.add(Dropout(dropout_rate))

        dec.add(Dense(units=512, activation='relu'))  # H2
        dec.add(Dropout(dropout_rate))

        dec.add(Reshape((8, 8, 8)))  # layers=8, each is of size 8*8:  H2r

        dec.add(Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='same', output_padding=(1, 1)))  # H3
        dec.add(LeakyReLU(alpha=0.3))  # all advanced activations in Keras, including LeakyReLU, are available as layers, and not as activations
        dec.add(Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='same', output_padding=(1, 1)))  # H4
        dec.add(LeakyReLU(alpha=0.3))
        dec.add(Conv2DTranspose(filters=4, kernel_size=(3, 3), strides=(2, 2), padding='same', output_padding=(1, 1)))  # H5
        dec.add(LeakyReLU(alpha=0.3))
        dec.add(Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', output_padding=(1, 1), activation='sigmoid'))    # H6

        print('output shape of dec: {}'.format(dec.output_shape))

        return dec

    cnn_model = cnn_with_time_distribution()
    lstm_model = lstm()
    encoder_model = encoder()
    decoder_model = decoder()

    input_cascade = Input(shape=input_shape_cnn_with_time_distribution)
    out_1 = cnn_model(input_cascade)
    out_2 = lstm_model(out_1)  # out_2 denotes the single hidden state at the last step of lstm_model
    out_3 = encoder_model(out_2)
    out_cascade = decoder_model(out_3)  # out_cascade would be of shape (chs=1, rows=128, cols=128)
    cascaded_model = Model(inputs=input_cascade, outputs=out_cascade)

    return cascaded_model


# This class could be instantiated to be either training_batch_generator or validation_batch_generator
#
# The use of keras.utils.Sequence guarantees the ordering and avoids duplicate data when using use_multiprocessing=True.
# This structure guarantees that the network will only train once on each sample point per epoch which is not the case with generators.
#
# The yield statement generates a value and suspends the generator's execution, until the generator is called again.
#
# *** When the generator is called again, the generator resumes to execute the statement immediately after the last
# yield statement (the suspension point) ***
#
# *** This allows to produce a sequence of values (i.e., iterate over a sequence) 'on the fly', rather than computing
# and storing them, and sending them back in a list all at once. Thus no need to store the entire sequence in memory ***
#
# a single output yielded by the generator makes a single batch. Different batches may have different sizes.
class BatchGenerator(keras.utils.Sequence):
    def __init__(self, training_batch=True, shuffle=True, B=30, T=20):
        self.__training_batch = training_batch
        self.__shuffle = shuffle
        self.__B = B  # maximal possible number of valid agents in an instance
        self.__T = T
        self.__sequence_indices_in_an_epoch = []

        if self.__training_batch:
            self.__starting_sequence_idx = 0
            self.__num_sequences = 4000
            print('training_batch_gen initiated')
        else:
            self.__starting_sequence_idx = 4000
            self.__num_sequences = 1000
            print('validation_batch_gen initiated')

        self.on_epoch_end()

    # There is only one instance of class BatchGenerator. At the end of each training/validation epoch, on_epoch_end() will be automatically called by the system
    def on_epoch_end(self):  # updates indices after each epoch
        self.__sequence_indices_in_an_epoch = np.arange(self.__starting_sequence_idx, self.__starting_sequence_idx+self.__num_sequences)
        if self.__shuffle:
            np.random.shuffle(self.__sequence_indices_in_an_epoch)

    def __len__(self):  # return the number of batches per epoch (a data point is (x_sequence, y_img), per agent; a batch is an instance)
        return self.__num_sequences

    ####################################################################################################################
    # input
    #    sequence_num, starting from 1
    #    agent_idx, starting from 0
    #    step_idx, starting from 0:
    #      if step_idx is not None: grab and process single x image at t=step_idx
    #      if step_idx is None: grab and process single y label
    # output
    #    (i) single normalized image of shape (chs=1, dsrows, dscols), if this single image can be grabbed
    #    (ii) None: current single x image can not be grabbed (which occurs if all agents of this instance reach their literal destination before t steps)
    #
    # This method does not use self within its body and hence does not actually change the class instance. Thus this
    # method could be static, i.e. callable without having created a class instance, like ClassName.StaticMethod()
    #
    # A static method is a method that knows nothing about the class or instance it was called on. The class instance
    # therefore is NOT passed in as an implicit first argument to the static method.
    # Static method improves code readability, signifying that this method does not depend on the state of the class instance itself
    ####################################################################################################################
    @staticmethod  # declare that this is a static method.
    def __grab_and_process_single_img(sequence_num, agent_idx, step_idx=None):  # private method
        original_num_rows = 2049
        original_num_cols = 2049
        spatial_dsr = 16

        # [0] grab either single x image or single y label
        if step_idx is not None:  # grab single x image
            base_dir = './microscopic_image_samples_T_steps_binary/'
            img_dir = base_dir + 'instance' + str(sequence_num) + '/' + 'agent' + str(agent_idx) + '/' + str(step_idx) + '.png'
        else:  # grab single y_img
            base_dir = './microscopic_labels_dsr_10_binary/'
            img_dir = base_dir + 'instance' + str(sequence_num) + '/' + 'agent' + str(agent_idx) + '.png'

        # [1] rgb2grey
        img = cv2.imread(img_dir, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # print('shape of img = {}'.format(img.shape))  # (2049, 2049)
        if img is None:
            assert (step_idx is not None)  # the img (None) being tried to grab must be a single x image. assert <==> raise error if not
            #print('In sequence num {}, for agent_idx {}, single x image at step_idx {} can not be grabbed.'.format(sequence_num, agent_idx, step_idx))
            return None

        # [2] crop the last row and last column and conduct down sampling
        img_ds = img[0:(original_num_rows - 1):spatial_dsr, 0:(original_num_cols - 1):spatial_dsr]  # i:j:k = [i, i+k, i+2k, ..., j), where j can not be reached
        # disp = Image.fromarray(img_ds)
        # disp.show()

        # [3] threshold to binary {0, 1} by swapping labels, for both single x(t) image and ground truth y_img
        img_ds_th = np.where(img_ds > (255/2.), 0, 1)

        # [4] dtype conversion from int64 to float32
        img_ds_th_32f = img_ds_th.astype(np.float32)

        # [5] insert a new dimension at axis=2
        img_dsrows_dscols_chs = np.expand_dims(img_ds_th_32f, axis=2)  # chs==1
        # print('shape of img_dsrows_dscols_chs = {}'.format(img_dsrows_dscols_chs.shape))  # (128, 128, 1)

        # [6] switch channel last (dsrows, dscols, chs) to channel first (chs, dsrows, dscols)
        # Move axes of an array to new positions. Other axes remain in their original order.
        # Source position: original position of the axes to move,
        # Destination position: destination position of the original axes.
        img_chs_dsrows_dscols = np.moveaxis(img_dsrows_dscols_chs, 2, 0)
        # print('shape of img_chs_dsrows_dscols = {}'.format(img_chs_dsrows_dscols.shape))  # (1, 128, 128)

        return img_chs_dsrows_dscols  # of shape (chs=1, dsrows, dscols), in binary {0., 1.}, with dtype float32

    # a data point is (x_sequence, y_img), per agent; a batch is an instance
    def __process_curr_data_point(self, sequence_num, curr_agent_idx):  # a private method
        # [0] get x_sequence
        list_x_sequence = []
        for t in range(self.__T):  # t denotes step index within a sequence
            x_chs_dsrows_dscols = self.__grab_and_process_single_img(sequence_num, curr_agent_idx, t)  # chs==1

            if x_chs_dsrows_dscols is not None:  # an x image is grabbed
                list_x_sequence.append(np.expand_dims(x_chs_dsrows_dscols, axis=0))
            else:  # can not grab x image ==> all agents reach their literal destination before t steps
                list_x_sequence.append(list_x_sequence[-1])  # copy the previous x image
                # print(np.array_equal(list_x_train_one_sample[-1], list_x_train_one_sample[-2]))

        x_T_chs_dsrows_dscols = np.concatenate(list_x_sequence, axis=0)  # x_sequence
        # print('shape of x_T_chs_dsrows_dscols = {}'.format(x_T_chs_dsrows_dscols.shape))  # (T, chs=1, dsrows, dscols)

        # [1] get the corresponding y_img
        y_chs_dsrows_dscols = self.__grab_and_process_single_img(sequence_num, curr_agent_idx)  # y_img, whose chs==1
        # print('shape of y_chs_dsrows_dscols = {}'.format(y_chs_dsrows_dscols.shape))

        # [2] append current data point to list of all data points
        # list_x_all_points_in_a_batch.append(np.expand_dims(x_T_chs_dsrows_dscols, axis=0))  # append x_sequence to a list
        # list_y_all_points_in_a_batch.append(np.expand_dims(y_chs_dsrows_dscols, axis=0))  # append the y_img to a list
        return np.expand_dims(x_T_chs_dsrows_dscols, axis=0), np.expand_dims(y_chs_dsrows_dscols, axis=0)

    # a data point is (x_sequence, y_img), per agent; a batch is an instance
    # __getitem__() returns a complete batch
    # batch_index, which could be viewed as a shared variable by all processes, denotes the index of batch, from 0 to 3999 or from 4000 to 4999, in order.
    def __getitem__(self, batch_index):
        sequence_index = self.__sequence_indices_in_an_epoch[batch_index]
        sequence_num = sequence_index + 1

        if self.__training_batch:
            print(' current training batch num is {}'.format(sequence_num))
        else:
            print(' current validation batch num is {}'.format(sequence_num))

        # for an instance (Batch), prepare training/testing batch: (x_BS_T_Ch_Row_Col, y_BS_Ch_Row_Col), by
        # concatenating np array vertically from empty where BS = batch size, total number of valid agents in one instance
        list_x_all_points_in_a_batch = []
        list_y_all_points_in_a_batch = []

        # all x_sequences in current instance
        x_base_dir = './microscopic_image_samples_T_steps_binary/' + 'instance' + str(sequence_num) + '/'
        x_agents = glob.glob(x_base_dir + '*/')

        # all y_imgs in current instance
        y_base_dir = './microscopic_labels_dsr_10_binary/' + 'instance' + str(sequence_num) + '/'
        y_imgs = glob.glob(y_base_dir + '*.png')

        # check whether the number of valid agents in x equals the number of valid agents in y, for the current instance
        assert (len(y_imgs) == len(x_agents))  # assert <==> raise error if not

        for curr_agent_idx in range(self.__B):
            curr_agent_dir_x = x_base_dir + 'agent' + str(curr_agent_idx) + '/'
            curr_agent_dir_y = y_base_dir + 'agent' + str(curr_agent_idx) + '.png'

            if (curr_agent_dir_x in x_agents) and (curr_agent_dir_y in y_imgs):  # current agent is valid
                x_1_T_chs_dsrows_dscols, y_1_chs_dsrows_dscols = self.__process_curr_data_point(sequence_num, curr_agent_idx)  # return one data point (one agent)
                list_x_all_points_in_a_batch.append(x_1_T_chs_dsrows_dscols)
                list_y_all_points_in_a_batch.append(y_1_chs_dsrows_dscols)
                # print('agent index {} matches in (x, y)'.format(curr_agent_idx))
            elif (curr_agent_dir_x not in x_agents) and (curr_agent_dir_y not in y_imgs):  # current agent is invalid
                # print(' agent index {} is invalid'.format(curr_agent_idx))
                continue
            else:
                # print(' agent index {} does not match in (x, y)'.format(curr_agent_idx))
                continue

        x_BS_T_Ch_Row_Col = np.concatenate(list_x_all_points_in_a_batch, axis=0)
        # print(' shape of x_BS_T_Ch_Row_Col = {}'.format(x_BS_T_Ch_Row_Col.shape))

        y_BS_Ch_Row_Col = np.concatenate(list_y_all_points_in_a_batch, axis=0)
        # print(' shape of y_BS_Ch_Row_Col = {}'.format(y_BS_Ch_Row_Col.shape))

        # a single output yielded by the generator makes a single batch. Different batches may have different sizes.
        return (x_BS_T_Ch_Row_Col, y_BS_Ch_Row_Col)


def weighted_binary_crossentropy(y_true, y_pred):
    # meaning of y_pred: predicted probability that the current pixel belongs to label 1
    # y_true and y_pred are tensor of the same shape
    # K.binary_crossentropy returns a tensor of the same shape as y_true and y_pred
    # tf.scalar_mul returns a tensor of the same type as input tensor
    # K.mean: if axis is None, all dimensions are reduced, and a tensor with a single element is returned.
    # K.sum: if axis is None, all dimensions are reduced, and a tensor with a single element is returned.

    weight_pos_f = 80.
    weight_neg_f = 1.

    mask_pos_b = tf.greater_equal(y_true, 255/2.)  # tf.greater_equal supports broadcasting and returns a boolean tensor of the same shape as y_true
    mask_neg_b = tf.logical_not(mask_pos_b)

    mask_pos_f = tf.to_float(mask_pos_b)  # cast a tensor to type float32
    mask_neg_f = tf.to_float(mask_neg_b)
    weight_map = tf.scalar_mul(weight_pos_f, mask_pos_f) + tf.scalar_mul(weight_neg_f, mask_neg_f)

    # return K.mean(K.binary_crossentropy(y_true, y_pred) * weight_map) / (K.sum(weight_map) + K.epsilon())
    return K.mean(K.binary_crossentropy(y_true, y_pred) * weight_map)


def soft_dice_loss(y_pred, y_true):
    '''
    meaning of y_pred: the predicted probability that the current pixel is activated (belongs to true label 1)

    The numerator is concerned with the common activations between prediction and target mask;
    The denominator is concerned with the quantity of activations in each mask separately;
    This has the effect of normalizing the loss according to the size of the target mask such that the soft Dice loss...
    does not struggle learning from classes with lesser spatial representation in an image.

    c is number of classes (channels). In the simplest case, c can be 1.
    y_pred: bs * c * X * Y( * Z...) network output, must sum to 1 over c channel (such as after softmax). For instance, BatchGenerator.__getitem__() returns y_BS_Ch_Row_Col
    y_true: bs * c * X * Y( * Z...) one hot encoding of ground truth

    # References
        https://www.jeremyjordan.me/semantic-segmentation/#loss
        https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # K.sum: sum of the values in a tensor along side the specified axis
    # '*' is element-wise multiplication of two tensors
    # '+' is element-wise summation of two tensors
    # K.mean: if axis is None, all dimensions are reduced, and a tensor with a single element is returned.

    # skip the batch axis and class (channel) axis for calculating Dice score
    axes = (2, 3)
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(y_pred + y_true, axes)
    # denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)  # an alternative of denominator

    return 1 - K.mean(numerator / (denominator + K.epsilon()), axis=None)  # average over classes and batch when axis=None


if __name__ == '__main__':
    model_index = 0

    print(keras.__version__)
    print(K.tensorflow_backend._get_available_gpus())

    # specify training hyper-parameters
    max_num_epoch = 100 * 2
    num_batch_size = 30  # 30 agents in each batch (instance)
    Learning_Rate = 0.1
    num_training_instances = 4000
    num_testing_instances = 1000

    # specify optimizer
    # decay: learning rate decay over each update
    # momentum: accelerate SGD and dampen oscillations with short-term memory. See: https://distill.pub/2017/momentum/
    opt = SGD(lr=Learning_Rate, decay=0.0, momentum=0.9)

    # specify loss
    loss_fn = 'mean_squared_error'
    # loss_fn = 'binary_crossentropy'  # loss of making decisions, per element (pixel)
    # loss_fn = weighted_binary_crossentropy
    # loss_fn = soft_dice_loss

    # instantiate and configure model
    model = cascade_all_models()
    model.compile(loss=loss_fn, optimizer=opt)

    # prepare callback1: early stopping
    # monitor on validation loss; if no improvement after patience epochs, stop training
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1)

    # prepare callback2: tensorboard
    tbcallback = keras.callbacks.TensorBoard(log_dir='./logs', batch_size=num_batch_size, write_graph=False)

    # prepare callback3: checkpoint
    cpcallback = keras.callbacks.ModelCheckpoint('./models/micro_wo_disentangle_binary_v{}.h5'.format(model_index), monitor='val_loss', save_best_only=False)

    # instantiate generator
    # There is only one instance of class TrainingBatchGenerator. At the end of each training epoch, on_epoch_end() will be automatically called by the system
    # There is only one instance of class ValidationBatchGenerator. At the end of each validation epoch, on_epoch_end() will be automatically called by the system
    training_batch_gen = BatchGenerator(training_batch=True, shuffle=False)
    validation_batch_gen = BatchGenerator(training_batch=False, shuffle=False)

    # train with callback
    print('Begin training...')
    tic()

    # use fit_generator to grab a training/validation batch on the fly (i.e., when it needs to obtain a new batch to
    # update nn in one step), thus to avoid loading all training/validation set all at once
    model.fit_generator(training_batch_gen,
                        steps_per_epoch=num_training_instances,  # A single output yielded by the generator makes a single batch. A batch is an instance. There are 4000 instances in training set.
                        epochs=max_num_epoch,
                        callbacks=[cpcallback],
                        validation_data=validation_batch_gen,    # on which to evaluate the loss and metrics at the end of each epoch
                        validation_steps=num_testing_instances,
                        max_queue_size=12,  # specify the number of batches to prepare in the queue. It does not mean you will have multiple generator instances
                        workers=12,         # launch multiple extra workers for both training_batch generator and validation_batch generator (thus the main process can focus on training)
                        use_multiprocessing=True,
                        shuffle=False,      # whether to shuffle the order of batches at the beginning of each epoch. Has no effect when steps_per_epoch is not None
                        initial_epoch=0)

    print('Finish training.')
    tac()
