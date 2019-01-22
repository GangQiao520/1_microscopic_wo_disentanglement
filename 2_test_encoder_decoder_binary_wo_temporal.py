import numpy as np
import cv2
from scipy import misc

import time
import os
import glob

import keras
from keras import backend as K
import tensorflow as tf

# Important Note:
# (1) model.evaluate() is for evaluating a trained model. Its output is accuracy or loss, not predictions of input data.
#
# (2) model.predict() outputs actual target value (label), predicted from input data.
#
# (3) model.predict() goes through all the data and predicts labels batch by batch. It thus internally splits data in
#     batches and feeds to model one batch at a time.
#
# (4) model.predict_on_batch(), on the other hand, assumes that the input data is exactly one batch and thus feeds it to
#     model. It won't try to split the input data.
#
# (5) model.predict_generator generates predictions for input samples from a data generator. Though the prediction
#     operation is batch based, it returns predictions of all input samples in the total number of batches


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour : {}min : {}sec'.format(t_hour, t_min, t_sec))


# input
#    y_chs_rows_cols, of shape (chs=1, rows, cols), in {0., 1.}
#
# output
#    rgbY_rows_cols_chs, of shape (rows, cols, chs=3), in [0, 255]
def inv_process_img(y_chs_rows_cols):

    # [1] switch channel first (chs, rows, cols) to channel last (rows, cols, chs)
    # Move axes of an array to new positions. Other axes remain in their original order.
    # Source position: original position of the axes to move,
    # Destination position: destination position of the original axes.
    y_rows_cols_chs = np.moveaxis(y_chs_rows_cols, 0, 2)
    # print('The shape of y_rows_cols_chs = {}'.format(y_rows_cols_chs.shape))  # chs==1
    # print('')

    # [2] un-normalization to {0, 255}, here chs==1
    y_rows_cols_chs = y_rows_cols_chs * 255.
    #y_rows_cols_chs = np.where(y_rows_cols_chs > (255/2.), 255., 0.)  # binarization
    y_rows_cols_chs = np.clip(y_rows_cols_chs, a_min=0., a_max=255.)
    y_rows_cols_chs = y_rows_cols_chs.astype(np.uint8)
    # print('The shape of y_rows_cols_chs = {}'.format(y_rows_cols_chs.shape))  # chs==1
    # print('dtype of y_rows_cols_chs = {}'.format(y_rows_cols_chs.dtype))
    # print('')

    # [3] grey2rgb
    rgbY_rows_cols_chs = np.concatenate([y_rows_cols_chs, y_rows_cols_chs, y_rows_cols_chs], axis=2)
    # print('The shape of rgbY_rows_cols_chs = {}'.format(rgbY_rows_cols_chs.shape))  # chs==3
    # print('dtype of rgbY_rows_cols_chs = {}'.format(rgbY_rows_cols_chs.dtype))
    # print('')

    return rgbY_rows_cols_chs  # of shape (rows, cols, chs=3), in [0, 255]


# This class is used to instantiate only testing_batch_generator
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
# A single output yielded by the generator makes a single batch. Different batches may have different sizes.
class BatchGenerator(keras.utils.Sequence):
    def __init__(self, testing_batch=True, shuffle=True, B=30):
        self.__testing_batch = testing_batch
        self.__shuffle = shuffle
        self.__B = B  # maximal possible number of valid agents in an instance
        self.__starting_sequence_idx = 4000
        self.__num_sequences = 1000
        print('testing_batch_gen initiated')

        # self.on_epoch_end()
        self.__sequence_indices_in_an_epoch = np.arange(self.__starting_sequence_idx, self.__starting_sequence_idx + self.__num_sequences)

    # There is only one instance of class BatchGenerator. At the end of each training/validation epoch, on_epoch_end() will be automatically called by the system
    # def on_epoch_end(self):  # update indices after each epoch
    #     self.__sequence_indices_in_an_epoch = np.arange(self.__starting_sequence_idx, self.__starting_sequence_idx+self.__num_sequences)
    #     if self.__shuffle:
    #         np.random.shuffle(self.__sequence_indices_in_an_epoch)

    def __len__(self):  # return the number of batches per epoch (a data point is (x_img, y_img), per agent; a batch is an instance)
        return self.__num_sequences

    ####################################################################################################################
    # input
    #    sequence_num, starting from 1
    #    agent_idx, starting from 0
    #    input_sample:
    #      if True, process x_img
    #      if False, process y_img
    # output
    #    normalized image of shape (chs=1, dsrows, dscols), for either x_img or ground truth y_img
    #
    # This method does not use self within its body and hence does not actually change the class instance. Thus this
    # method could be static, i.e. callable without having created a class instance, like ClassName.StaticMethod()
    #
    # A static method is a method that knows nothing about the class or instance it was called on. The class instance
    # therefore is NOT passed in as an implicit first argument to the static method.
    # Static method improves code readability, signifying that this method does not depend on the state of the class instance itself
    ####################################################################################################################
    @staticmethod  # declare that this is a static method.
    def __grab_and_process_single_img(sequence_num, agent_idx, input_sample=True):  # private method
        original_num_rows = 2049
        original_num_cols = 2049
        spatial_dsr = 16

        # [0] grab either x_img or y_img for agent_idx
        if input_sample is True:  # grab x_img
            base_dir = '../0_Honglu_data_preprocessing/microscopic_domain/microscopic_image_samples_all_steps_binary/'
            img_dir = base_dir + 'instance' + str(sequence_num) + '/' + 'agent' + str(agent_idx) + '_all_steps.png'
        else:  # grab y_img
            base_dir = '../0_Honglu_data_preprocessing/microscopic_domain/microscopic_labels_dsr_10_binary/'
            img_dir = base_dir + 'instance' + str(sequence_num) + '/' + 'agent' + str(agent_idx) + '.png'

        # [1] rgb2grey
        img = cv2.imread(img_dir, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # print('shape of img = {}'.format(img.shape))  # (2049, 2049)

        # [2] crop the last row and last column and conduct down sampling
        img_ds = img[0:(original_num_rows - 1):spatial_dsr, 0:(original_num_cols - 1):spatial_dsr]  # i:j:k = [i, i+k, i+2k, ..., j), where j can not be reached
        # disp = Image.fromarray(img_ds)
        # disp.show()

        # [3] threshold to binary {0, 1}, for both x_img and ground truth y_img
        # assign the activated pixels to be 1
        img_ds_th = np.where(img_ds > (255/2.), 0, 1)      # for model_index binary_wo_temporal_v{0, 1, 2, 3}

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

    # a data point is (x_img, y_img), per agent; a batch is an instance
    def __process_curr_data_point(self, sequence_num, curr_agent_idx):  # a private method
        # [0] get x_img
        x_chs_dsrows_dscols = self.__grab_and_process_single_img(sequence_num, curr_agent_idx, input_sample=True)  # x_img, whose chs==1
        # print('shape of x_chs_dsrows_dscols = {}'.format(x_chs_dsrows_dscols.shape))  # (chs=1, dsrows, dscols)

        # [1] get the corresponding y_img
        y_chs_dsrows_dscols = self.__grab_and_process_single_img(sequence_num, curr_agent_idx, input_sample=False)  # y_img, whose chs==1
        # print('shape of y_chs_dsrows_dscols = {}'.format(y_chs_dsrows_dscols.shape))

        # [2] append current data point to list of all data points
        return np.expand_dims(x_chs_dsrows_dscols, axis=0), np.expand_dims(y_chs_dsrows_dscols, axis=0)

    # a data point is (x_img, y_img), per agent; a batch is an instance
    # __getitem__() returns a complete batch
    # batch_index, which could be viewed as a shared variable by all processes, denotes the index of batch, from 4000 to 4999, in order.
    def __getitem__(self, batch_index):
        sequence_index = self.__sequence_indices_in_an_epoch[batch_index]
        sequence_num = sequence_index + 1
        print(' current testing batch num is {}'.format(sequence_num))

        # for an instance (Batch), prepare training/testing batch: (x_BS_Ch_Row_Col, y_BS_Ch_Row_Col), by
        # concatenating np array vertically from empty where BS = batch size, total number of valid agents in one instance
        list_x_all_points_in_a_batch = []
        list_y_all_points_in_a_batch = []

        # all x_imgs in current instance
        x_base_dir = '../0_Honglu_data_preprocessing/microscopic_domain/microscopic_image_samples_all_steps_binary/' + 'instance' + str(sequence_num) + '/'
        x_agents = glob.glob(x_base_dir + '*.png')

        # all y_imgs in current instance
        y_base_dir = '../0_Honglu_data_preprocessing/microscopic_domain/microscopic_labels_dsr_10_binary/' + 'instance' + str(sequence_num) + '/'
        y_imgs = glob.glob(y_base_dir + '*.png')

        # check whether the number of valid agents in x equals the number of valid agents in y, for the current instance
        assert (len(y_imgs) == len(x_agents))  # assert <==> raise error if not

        for curr_agent_idx in range(self.__B):
            curr_agent_dir_x = x_base_dir + 'agent' + str(curr_agent_idx) + '_all_steps.png'
            curr_agent_dir_y = y_base_dir + 'agent' + str(curr_agent_idx) + '.png'

            if (curr_agent_dir_x in x_agents) and (curr_agent_dir_y in y_imgs):  # current agent is valid
                x_1_chs_dsrows_dscols, y_1_chs_dsrows_dscols = self.__process_curr_data_point(sequence_num, curr_agent_idx)  # return one data point (one agent)
                list_x_all_points_in_a_batch.append(x_1_chs_dsrows_dscols)
                list_y_all_points_in_a_batch.append(y_1_chs_dsrows_dscols)
                # print('agent index {} matches in (x, y)'.format(curr_agent_idx))
            elif (curr_agent_dir_x not in x_agents) and (curr_agent_dir_y not in y_imgs):  # current agent is invalid
                # print(' agent index {} is invalid'.format(curr_agent_idx))
                continue
            else:
                # print(' agent index {} does not match in (x, y)'.format(curr_agent_idx))
                continue

        x_BS_Ch_Row_Col = np.concatenate(list_x_all_points_in_a_batch, axis=0)
        # print(' shape of x_BS_Ch_Row_Col = {}'.format(x_BS_Ch_Row_Col.shape))

        y_BS_Ch_Row_Col = np.concatenate(list_y_all_points_in_a_batch, axis=0)
        # print(' shape of y_BS_Ch_Row_Col = {}'.format(y_BS_Ch_Row_Col.shape))

        # a single output yielded by the generator makes a single batch. Different batches may have different sizes.
        # return a batch of input samples
        return x_BS_Ch_Row_Col  # do not return (x_BS_Ch_Row_Col, y_BS_Ch_Row_Col)


def weighted_binary_crossentropy_v1(y_true, y_pred):
    # compared with weighted_binary_crossentropy_v0, the weights for pos and neg elements are swapped
    #
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

    model_index = 'binary_wo_temporal_v3'   # binary_wo_temporal_v{0, 1, 2, 3}
    seq_starting_index = 4000
    num_seq = 1000   # number of test instances. An instance is a batch. A data point is (x_sequence, y_img), per agent;
    B = 30           # maximal possible number of valid agents in an instance

    # [1] load a trained model
    dir_model = './models/micro_wo_disentangle_' + model_index + '.h5'

    if (model_index == 'binary_wo_temporal_v0') or (model_index == 'binary_wo_temporal_v1'):
        model = keras.models.load_model(dir_model)
    elif model_index == 'binary_wo_temporal_v2':
        model = keras.models.load_model(dir_model, custom_objects={'weighted_binary_crossentropy_v1': weighted_binary_crossentropy_v1})
    elif model_index == 'binary_wo_temporal_v3':
        model = keras.models.load_model(dir_model, custom_objects={'soft_dice_loss': soft_dice_loss})
    else:
        print('Could not load the corresponding model.')

    # [2] prepare testing_batch_generator
    testing_batch_gen = BatchGenerator(testing_batch=True)

    # [3] predict y label. Computation is done in batches.
    # raw_pred_y_ALL_Ch_Row_Col is the raw predicted y labels, with ALL (29175) the total number of agents over all testing instances, Ch==1
    # The prediction is per element (per pixel). With binary cross-entropy loss, it is the predicted probability that the current element belongs to label 1
    tic()
    raw_pred_y_ALL_Ch_Row_Col = model.predict_generator(testing_batch_gen,
                                                        steps=num_seq,              # total number of batches to yield from generator before stopping
                                                        max_queue_size=12,
                                                        workers=12,
                                                        use_multiprocessing=True,
                                                        verbose=0)
    tac()
    print('shape of raw_pred_y_ALL_Ch_Row_Col = {}'.format(raw_pred_y_ALL_Ch_Row_Col.shape))  # (29175, 1, 128, 128)
    print('')

    dir_raw_y = './predicted_labels_binary/' + model_index
    if not os.path.exists(dir_raw_y):
        os.makedirs(dir_raw_y)
    np.save(dir_raw_y + '/raw_pred_y_ALL_Ch_Row_Col.npy', raw_pred_y_ALL_Ch_Row_Col)
    #raw_pred_y_ALL_Ch_Row_Col = np.load(dir_raw_y + '/raw_pred_y_ALL_Ch_Row_Col.npy')

    # [3] process and save prediction
    point_index = 0
    for seq_idx in range(seq_starting_index, seq_starting_index+num_seq):
        sequence_num = seq_idx + 1

        print('The current test instance is of number {}'.format(sequence_num))
        dir_y = './predicted_labels_binary/' + model_index + '/instance' + str(sequence_num)
        if not os.path.exists(dir_y):
            os.makedirs(dir_y)

        # all x_sequences in current instance
        x_base_dir = '../0_Honglu_data_preprocessing/microscopic_domain/microscopic_image_samples_all_steps_binary/' + 'instance' + str(sequence_num) + '/'
        x_imgs = glob.glob(x_base_dir + '*.png')

        # all y_imgs in current instance
        y_base_dir = '../0_Honglu_data_preprocessing/microscopic_domain/microscopic_labels_dsr_10_binary/' + 'instance' + str(sequence_num) + '/'
        y_imgs = glob.glob(y_base_dir + '*.png')

        # check whether the number of valid agents in x equals the number of valid agents in y, for the current instance
        assert (len(y_imgs) == len(x_imgs))  # assert <==> raise error if not

        for curr_agent_idx in range(B):
            curr_agent_dir_x = x_base_dir + 'agent' + str(curr_agent_idx) + '_all_steps.png'
            curr_agent_dir_y = y_base_dir + 'agent' + str(curr_agent_idx) + '.png'

            if (curr_agent_dir_x in x_imgs) and (curr_agent_dir_y in y_imgs):      # current agent is valid
                raw_pred_y_Ch_Row_Col = raw_pred_y_ALL_Ch_Row_Col[point_index, ...]  # get a predicted label for a single point (i.e., a single agent)
                pred_y_Row_Col_Ch = inv_process_img(raw_pred_y_Ch_Row_Col)           # input Ch==1, output Ch==3
                misc.imsave(dir_y + '/RawPredY_instanceNum{}_agentIdx{}.png'.format(sequence_num, curr_agent_idx), pred_y_Row_Col_Ch)
                point_index += 1
                # print(' agent index {} matches in (x, y)'.format(curr_agent_idx))
            elif (curr_agent_dir_x not in x_imgs) and (curr_agent_dir_y not in y_imgs):  # current agent is invalid
                print(' agent index {} is invalid'.format(curr_agent_idx))
                continue
            else:
                print(' agent index {} does not match in (x, y)'.format(curr_agent_idx))
                continue

    print('Finished testing, with total number of testing points {}'.format(point_index))







