
# 1_microscopic_wo_disentanglement

[1] This repository contains two encoder-decoder models:

	1_train_encoder_decoder_binary.py and 2_test_encoder_decoder_binary.py are paired files for training and testing the model with LSTM;

	1_train_encoder_decoder_binary_wo_temporal.py and 2_test_encoder_decoder_binary_wo_temporal.py are paired files for training and testing the model without LSTM;

[2] The script requres keras and tensorflow backend (with channel-first input format).

[3] You may simply change the model_index and the name of the loss function to train/store/test different models.

[4] If your computer does not support more than 12 cores for concurrency, you may change the parameter 'workers' to be a smaller value in the paired training script and testing script:

In training script:
model.fit_generator(training_batch_gen, steps_per_epoch=num_training_instances, epochs=max_num_epoch, validation_data=validation_batch_gen, validation_steps=num_testing_instances, max_queue_size=12,  workers=12, use_multiprocessing=True, shuffle=False, initial_epoch=0) 

In testing script:
model.predict_generator(testing_batch_gen, steps=num_seq, max_queue_size=12, workers=12, use_multiprocessing=True, verbose=0)

[5] For illustration of the data, I just uploaded microscopic_image_samples_all_steps_binary/instance1 and microscopic_labels_dsr_10_binary/instance1
I did not upload all instances and microscopic_image_samples_T_steps_binary due to the file number and size (more than 30 GB). You may come to me to copy the data. 
