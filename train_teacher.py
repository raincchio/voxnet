from __future__ import print_function

import argparse
import os
import time

import SimpleITK as sitk
import numpy as np
import tensorflow as tf

from BatchDataReader import BatchDataset
# from model import DenseVoxNet
from model.sparsevoxnet_dsc import SparseVoxNet_DSC
from model.sparsevoxnet_ds import SparseVoxNet_DS
from model.sparsevoxnet_d import SparseVoxNet_D
from model.sparsevoxnet import SparseVoxNet
from model.densevoxnet import DenseVoxNet
from utils import pre_process_isotropic, generate_score_map_partition, generate_score_map_patch2Img, partition_Img, \
    patches2Img_vote, imresize3d, RemoveMinorCC, saveLableImage, evaluation

parser = argparse.ArgumentParser(description="train VoxDenseNet")
parser.add_argument("--iteration", "-i", default=15000, type=int, help="default 15000 number of iterations, default=15000")
parser.add_argument("--dispaly_step", "-s", default=1000, type=int, help="number of steps to display, default=1000")
parser.add_argument("--train_file", "-f", default="train.txt", type=str, help="file of training dataset")
parser.add_argument("--test_file", "-t", default="test.txt", type=str, help="file of training dataset")
parser.add_argument("--random_crop", default=True, type=bool, help="random crop to images")
parser.add_argument("--batch_size", default=3, type=int, help="batch size, default=3")
parser.add_argument("--input_size", default=64, type=int, help="shape of input for the network, default=64")
parser.add_argument("--learning_rate", "-r", default=0.01, type=float, help="learning rate, default=0.05")
parser.add_argument("--logs_dir", default="logs/", type=str, help="location of trained model")
parser.add_argument("--data_zoo", default="data/", type=str, help="location of data")
parser.add_argument("--n_classes", default=3, type=int, help="number of classes")
parser.add_argument("--mode", default="test", type=str, help="train/test")
parser.add_argument("--model", default="DenseVoxNet", type=str, help="model:DenseVoxNet,SparseVoxNet,SparseVoxNet_D")
args = parser.parse_args()
train_params = vars(args)
print("Params:")
for k, v in train_params.items():
    print("%s: %s" % (k, v))

print("Initialize the model...")
if args.mode == "train":
    is_training = True
else:
    is_training = False
if args.model == 'DenseVoxNet':
    net = DenseVoxNet(is_training=is_training)
    args.logs_dir += 'densevoxnet/'
elif args.model == 'SparseVoxNet':
    net = SparseVoxNet(is_training=is_training)
    args.logs_dir += 'sparsevoxnet1/'
elif args.model == 'SparseVoxNet_D':
    net = SparseVoxNet_D(is_training=is_training)
    args.logs_dir += 'sparsevoxnet_d/'
elif args.model == 'SparseVoxNet_DS':
    net = SparseVoxNet_DS(is_training=is_training)
    args.logs_dir += 'sparsevoxnet_ds/'
elif args.model == 'SparseVoxNet_DSC':
    net = SparseVoxNet_DSC(is_training=is_training)
    args.logs_dir += 'sparsevoxnet_dsc/'
####参数输出####
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)


shape = [None, args.input_size, args.input_size, args.input_size, 1]
image = tf.placeholder(tf.float32, shape=shape, name="input_image")
label = tf.placeholder(tf.int32, shape=shape, name="label")

logits1, logits2, prob_map, pred_annotation = net(image)
loss1 = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=tf.squeeze(label, squeeze_dims=[4]),
                                                   name="entropy_1"))

loss2 = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=tf.squeeze(label, squeeze_dims=[4]),
                                                   name="entropy_2"))
loss = loss1 + 0.33 * loss2

accuracy = tf.reduce_mean(
    tf.cast(
        tf.equal(tf.cast(pred_annotation, dtype=tf.int32), tf.squeeze(label, squeeze_dims=[4])), "float"))
tf.summary.scalar("entropy", loss)

tf.summary.scalar("accuracy", accuracy)
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
end_learning_rate = 0.001
decay_steps = 5000
learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                          decay_steps, end_learning_rate,
                                          power=0.5, cycle=True)
tf.summary.scalar("learning_rate", learning_rate)

print("Setting up train op...")
train_op = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
# train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

print("Setting up summary op...")
summary_op = tf.summary.merge_all()

print("Setting up image reader...")
with open(args.train_file) as f:
    train_dataset_list = f.read().splitlines()
with open(args.test_file) as f:
    test_dataset_list = f.read().splitlines()

print("Setting up dataset reader...")
train_dataset_reader = BatchDataset(train_dataset_list)
test_dataset_reader = BatchDataset(test_dataset_list)
print("Setting up Saver...")
saver = tf.train.Saver()

if args.mode == "train":
    summary_writer = tf.summary.FileWriter(args.logs_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    min_loss = 0
    for itr in range(args.iteration + 1):
        start_time = time.time()
        train_images, train_labels = train_dataset_reader.next_batch(args.batch_size, args.random_crop,
                                                                     args.input_size)
        test_images, test_labels = test_dataset_reader.next_batch(args.batch_size, args.random_crop,
                                                                  args.input_size)
        feed_dict_train = {image: train_images, label: train_labels}
        feed_dict_test = {image: test_images, label: test_labels}

        tmp_, train_loss = sess.run([train_op, loss], feed_dict=feed_dict_train)
        # tmp_1, train_loss1 = sess.run([loss1, loss2], feed_dict=feed_dict_train)

        test_accuracy = sess.run(accuracy, feed_dict=feed_dict_test)

        print("Step: %d, Train loss: %g" % (itr, train_loss))
        print("Step: %d, Test accuracy: %g, Time: %gs" % (itr, test_accuracy, time.time() - start_time))

        if itr % 20 == 0:
            summary_str = sess.run(summary_op, feed_dict=feed_dict_train)
            summary_writer.add_summary(summary_str, itr)

            if test_accuracy > min_loss:
                """valid"""
                min_loss = test_accuracy
                saver.save(sess, args.logs_dir + "model.ckpt", itr)
        #
        # if itr % 10 == 0:
        #     """valid"""
        #     saver.save(sess, args.logs_dir + "model.ckpt", itr)

else:
    # parameters
    ckpt = tf.train.get_checkpoint_state(args.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    #this is a good test

    use_isotropic = 0
    ita = 4
    tr_mean = 0
    dice_t = [0] * 2
    asd_t = [0] * 2
    has_t = [0] * 2
    for id in range(0, 1):
        # read data and pre-processing
        tic = time.time()
        # id=17
        print("test sample #%d\n" % id)

        vol_path = os.path.join(args.data_zoo, "training_axial_crop_pat" + str(id) + ".nii.gz")
        vol_label = os.path.join(args.data_zoo, "training_axial_crop_pat" + str(id) + "-label.nii.gz")
        # vol_path = os.path.join('test/', "testing_axial_crop_pat" + str(id) + ".nii.gz")
        # vol_label = vol_path
        vol_src = sitk.GetArrayFromImage(sitk.ReadImage(vol_path))
        vol_label = sitk.GetArrayFromImage(sitk.ReadImage(vol_label))

        # vol_src = np.transpose(vol_src, [0, 2, 1])
        # vol_label = np.transpose(vol_label, [0, 2, 1])
        vol_data, vol_label = pre_process_isotropic(vol_src, vol_label, use_isotropic, id)
        data = vol_data - tr_mean

        # average fusion scheme
        # patch_list_avg, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z = \
        #     generate_score_map_partition(data, args.input_size, ita)
        #
        # # patch_list_label, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z = \
        # #     generate_score_map_partition(vol_label, args.input_size, ita)
        #
        # res_list_avg = []
        # for i in range(len(patch_list_avg)):
        #     crop_data = patch_list_avg[i]
        #     # crop_label = patch_list_label[i]
        #     # crop_data = np.transpose(crop_data, [2, 1, 0])
        #     crop_data = np.expand_dims(np.expand_dims(crop_data, axis=0), axis=4)
        #     crop_label = np.zeros((1, args.input_size, args.input_size, args.input_size, 1))
        #     res_score = sess.run(prob_map, feed_dict={image: crop_data, label: crop_label})
        #     res_score = np.squeeze(res_score)
        #
        #     res_list_avg.append(res_score)
        #
        # avg_label = generate_score_map_patch2Img(res_list_avg, ss_h, ss_w, ss_l, padding_size_x, padding_size_y,
        #                                          padding_size_z, args.input_size, ita)

        # major voting scheme
        pred_list = []
        patch_list_vote, r, c, h = partition_Img(data, args.input_size, ita)

        for i in range(len(patch_list_vote)):
            crop_data = patch_list_vote[i]
            # crop_data = np.transpose(crop_data, [2, 1, 0])
            crop_data = np.expand_dims(np.expand_dims(crop_data, axis=0), axis=4)
            crop_label = np.zeros((1, args.input_size, args.input_size, args.input_size, 1))
            res_L2, _ = sess.run([prob_map,pred_annotation], feed_dict={image: crop_data, label: crop_label})
            print(i,len(patch_list_vote))
            res_L2 = np.squeeze(res_L2)
            res_lable = np.argmax(res_L2, axis=3)
            pred_list.append(res_lable)

        vote_label = patches2Img_vote(pred_list, r, c, h, args.input_size, ita)

        # post-processing
        if use_isotropic == 1:
            vote_label = imresize3d(vote_label, [], vol_src.shape, 'reflect')
            # avg_label = imresize3d(avg_label, [], vol_src.shape, 'reflect')

        # remove minor connected components
        vote_label = RemoveMinorCC(vote_label, 0.2)
        # avg_label = RemoveMinorCC(avg_label, 0.2)

        # show predicted label picture, optional
        vote_image = saveLableImage(vote_label, './predict_label_vote')
        # avg_image = saveLableImage(avg_label, './predict_label_avg')

        # compute evaluation Index
        vote_dice, vote_adb, vote_hau = evaluation(vote_label, vol_label)
        # avg_dice, avg_adb, avg_hau = evaluation(avg_label, vol_label)

        dice_t[0] += vote_dice[0]
        dice_t[1] += vote_dice[1]
        asd_t[0] += (vote_adb[0][0]+vote_adb[0][1])/2
        asd_t[1] += (vote_adb[1][0]+vote_adb[1][1])/2
        has_t[0] += vote_hau[0]
        has_t[1] += vote_hau[1]

        print('心肌dice系数得分为：'+str(vote_dice[0])+'  abd系数得分为：'+str(vote_adb[0])+'  hasu系数得分为：'+str(vote_hau[0])+'\n')
        print('血池dice系数得分为：' + str(vote_dice[1]) + '  abd系数得分为：' + str(vote_adb[1]) + '  hasu系数得分为：' + str(vote_hau[1]) + '\n')

    # print('心肌dice系数得分为：' + str(dice_t[0] / 10) + '  abd系数得分为：' + str(asd_t[0] / 10) + '  hasu系数得分为：' + str(
    #     has_t[0] / 10) + '\n')
    # print('血池dice系数得分为：' + str(dice_t[1] / 10) + '  abd系数得分为：' + str(asd_t[1] / 10) + '  hasu系数得分为：' + str(
    #     has_t[1] / 10) + '\n')


