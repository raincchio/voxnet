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
    patches2Img_vote, imresize3d, RemoveMinorCC, saveLableImage, evaluation, frame_similarity, kl_divergence, optimistic_restore

parser = argparse.ArgumentParser(description="train VoxDenseNet")
parser.add_argument("--iteration", "-i", default=15000, type=int, help="default 15000 number of iterations, default=15000")
parser.add_argument("--dispaly_step", "-s", default=1000, type=int, help="number of steps to display, default=1000")
parser.add_argument("--train_file", "-f", default="train.txt", type=str, help="file of training dataset")
parser.add_argument("--test_file", "-t", default="test.txt", type=str, help="file of training dataset")
parser.add_argument("--random_crop", default=True, type=bool, help="random crop to images")
parser.add_argument("--batch_size", default=1, type=int, help="batch size, default=3")
parser.add_argument("--input_size", default=64, type=int, help="shape of input for the network, default=64")
parser.add_argument("--learning_rate", "-r", default=0.00005, type=float, help="learning rate, default=0.05")
parser.add_argument("--logs_dir", default="logs_last/", type=str, help="location of trained model")
parser.add_argument("--data_zoo", default="data/", type=str, help="location of data")
parser.add_argument("--n_classes", default=3, type=int, help="number of classes")
parser.add_argument("--mode", default="train", type=str, help="train/test")
parser.add_argument("--model", default="DenseVoxNet", type=str, help="model:DenseVoxNet,SparseVoxNet,SparseVoxNet_D")
args = parser.parse_args()
train_params = vars(args)
print("Params:")
for k, v in train_params.items():
    print("%s: %s" % (k, v))

print("Initialize the model...")

####参数输出####
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

shape = [None, args.input_size, args.input_size, args.input_size, 1]
image1 = tf.placeholder(tf.float32, shape=shape, name="input_image")
label1 = tf.placeholder(tf.int32, shape=shape, name="label")
teacher_net = DenseVoxNet(is_training=False)
logits1_t, logits2_t, prob_map_t_, pred_annotation_t_ = teacher_net(image1)

#student
shape2 = [None, args.input_size, args.input_size, args.input_size, 3]
shape3 = [None, args.input_size, args.input_size, args.input_size]
image2 = tf.placeholder(tf.float32, shape=shape, name="input_image2")
label2 = tf.placeholder(tf.int32, shape=shape, name="label2")
prob_map_t = tf.placeholder(tf.float32, shape=shape2, name="prob_map_t")
pred_annotation_t = tf.placeholder(tf.float32, shape=shape3, name="pred_annotation_t")
student_net = SparseVoxNet_DS(is_training=True)

logit1, logit2, prob_map, pred_annotation, feature1_distilation, feature2_distilation, feature_t, feature1_distilation_logit, feature2_distilation_logit = student_net(image2)

loss1 = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit1, labels=tf.squeeze(label2, squeeze_dims=[4]),
                                                   name="entropy_1"))
loss2 = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit2, labels=tf.squeeze(label2, squeeze_dims=[4]),
                                                   name="entropy_2"))

kl_loss = kl_divergence(prob_map, prob_map_t,name='name="kl_divergence"')

frame_similarity_loss = frame_similarity(pred_annotation, pred_annotation_t, name="frame_similarity")


self_distilation_label_loss1 = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature1_distilation_logit, labels=tf.squeeze(label2, squeeze_dims=[4]),
                                                   name="self_distilation_label_loss1"))

self_distilation_label_loss2 = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature2_distilation_logit, labels=tf.squeeze(label2, squeeze_dims=[4]),
                                                   name="self_distilation_label_loss2"))

self_distilation_feature_loss1 = tf.reduce_mean(tf.squared_difference(feature1_distilation, feature_t))
self_distilation_feature_loss2 = tf.reduce_mean(tf.squared_difference(feature2_distilation, feature_t))


loss = loss1 + 0.33 * loss2+ 0.01 *kl_loss + 0.1 *frame_similarity_loss + self_distilation_label_loss1 + self_distilation_label_loss2 \
       + self_distilation_feature_loss1 + self_distilation_feature_loss2

accuracy = tf.reduce_mean(
    tf.cast(
        tf.equal(tf.cast(pred_annotation, dtype=tf.int32), tf.squeeze(label2, squeeze_dims=[4])), "float"))
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

train_op1 = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss1, global_step=global_step)
train_op2 = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss2, global_step=global_step)

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
    ckpt = tf.train.get_checkpoint_state('logs/densevoxnet')
    optimistic_restore(sess, ckpt.model_checkpoint_path)
    min_loss = 0
    for itr in range(args.iteration + 1):
        start_time = time.time()
        train_images, train_labels = train_dataset_reader.next_batch(args.batch_size, args.random_crop,
                                                                     args.input_size)
        test_images, test_labels = test_dataset_reader.next_batch(args.batch_size, args.random_crop,
                                                                  args.input_size)
        feed_dict_train = {image1: train_images, label1: train_labels}
        feed_dict_test = {image2: test_images, label2: test_labels}
        prob_map_, pred_annotation_ = sess.run([prob_map_t_, pred_annotation_t_], feed_dict=feed_dict_train)

        feed_dict_train_ = {image2: train_images, label2: train_labels, prob_map_t: prob_map_,pred_annotation_t: pred_annotation_ }
        tmp_, train_loss = sess.run([train_op, loss], feed_dict=feed_dict_train_)


        sess.run([train_op1], feed_dict=feed_dict_train_)
        if itr%5==0:
            sess.run([train_op1], feed_dict=feed_dict_train_)


        # loss1,loss2,kl_loss,frame_similarity_loss,self_distilation_label_loss1 ,self_distilation_label_loss2,self_distilation_feature_loss1
                          # ,self_distilation_feature_loss2
        test_accuracy = sess.run(accuracy, feed_dict=feed_dict_test)

        print("Step: %d, Train loss: %g" % (itr, train_loss))
        print("Step: %d, Test accuracy: %g, Time: %gs" % (itr, test_accuracy, time.time() - start_time))

        if itr % 20 == 0:
            summary_str = sess.run(summary_op, feed_dict=feed_dict_train_)
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
            res_L2 = sess.run(prob_map, feed_dict={image2: crop_data, label2: crop_label})
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


