import tensorflow as tf
import numpy as np
from functions import checkFile, generateRandomList
import scipy.io as sio
import time
from skimage.measure import compare_psnr
import spectral as sp
import matplotlib.pyplot as plt


class tonwmd(object):
    '''
    implements of TONWMD
    '''

    def __init__(self, channel, mschannel, train_pieces_path, valid_pieces_path, model_save_path,
                 test_data_label_path, output_save_path, total_num, valid_num, train_batch_size=64,
                 valid_batch_size=16, piece_size=32, ratio=8, maxpower=20, test_start=21,
                 test_end=32, test_height=512, test_width=512):
        # self.choose_dataset(num)

        self.setDataAbout(channel, mschannel, train_pieces_path, valid_pieces_path, model_save_path,
                          test_data_label_path, output_save_path, total_num, valid_num, train_batch_size,
                          valid_batch_size, piece_size, ratio, maxpower, test_start, test_end,
                          test_height, test_width)

        self.data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channels], name='Xes')
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channels], name='Xrel')
        self.istraining = tf.placeholder(dtype=tf.bool, shape=[], name='istraining')
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        self.running_size = tf.placeholder(dtype=tf.int32, shape=[], name='running_size')

        self.weightlist = []
        self.resultlist = []
        self.helplist = []
        self.losslist1 = []
        self.losslist2 = []

        checkFile(self.model_save_path)

    def setDataAbout(self, channel, mschannel, train_pieces_path, valid_pieces_path, model_save_path,
                     test_data_label_path, output_save_path, total_num, valid_num, train_batch_size,
                     valid_batch_size, piece_size, ratio, maxpower, test_start, test_end, test_height,
                     test_width):

        self.channels = channel
        self.mschannels = mschannel

        self.model_save_path = model_save_path
        self.train_pieces_path = train_pieces_path
        self.valid_pieces_path = valid_pieces_path


        self.test_data_label_path = test_data_label_path
        self.output_save_path = output_save_path

        self.total_num = total_num
        self.valid_num = valid_num

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.piece_size = piece_size
        self.ratio = ratio

        # maxpower to select model
        self.maxpower = maxpower

        self.test_start = test_start
        self.test_end = test_end
        self.test_height = test_height
        self.test_width = test_width

    def activator(self, X, features, atname):
        if atname == 'prelu':
            self.alphas = tf.Variable(tf.constant(0.1, shape=[features]), name="prelu_alphas")
            return tf.nn.relu(X) + tf.multiply(self.alphas, (X - tf.abs(X))) * 0.5
        elif atname == 'relu':
            return tf.nn.relu(X)
            pass

    def buildConv(self, X, name, cnnSize, inputNum, outputNum, useBias=False, useActivator=True, activator='prelu',
                  useBatchNorm=False, addweights=True):
        with tf.variable_scope(name):
            weight = self.getWeight([cnnSize, cnnSize, inputNum, outputNum], name='W')
            output = self.convolution(X, weight)
            if useBias:
                bias = self.getBias([outputNum], name='b')
                output = tf.add(output, bias)
            if useBatchNorm:
                output = tf.layers.batch_normalization(output, training=self.istraining, name='BN')
            if useActivator:
                output = self.activator(output, outputNum, activator)
            if addweights:
                self.weightlist.append(weight)
                self.resultlist.append(output)
            return output

    def Build_Region_Level_NL(self, X, num):
        # to avoid the size too large to compute, we simply divide the feature into 2*2 parts
        # and after passing the non-local module, we concat them
        h = w = self.running_size
        cuth = h // 2
        cutw = w // 2
        c = num
        left_top = X[:, :cuth, :cutw, :]
        right_top = X[:, :cuth, cutw:, :]
        left_bottom = X[:, cuth:, :cutw, :]
        right_bottom = X[:, cuth:, cutw:, :]
        left_top = self.BuildNL(left_top, 'left_top', cuth, cutw, c)
        right_top = self.BuildNL(right_top, 'right_top', cuth, cutw, c)
        left_bottom = self.BuildNL(left_bottom, 'left_bottom', cuth, cutw, c)
        right_bottom = self.BuildNL(right_bottom, 'right_bottom', cuth, cutw, c)

        top = tf.concat([left_top, right_top], axis=2)
        bottom = tf.concat([left_bottom, right_bottom], axis=2)
        X = tf.concat([top, bottom], axis=1)
        return X

    def BuildNL(self, X, name, h, w, c):
        '''
        Non-local module
        :param X:
        :param name:
        :param h:
        :param w:
        :param c:
        :return:
        '''
        with tf.variable_scope(name):
            g_x = self.buildConv(X, 'g_x', 1, c, c // 2, useActivator=False, addweights=False)
            g_x = tf.reshape(g_x, [-1, h * w, c // 2])
            # print(g_x.get_shape())

            theta_x = self.buildConv(X, 'theta_x', 1, c, c // 2, useActivator=False, addweights=False)
            theta_x = tf.reshape(theta_x, [-1, h * w, c // 2])
            phi_x = self.buildConv(X, 'phi_x', 1, c, c // 2, useActivator=False, addweights=False)
            phi_x = tf.reshape(phi_x, [-1, h * w, c // 2])
            phi_x = tf.transpose(phi_x, (0, 2, 1))
            f = tf.matmul(theta_x, phi_x)
            f_div_c = tf.nn.softmax(f, axis=-1)

            y = tf.matmul(f_div_c, g_x)
            y = tf.reshape(y, [-1, h, w, c // 2])
            y = self.buildConv(y, 'y_just', 1, c // 2, c, useActivator=False, addweights=False)
            return tf.add(y, X)

    def buildGraph(self):
        '''
        network composed of non-local module, skip-connection and inception
        :return:
        '''
        input_tensor = self.data
        input_num = self.channels
        # input_tensor = tf.concat([self.data,self.ms_data],axis=-1)
        # input_num = self.channels + self.mschannels
        output_num = 64
        # first conv to extract the shallow features
        output = self.buildConv(input_tensor, 'first_Conv', 3, input_num, 64, useActivator=False, addweights=False)
        # non-local module
        output = self.Build_Region_Level_NL(output, output_num)
        self.resultlist.append(output)
        # skip connection
        total_num = output_num
        num_list = [96, 76, 65, 55, 47, 39, 32]
        num1 = 64
        num2 = 32
        total_num += sum(num_list)
        for ind, val in enumerate(num_list):
            output = self.buildConv(output, 'conv%d' % ind, 3, output_num, val, useBias=True, useBatchNorm=True)
            output_num = val
        concat = tf.concat(self.resultlist, axis=3, name='concat1')
        # inception
        self.buildConv(concat, 'conv_in1', 1, total_num, num1, useBias=True)
        self.buildConv(concat, 'conv_in21', 1, total_num, num2, useBias=True)
        self.buildConv(self.resultlist[-1], 'conv_in22', 3, num2, num2, useBias=True)
        concat2 = tf.concat([self.resultlist[-1], self.resultlist[-3]], axis=3, name='concat2')
        # the final conv to adjust dimensions
        self.buildConv(concat2, 'final_just', 1, num1 + num2, self.channels, useActivator=False)
        self.output = tf.add(self.data, self.resultlist[-1], name='fusion')
        self.resultlist.clear()

    def initGpu(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

    def init_train_steps(self):
        self.initial_lr = 2e-3
        self.end_lr = 1e-7
        self.perdecay_epoch = 10
        self.epoch_completed = 0
        self.currnt_epoch = 0
        self.clipping_norm = 5
        self.l2_decay = 1e-4

        self.lr = self.initial_lr

        self.train_batch_input = np.zeros([self.train_batch_size, self.piece_size, self.piece_size, self.channels])
        # self.train_batch_ms = np.zeros([self.train_batch_size, self.piece_size, self.piece_size, self.mschannels])
        self.train_batch_label = np.zeros([self.train_batch_size, self.piece_size, self.piece_size, self.channels])
        self.valid_batch_input = np.zeros([self.valid_batch_size, self.piece_size, self.piece_size, self.channels])
        # self.valid_batch_ms = np.zeros([self.valid_batch_size, self.piece_size, self.piece_size, self.mschannels])
        self.valid_batch_label = np.zeros([self.valid_batch_size, self.piece_size, self.piece_size, self.channels])
        self.train_step = 0

    def buildOptimizaer(self):
        self.buildPSNRAndSSIM()
        self.buildLoss()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # self.training_optimizer = self.add_optimizer_op(self.loss, self.lr_input)
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # self.optimizer = optimizer.minimize(self.loss)
            trainables = tf.trainable_variables()
            grads = tf.gradients(self.loss, trainables)
            clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=5)
            grad_var_pairs = zip(clipped_grads, trainables)
            self.optimizer = optimizer.apply_gradients(grad_var_pairs)
            # gradients = optimizer.compute_gradients(self.loss)
            # capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            # self.optimizer = optimizer.apply_gradients(capped_gradients)

    def buildSaver(self):
        self.saver = tf.train.Saver(max_to_keep=100)
        checkFile(self.model_save_path)

    def buildPSNRAndSSIM(self):
        self.psnr = tf.reduce_mean(tf.image.psnr(self.label, self.output, max_val=1.0))
        self.ssim = tf.reduce_mean(tf.image.ssim(self.label, self.output, max_val=1.0))

    def loadTrainBatch(self):
        self.helplist.clear()
        generateRandomList(self.helplist, self.total_num, self.train_batch_size)
        for ind, val in enumerate(self.helplist):
            mat = sio.loadmat(self.train_pieces_path + '%d.mat' % val)
            self.train_batch_input[ind, :, :, :] = mat['data']
            self.train_batch_label[ind, :, :, :] = mat['label']

    def paintTrend(self):
        plt.title('loss-trend')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(self.losslist1, color='r')
        plt.plot(self.losslist2, color='g')
        plt.legend(['train', 'valid'])
        plt.show()

    def loadValidBatch(self):
        self.helplist.clear()
        generateRandomList(self.helplist, self.valid_num, self.valid_batch_size)
        for ind, val in enumerate(self.helplist):
            mat = sio.loadmat(self.valid_pieces_path + '%d.mat' % val)
            self.valid_batch_input[ind, :, :, :] = mat['data']
            self.valid_batch_label[ind, :, :, :] = mat['label']

    def buildLoss(self):
        # L1-norm as the spatial loss
        spatial_loss = tf.reduce_mean(tf.abs(self.output - self.label), name='spt_loss')
        # SAM as the spectral loss
        xcnn_norm = tf.sqrt(tf.reduce_sum(tf.square(self.output), axis=3))
        xes_norm = tf.sqrt(tf.reduce_sum(tf.square(self.label), axis=3))
        xcnn_xes = tf.reduce_sum(tf.multiply(self.output, self.label), axis=3)
        g = 1e-3
        cosin = xcnn_xes / ((xcnn_norm * xes_norm) + g)
        eta1 = 1e-3
        spectral_loss = tf.reduce_mean(tf.acos(cosin), name='spec_loss')
        # # SSIM related as the structural loss
        eta2 = 1e-3
        ssim = tf.image.ssim(self.label, self.output, max_val=1.0)
        structural_loss = tf.reduce_mean(tf.square(1 - ssim, name='stru_loss'))
        # weight_decay to speed the training
        l2_norm_losses = [tf.nn.l2_loss(w) for w in self.weightlist]
        l2_norm_loss = self.l2_decay * tf.add_n(l2_norm_losses)
        self.loss = spatial_loss + eta1 * spectral_loss + eta2 * structural_loss + l2_norm_loss

    def trainBatch(self, sess):
        self.loadTrainBatch()
        self.resultlist.clear()
        self.weightlist.clear()
        sess.run(self.optimizer, feed_dict={self.data: self.train_batch_input, self.label: self.train_batch_label,
                                            self.learning_rate: self.lr, self.istraining: True,
                                            self.running_size: self.piece_size})

    def endAllEpochs(self):
        # self.resultlist.clear()
        self.weightlist.clear()
        self.helplist.clear()
        print('training is finished')
        self.paintTrend()
        self.losslist1.clear()
        self.losslist2.clear()

    def epochManipulate(self):
        '''
        every 10 epoch to decay 0.5 until less than 1e-7
        :return:
        '''
        self.epoch_completed += 1
        if self.epoch_completed >= self.perdecay_epoch:
            self.epoch_completed = 0
            self.lr = 0.5 * self.lr

    def selectModel(self, sess):
        self.loadValidBatch()
        psnr_val, ssim_val, loss_val = self.session.run([self.psnr, self.ssim, self.loss],
                                                        feed_dict={self.data: self.train_batch_input,
                                                                   self.istraining: True,
                                                                   self.label: self.train_batch_label,
                                                                   self.learning_rate: self.lr,
                                                                   self.running_size:self.piece_size,
                                                                   })
        psnr_val2, ssim_val2, loss_val2 = self.session.run([self.psnr, self.ssim, self.loss],
                                                           feed_dict={self.data: self.valid_batch_input,
                                                                      self.istraining: False,
                                                                      self.label: self.valid_batch_label,
                                                                      self.running_size:self.piece_size
                                                                      })
        self.losslist1.append(loss_val)
        self.losslist2.append(loss_val2)
        print('epoch%s----train-----psnr:%s  ssim:%s  loss:%s-----' % (
            self.currnt_epoch, psnr_val, ssim_val, loss_val))
        print('     ----valid-----psnr:%s  ssim:%s  loss:%s-----' % (psnr_val2, ssim_val2, loss_val2))

        t = psnr_val2 * 0.5 + ssim_val2 * 0.5
        if t > self.maxpower:
            print('get a satisfying model')
            self.maxpower = t
            self.saver.save(sess, self.model_save_path, global_step=self.currnt_epoch)
            # print('one model saved successfully')

    def train(self):
        self.initGpu()
        self.init_train_steps()
        self.buildGraph()
        self.buildOptimizaer()
        self.buildSaver()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            while self.lr > self.end_lr:
                # self.start_epoch = 0
                # while self.currnt_epoch <= self.total_epoch:
                self.trainBatch(sess)
                self.train_step += 1
                if self.train_step * self.train_batch_size >= self.total_num:
                    self.train_step = 0
                    self.currnt_epoch += 1
                    self.selectModel(sess)
                    self.epochManipulate()
            self.endAllEpochs()

    def test(self):
        start = time.perf_counter()
        self.initGpu()
        self.buildGraph()
        self.saver = tf.train.Saver()
        psnr = 0

        b = self.train_batch_size
        h = self.test_height
        w = self.test_width

        # the orignal image is too large to be taken into the model directly, we divide it and then group
        piece_size = 32
        # the other are cut into 32*32 as the same size with trainging patches
        piece_count = (h // piece_size) * (w // piece_size)
        input_pieces = np.zeros([piece_count, piece_size, piece_size, self.channels], dtype=np.float32)
        checkFile(self.output_save_path)
        test_start = self.test_start
        test_end = self.test_end

        with self.session as sess:
            latest_model = tf.train.get_checkpoint_state(self.model_save_path)
            self.saver.restore(sess, latest_model.model_checkpoint_path)
            # self.saver.restore(sess,self.model_save_path+'-103')
            for i in range(test_start, test_end + 1):
                mat = sio.loadmat(self.test_data_label_path + '%d.mat' % i)
                data = mat['XES']
                X = mat['X']
                self.helplist.clear()
                count = 0
                icount = 0
                for x in range(0, h, piece_size):
                    for y in range(0, w, piece_size):
                        input_pieces[count, :, :, :] = data[x:x + piece_size, y:y + piece_size, :]
                        # input_pieces2[count, :, :, :] = Z[x:x + piece_size, y:y + piece_size, :]
                        count += 1
                while count >= b:
                    output = sess.run(self.output,
                                      feed_dict={self.data: input_pieces[icount * b:icount * b + b, :, :, :],
                                                 # self.ms_data: input_pieces2[icount * b:icount * b + b, :, :, :],
                                                 self.istraining: False, self.running_size: piece_size})
                    self.helplist.append(output)
                    count -= b
                    icount += 1
                if count > 0:
                    output = sess.run(self.output,
                                      feed_dict={self.data: input_pieces[icount * b:icount * b + count, :, :, :],
                                                 # self.ms_data: input_pieces2[icount * b:icount * b + count, :, :, :],
                                                 self.istraining: False, self.running_size: piece_size})
                    self.helplist.append(output)
                input_pieces = np.concatenate(self.helplist, axis=0)
                count = 0
                for x in range(0, h, piece_size):
                    for y in range(0, w, piece_size):
                        data[x:x + piece_size, y:y + piece_size, :] = input_pieces[count, :, :, :]
                        count += 1

                output = data
                output[output < 0] = 0
                output[output > 1] = 1.0
                sio.savemat(self.output_save_path + '%d.mat' % i, {'XCNN': output})
                # rgb = spectralDegrade(output, R)
                # plt.imshow(rgb)
                # plt.show()
                # psnr += compare_psnr(Z, rgb)
                psnr += compare_psnr(X, output)
                print('%d has finished' % i)
            print(psnr / (test_end - test_start + 1))
            print((test_end - test_start + 1))
            end = time.perf_counter()
            print('用时%ss' % ((end - start) / (test_end - test_start + 1)))

    def convolution(self, X, W):
        return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

    def getWeight(self, shape, name):
        nl = shape[0] * shape[1] * shape[2]
        # a = 0.1
        # sttdev = np.sqrt(2 / nl / (1 + a ** 2))
        sttdev = np.sqrt(2 / nl)
        return tf.Variable(tf.truncated_normal(shape, stddev=sttdev), name=name)

    def getBias(self, shape, name):
        return tf.Variable(tf.constant(0.0, shape=shape), name=name)


if __name__ == '__main__':
    pass
