# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 10:40:21 2020

@author: ww
"""



from tflearn.layers.conv import global_avg_pool
import tensorflow as tf
import tensorflow.contrib.layers as ly
import os
from mat_convert_to_tfrecord_p_end import read_and_decode
from utils3 import conv, lrelu
#import h5py
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def carafe(x, weight_dedcay, i, scale=2, k_up=5):
    b, h, w, c = x.get_shape().as_list()
    h_, w_  = h*scale, w*scale
    
    w = ly.conv2d(x, num_outputs = c, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.relu,
                          reuse = tf.AUTO_REUSE, scope='w'+str(i))    
    w = ly.conv2d(w, num_outputs = (scale*k_up)**2, kernel_size = 3, stride = 1,
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = None,
                          reuse = tf.AUTO_REUSE, scope='w1'+str(i))    
    w = tf.depth_to_space(w, 2)
    w = tf.nn.softmax(w, axis=-1)
    
    x = tf.image.resize_images(x, [h_, w_], method=1)
    x = tf.extract_image_patches(x, ksizes=[1,k_up, k_up, 1], strides=[1,1,1,1], rates=[1,scale,scale,1], padding='SAME')
    x = tf.reshape(x, (b, h_, w_, -1, c))
    
    x = tf.einsum('abcd,abcde->abce', w, x)
    #print(x.get_shape().as_list())
    return x
    
def upsample(x, weight_decay, reuse=True):
    with tf.variable_scope('up_net'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()
    for i in range(3):
        x = carafe(x, weight_decay, i)
    return x

def fusion_net(Z, Y, weight_decay, num_spectral = 31, num_fm = 64, num_ite=8, reuse=False):
    
    with tf.variable_scope('fusion_net'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        X = Fusion(Z, Y, weight_decay)
        Xs = X
        
        for i in range(num_ite):
            X = boost_lap(X, Z, Y, weight_decay)
            
            Xs = tf.concat([Xs, X], axis=3)
    
        X = conv(Xs, num_spectral, weight_decay, use_bias=False, scope='out_conv')

    return X

def boost_lap(X, Z_in, Y_in, weight_decay, num_spectral = 31, num_fm = 64, reuse= True):
    
    with tf.variable_scope('recursive'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        Z = conv(X, 3, weight_decay, scope='dz')
        Z = lrelu(Z)
        
        Y = conv(X, num_spectral, weight_decay, kernel=12, stride=8, scope='dy')
        Y = lrelu(Y)

        dZ = Z_in - Z
        dY = Y_in - Y
    
        dX = Fusion(dZ, dY, weight_decay)
        X = X + dX
    
    
    return X
    
    
def Fusion(Z, Y, weight_decay, num_spectral = 31, num_fm = 64, reuse = True):
    
    with tf.variable_scope('py'):        
        if reuse:
            tf.get_variable_scope().reuse_variables() 
            
        lms = upsample(Y, weight_decay)
        Xin = tf.concat([lms,Z], axis=3)
        Xt = conv(Xin, num_fm, weight_decay, scope="in")
        Xt = lrelu(Xt)

        for i in range(4):
            Xi = conv(Xt, num_fm, weight_decay, scope="res"+str(i)+"1")
            Xi = lrelu(Xi)
            Xi = conv(Xi, num_fm, weight_decay, scope="res"+str(i)+"2")
            
            mask = global_avg_pool(Xi)
            mask = tf.layers.dense(inputs=mask, units=num_fm//16, use_bias=True, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="se"+str(i)+"1")
            mask = tf.layers.dense(inputs=mask, units=num_fm, use_bias=True, reuse = tf.AUTO_REUSE, name="se"+str(i)+"2")
            mask = tf.reshape(mask, [-1, 1, 1, num_fm])
            mask = tf.sigmoid(mask)
            Xi = tf.multiply(Xi, mask)
            Xt = Xt + Xi
        X = conv(Xt, num_spectral, weight_decay, scope="out")
        
        return X      
                          

if __name__ =='__main__':

    tf.reset_default_graph()   

    train_batch_size = 32 # training batch size
    test_batch_size = 32
    image_size = 64      # patch size
    iterations = 251000 # total number of iterations to use.
    model_directory = './models_ibp_sn22/' # directory to save trained model to.
    #train_data_name = './training_data/train.mat'  # training data
    train_data = './training_data/trainlap.tfrecords'  # training data
    restore = False  # load model or not
    method = 'Adam'  # training method: Adam or SGD
    weight_decay = 1e-5

############## placeholder for training
    gt = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,31])
    ms = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size//8,image_size//8,31])
    pan = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,3])
    lr = tf.placeholder(dtype = tf.float32,shape = [])
    pan_batch, gt_batch, ms_batch = read_and_decode(train_data, batch_size=train_batch_size)

######## network architecture
    X = fusion_net(pan, ms, weight_decay)
    

######## loss function
    mse = tf.reduce_mean(tf.abs(X - gt))

##### Loss summary
    mse_loss_sum = tf.summary.scalar("mse_loss",mse)
 
    all_sum = tf.summary.merge([mse_loss_sum])
         
    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'fusion_net')    

    
    if method == 'Adam':
        g_optim = tf.train.AdamOptimizer(lr, beta1 = 0.9) \
                          .minimize(mse, var_list=t_vars)

    else:
        global_steps = tf.Variable(0,trainable = False)
        lr = tf.train.exponential_decay(0.1,global_steps,decay_steps = 50000, decay_rate = 0.1)
        clip_value = 0.1/lr
        optim = tf.train.MomentumOptimizer(lr,0.9)
        gradient, var   = zip(*optim.compute_gradients(mse,var_test_mselist = t_vars))
        gradient, _ = tf.clip_by_global_norm(gradient,clip_value)
        g_optim = optim.apply_gradients(zip(gradient,var),global_step = global_steps)
        
##### GPU setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)

#    init = tf.global_variables_initializer()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)#2979432    
    
    saver = tf.train.Saver()
    with tf.Session() as sess:  
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
 
        if restore:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess,ckpt.model_checkpoint_path)
            

        
        for i in range(iterations):
            
            if i >= 0 and i <= 20000:
                LR = 4e-4
            elif i>20000 and i <= 60000:
                LR = 2e-4
            elif i>60000 and i <= 1400000:
                LR = 1e-4
            else:
                LR = 5e-5
            train_pan, train_gt, train_ms = sess.run([pan_batch, gt_batch, ms_batch])
            _,mse_loss,merged = sess.run([g_optim, mse,all_sum],feed_dict = {gt: train_gt, ms: train_ms,
                                          pan: train_pan, lr:LR})

            if i % 100 == 0:

                print ("Iter: " + str(i) + " MSE: " + str(mse_loss))
                
            if i % 10000 == 0 and i != 0:             
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess,model_directory+'/model-'+str(i)+'.ckpt')
                print ("Save Model")
        coord.request_stop()
        coord.join(threads)
