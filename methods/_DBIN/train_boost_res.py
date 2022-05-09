# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:58:30 2019

@author: ww
"""



import tensorflow as tf
import tensorflow.contrib.layers as ly
import os
from mat_convert_to_tfrecord_p_end import read_and_decode
#import h5py
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def fusion_net(Z, Y, num_spectral = 31, num_fm = 128, num_ite=5, reuse=False):
    
    with tf.variable_scope('fusion_net'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        X = Fusion(Z, Y)
        Xs = X
        
        for i in range(num_ite):
            X = boost_lap(X, Z, Y)
            
            Xs = tf.concat([Xs, X], axis=3)
    
        X = ly.conv2d(Xs, num_outputs = num_spectral, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = None)
    return X

def boost_lap(X, Z_in, Y_in, num_spectral = 31, num_fm = 128, reuse= True):
    
    with tf.variable_scope('recursive'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        Z = ly.conv2d(X, num_outputs = 3, kernel_size = 3, stride = 1, 
                      weights_regularizer = ly.l2_regularizer(weight_decay), 
                      weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu,
                      reuse = tf.AUTO_REUSE, scope='dz')
        
        Y =  ly.conv2d(X, num_outputs = num_spectral, kernel_size = 12, stride = 8, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, 
                          reuse = tf.AUTO_REUSE, scope='dy')
    
        dZ = Z_in - Z
        dY = Y_in - Y
    
        dX = Fusion(dZ, dY)
        X = X + dX
    
    
    return X
    
    
def Fusion(Z, Y, num_spectral = 31, num_fm = 128, reuse = True):
    
    with tf.variable_scope('py'):        
        if reuse:
            tf.get_variable_scope().reuse_variables() 
            
        lms = ly.conv2d_transpose(Y,num_spectral,12,8,activation_fn = None,
                                   weights_initializer = ly.variance_scaling_initializer(), 
                                   weights_regularizer = ly.l2_regularizer(weight_decay), reuse = tf.AUTO_REUSE, scope="lms")
        Xin = tf.concat([lms,Z], axis=3)
        Xt = ly.conv2d(Xin, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="in")
        for i in range(4):
            Xi = ly.conv2d(Xt, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="res"+str(i)+"1")
            Xi = ly.conv2d(Xi, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="res"+str(i)+"2")
            Xt = Xt + Xi    
        X = ly.conv2d(Xt, num_outputs = num_spectral, kernel_size = 3, stride = 1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="out")
    
        return X      
    
                          

if __name__ =='__main__':

    tf.reset_default_graph()   

    train_batch_size = 32 # training batch size
    test_batch_size = 32
    image_size = 64      # patch size
    iterations = 251000 # total number of iterations to use.
    model_directory = './models_boost_res/' # directory to save trained model to.
    #train_data_name = './training_data/train.mat'  # training data
    train_data = './training_data/trainlap.tfrecords'  # training data
    restore = False  # load model or not
    method = 'Adam'  # training method: Adam or SGD
    weight_decay = 2e-5

############## placeholder for training
    gt = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,31])
    ms = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size//8,image_size//8,31])
    pan = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,3])
    lr = tf.placeholder(dtype = tf.float32,shape = [])
    pan_batch, pan2_batch, pan4_batch, gt_batch,gt2_batch,gt4_batch, ms_batch = read_and_decode(train_data, batch_size=train_batch_size)

######## network architecture
    X = fusion_net(pan, ms)
    

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
    config.gpu_options.allow_growth = True
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
        sess.close()