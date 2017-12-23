# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 08:44:23 2017

@author: amitrc
"""

import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import scipy.misc as scimisc
import os
import time

def im2double(inp):
    return cv2.normalize(inp.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

def conv2d(inp, filt, stride=[1, 1, 1, 1]):
    return tf.nn.conv2d(inp, filt, strides=stride, padding = 'SAME')

def max_pool(inp, fil_size=[1, 2, 2, 1], stride=[1, 2, 2, 1]):
    return tf.nn.max_pool(inp, ksize=fil_size, strides=stride, padding = 'SAME')

def relu(inp):
    return tf.nn.relu(inp)

def lrelu(inp, leakiness=0.1):
    return tf.maximum(inp, leakiness*inp)

def fully_connected(x, w, b):
    return tf.matmul(x, w) + b

def batch_norm(x, epsilon=1e-5, momentum = 0.9, train=True, name=None):
    return tf.contrib.layers.batch_norm(x,
                      decay=momentum, 
                      updates_collections=None,
                      epsilon=epsilon,
                      scale=True,
                      is_training=train,
                      scope=name)

def convGRU(x, h, wxg, whg, bg, wxo, who, bo):
    # x -> [batch_size X xdim X ydim X input_channels]
    # h -> [1 X xdim X ydim X hidden_channels]
    
    xg = conv2d(x, wxg) # [batch_size * xdim * ydim * hidden_channels]
    hg = conv2d(h, whg) # [1 * xdim * ydim * hidden_channels]
    yg = xg + hg + bg
    
    r,u = tf.split(yg, 2, axis=3)
    r = tf.sigmoid(r)
    u = tf.sigmoid(u)
    
    xo = conv2d(x, wxo) 
    ho = conv2d(r*h, who)
    yo = tf.tanh(xo + ho + bo)
    
    ht = (1.0-u)*h + u*yo
    return ht

def build_model(xdim=64, ydim=None, b_size=64, inp_c=3, lr=0.002, T=8):
    if not ydim:
        ydim = xdim
    
    x = tf.placeholder(dtype=tf.float32, shape=[b_size, xdim, ydim, inp_c])
    y = tf.placeholder(dtype=tf.float32, shape=[b_size//T, xdim, ydim, inp_c])
    z = tf.placeholder(dtype=tf.float32, shape=[b_size//T, xdim, ydim, inp_c])
    prev_inp = tf.split(x, T, axis=0)
    grads = tf.zeros(shape=[b_size//T, xdim, ydim, inp_c])
    for i in range(1,T):
        tmp = (prev_inp[i]-prev_inp[i-1])**2
        grads += (tmp-tf.reduce_min(tmp))/(tf.reduce_max(tmp)-tf.reduce_min(tmp))
    grads /= float(T)
    prev_inp = prev_inp[-1]
    
    conv_0 = 3
    conv_0_c = 32
    conv_1 = 3
    conv_1_c = 32
    conv_2 = 3
    conv_2_c = 64
    conv_gru = 3
    conv_gru_c = 128
    deconv_1 = 3
    deconv_1_c = conv_2_c
    deconv_2 = 3
    deconv_2_c = conv_1_c
    deconv_3 = 3
    deconv_3_c = conv_1_c//2
    deconv_4 = 3
    deconv_4_c = inp_c
    deconv_5 = 3
    deconv_5_c = inp_c
    
    
    w_conv00 = tf.Variable(tf.truncated_normal([conv_0, conv_0, inp_c, conv_0_c], stddev=0.1))
    b_conv00 = tf.Variable(tf.truncated_normal([conv_0_c], stddev=0.1))
    hidden_conv00 = lrelu(conv2d(x, w_conv00) + b_conv00)
    w_conv01 = tf.Variable(tf.truncated_normal([conv_0, conv_0, conv_0_c, conv_0_c], stddev=0.1))
    b_conv01 = tf.Variable(tf.truncated_normal([conv_0_c], stddev=0.1))
    hidden_conv01 = lrelu(conv2d(hidden_conv00, w_conv01) + b_conv01)
    prev_conv0 = tf.split(hidden_conv01, T, axis=0)
    prev_conv0 = prev_conv0[-1]
    
    w_conv1 = tf.Variable(tf.truncated_normal([conv_1, conv_1, conv_0_c, conv_1_c], stddev=0.1))
    b_conv1 = tf.Variable(tf.truncated_normal([conv_1_c], stddev=0.1))
    hidden_conv1 = lrelu(conv2d(hidden_conv01, w_conv1) + b_conv1)
    hidden_pool1 = batch_norm(max_pool(hidden_conv1), name='conv1_bn')
    xdim_1 = xdim//2
    ydim_1 = ydim//2
    
    prev_conv1 = tf.split(hidden_pool1, T, axis=0)
    prev_conv1 = prev_conv1[-1]
    
    w_conv2 = tf.Variable(tf.truncated_normal([conv_2, conv_2, conv_1_c, conv_2_c], stddev=0.1))
    b_conv2 = tf.Variable(tf.truncated_normal([conv_2_c], stddev=0.1))
    hidden_conv2 = lrelu(conv2d(hidden_pool1, w_conv2) + b_conv2)
    hidden_pool2 = batch_norm(max_pool(hidden_conv2), name='conv2_bn')
    
    frames = tf.split(hidden_pool2, T, axis=0)
    prev_conv2 = frames[-1]
    
    h0 = tf.Variable(tf.zeros([1, frames[0].shape[1], frames[0].shape[2], conv_gru_c], dtype=tf.float32))
    wxg = tf.Variable(tf.truncated_normal([conv_gru, conv_gru, conv_2_c, 2*conv_gru_c]))
    whg = tf.Variable(tf.truncated_normal([conv_gru, conv_gru, conv_gru_c, 2*conv_gru_c]))
    bg = tf.Variable(tf.truncated_normal([2*conv_gru_c], stddev=0.1))
    wxo = tf.Variable(tf.truncated_normal([conv_gru, conv_gru, conv_2_c, conv_gru_c]))
    who = tf.Variable(tf.truncated_normal([conv_gru, conv_gru, conv_gru_c, conv_gru_c]))
    bo = tf.Variable(tf.truncated_normal([conv_gru_c], stddev=0.1))
    
    #w_norm = tf.concat([tf.concat([wxg, whg], axis=2), tf.concat([wxo, who], axis=2)], axis=3)
    for i in range(T):
        if i == 0:
            h = convGRU(frames[i], h0, wxg, whg, bg, wxo, who, bo)
        else:
            h = convGRU(frames[i], h, wxg, whg, bg, wxo, who, bo)
    
    h = tf.concat([h, prev_conv2], axis=3)
    w_deconv1 = tf.Variable(tf.truncated_normal([deconv_1, deconv_1, deconv_1_c, conv_gru_c+conv_2_c], stddev=0.1))
    b_deconv1 = tf.Variable(tf.truncated_normal([deconv_1_c], stddev=0.1))
    hidden_deconv1 = tf.nn.conv2d_transpose(h, w_deconv1, output_shape=[b_size//T, xdim_1, ydim_1, deconv_1_c], strides=[1, 2, 2, 1])
    hidden_deconv1 = tf.reshape(tf.nn.bias_add(hidden_deconv1, b_deconv1), hidden_deconv1.get_shape())
    hidden_deconv1 = lrelu(batch_norm(hidden_deconv1,name='deconv1_bn'))
    
    hidden_deconv1 = tf.concat([hidden_deconv1, prev_conv1], axis=3)
    w_deconv2 = tf.Variable(tf.truncated_normal([deconv_2, deconv_2, deconv_2_c, deconv_1_c+conv_1_c], stddev=0.1))
    b_deconv2 = tf.Variable(tf.truncated_normal([deconv_2_c], stddev=0.1))
    hidden_deconv2 = tf.nn.conv2d_transpose(hidden_deconv1, w_deconv2, output_shape=[b_size//T, xdim, ydim, deconv_2_c], strides=[1, 2, 2, 1])
    hidden_deconv2 = tf.reshape(tf.nn.bias_add(hidden_deconv2, b_deconv2), hidden_deconv2.get_shape())
    hidden_deconv2 = lrelu(batch_norm(hidden_deconv2, name='deconv2_bn'))
    
    hidden_deconv2 = tf.concat([hidden_deconv2, prev_conv0], axis=3)
    w_deconv3 = tf.Variable(tf.truncated_normal([deconv_3, deconv_3, deconv_3_c, deconv_2_c+conv_0_c], stddev=0.1))
    b_deconv3 = tf.Variable(tf.truncated_normal([deconv_3_c], stddev=0.1))
    hidden_deconv3 = tf.nn.conv2d_transpose(hidden_deconv2, w_deconv3, output_shape=[b_size//T, xdim, ydim, deconv_3_c], strides=[1, 1, 1, 1])
    hidden_deconv3 = tf.reshape(tf.nn.bias_add(hidden_deconv3, b_deconv3), hidden_deconv3.get_shape())
    hidden_deconv3 = lrelu(batch_norm(hidden_deconv3, name='deconv3_bn'))
    
    w_deconv4 = tf.Variable(tf.truncated_normal([deconv_4, deconv_4, deconv_4_c, deconv_3_c], stddev=0.1))
    b_deconv4 = tf.Variable(tf.truncated_normal([deconv_4_c], stddev=0.1))
    hidden_deconv4 = tf.nn.conv2d_transpose(hidden_deconv3, w_deconv4, output_shape=[b_size//T, xdim, ydim, deconv_4_c], strides=[1, 1, 1, 1])
    out = tf.reshape(tf.nn.bias_add(hidden_deconv4, b_deconv4), hidden_deconv4.get_shape())
    #hidden_deconv4 = lrelu(batch_norm(hidden_deconv4, name='deconv4_bn'))
    
#    w_deconv5 = tf.Variable(tf.truncated_normal([deconv_5, deconv_5, deconv_5_c, deconv_4_c], stddev=0.1))
#    b_deconv5 = tf.Variable(tf.truncated_normal([deconv_5_c], stddev=0.1))
#    hidden_deconv5 = tf.nn.conv2d_transpose(hidden_deconv4, w_deconv5, output_shape=[b_size//T, xdim, ydim, deconv_5_c], strides=[1, 1, 1, 1])
#    out = tf.reshape(tf.nn.bias_add(hidden_deconv5, b_deconv5), hidden_deconv5.get_shape())
    
    #out = tf.sigmoid(out)
    #loss = tf.reduce_mean((out-y)*(out-y))
    #reg = -tf.reduce_mean((tf.sigmoid(out) - z)**2)
    reg = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=z))
    grads_mean = tf.reduce_mean(grads)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=y)*(grads))
    #loss = 10*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=y)) #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=z))
    return x, y, z, tf.train.AdamOptimizer(learning_rate=lr).minimize(loss), loss, tf.sigmoid(out), reg

def load_data(T=8, gap=5, inp_dim=64, inp_c=3, paths=['walking'], b_size=32, repeats=1):
    fnames = []
    for path in paths:
        tmp = glob(os.path.join(path+'/', '*.avi'))
        print(path + ' - ' + str(len(tmp)) + ' files')
        fnames.extend(tmp)
    
    
    print('Reading ', len(fnames)*repeats, ' files...')
    X = np.zeros([len(fnames)*repeats, T, inp_dim, inp_dim, inp_c], dtype=np.float32)
    y = np.zeros([len(fnames)*repeats, inp_dim, inp_dim, inp_c])
    
    #fnames = np.array(fnames)
    #np.random.shuffle(fnames)
    
    for img_idx,fname in enumerate(fnames):
        cap = cv2.VideoCapture(fname)
        
        #skip first few frames
        for i in range(40):
            _ = cap.read()
        
        for rep in range(repeats):
            for i in range(T):
                _, frame = cap.read()
                frame = im2double(scimisc.imresize(frame, [inp_dim, inp_dim]))
                X[img_idx*repeats + rep, i, :, :, :] = frame
                #cv2.imwrite('example/' + str(img_idx*repeats + rep*T + i)+'.jpg',frame*255)
                
            for i in range(gap):
                _, frame = cap.read()
            frame = im2double(scimisc.imresize(frame, [inp_dim, inp_dim]))
            #cv2.imwrite('example/' + str(img_idx*repeats + rep*T) +'O.jpg',frame*255)
            y[img_idx*repeats + rep, :, :, :] = frame
        cap.release()
    
    idx = []
#    thresh = 0.2
#    for i in range(X.shape[0]):
#        grad = np.zeros([X.shape[2], X.shape[3], X.shape[4]])
#        for j in range(1, X.shape[1]):
#            grad += np.abs(X[i,j] - X[i,j-1])
#        if np.mean(grad) > thresh:
#            idx.append(i)
#            for j in range(T):
#                cv2.imwrite('example/' + str(i*T + j)+'.jpg',X[i,j]*255)
#    idx = np.array(idx)
#    print('Left with ', len(idx), ' videos after filtering')
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    X_train = X[b_size:]
    y_train = y[b_size:]
    X_val = X[:b_size]
    y_val = y[:b_size]
    idx = np.random.randint(0,X_train.shape[0], size=[b_size])
    return X_train, y_train, X_train[idx], y_train[idx]

def get_batch(X, y, batch_size, get_all=False):
    N, T, dim_x, dim_y, dim_c = X.shape
    if get_all:
        idx = np.arange(0, N)
    else:
        idx = np.random.randint(0, N, [batch_size])
    
    XX = np.zeros([T*batch_size, dim_x, dim_y, dim_c], dtype=np.float32)
    yy = np.zeros([batch_size, dim_x, dim_y, dim_c], dtype=np.float32)
    
    for i in range(T):
        XX[i*batch_size:(i+1)*batch_size, :,:,:] = X[idx, i, :,:,:]
        
    yy = y[idx, :,:,:]    
    return XX, yy

def verify_batch(X, y):
    XX, yy = get_batch(X, y, 3)
    
    for img_idx in range(3):
        for t in range(8):
            cv2.imshow('new_frame', XX[t*3 + img_idx,:,:,:])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualize(inp, expected, prev, it):
    prev = np.reshape(prev,  [-1, prev.shape[2], prev.shape[3]])
    inp = np.reshape(inp, [-1, inp.shape[2], inp.shape[3]])
    expected = np.reshape(expected, [-1, expected.shape[2], expected.shape[3]])
    vis = np.concatenate((prev, inp, expected), axis=1)
    cv2.imwrite('results/vis'+str(it)+'.jpg', vis*255)

def main():
    batch_size = 8
    xdim=108
    iterations = 30001
    T = 8
    checkpoints_folder = 'checkpoints/trial8/'
    train_x, train_y, val_x, val_y = load_data(inp_dim=xdim,paths=['walking', 
                                                      'running', 
                                                      'jogging',
                                                      'boxing',
                                                      'handwaving',
                                                      'handclapping'], b_size=batch_size, repeats=8)
    print(train_x.shape)
    #verify_batch(data_X, data_y)
    
    
    feed_x, feed_y, feed_z, optim, loss, output, reg = build_model(xdim=xdim, b_size=batch_size*T, lr=0.04, T=T)
    x_val, y_val = get_batch(val_x, val_y, batch_size, get_all=True)
    frames = np.split(x_val, T, axis=0)
    grads = np.zeros([8,108,108,3])
    for i in range(1,batch_size):
        grads += np.abs(frames[i]-frames[i-1])
    for i in range(T):
        grads[i] = (grads[i]-np.min(grads[i]))/(np.max(grads[i])-np.min(grads[i]))
        print(np.mean(grads[i]))
    #grads *= grads
    visualize(grads, frames[T-1], frames[0], 'grads')
    #train_x = train_x[1:,:,:,:,:]
    #train_y = train_y[1:,:,:,:]
    #print(train_x.shape)
    
    
    #cv2.imwrite('results/real'+'.jpg', scimisc.imresize(y_val[0, :,:,:], [120, 120]))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, checkpoints_folder)
        tot_epochs = iterations*batch_size//train_x.shape[0]
        iter_per_epoch = train_x.shape[0]//batch_size
        prev_norm = 0.0
        tim = time.time()
        
        for it in range(iterations):
            x_batch, y_batch = get_batch(train_x, train_y, batch_size)
            
            optim.run(feed_dict={
                    feed_x: x_batch,
                    feed_y: y_batch,
                    feed_z: x_batch[batch_size*(T-1):]
                    })
            if it % 50 == 0:
                #idx = np.random.randint(0, x_train.shape[0], size=[b_size])
                scores = output.eval(feed_dict={
                        feed_x: x_val,
                        feed_y: y_val,
                        feed_z: x_val[batch_size*(T-1):]
                        })
                visualize(scores, y_val,x_val[batch_size*(T-1):], it)
                val_loss = loss.eval(feed_dict={
                    feed_x: x_val,
                    feed_y: y_val,
                    feed_z: x_val[batch_size*(T-1):]
                    })
                prev_val_loss = loss.eval(feed_dict={
                        feed_x: x_val,
                        feed_y: x_val[batch_size*(T-1):],
                        feed_z: x_val[batch_size*(T-1):]
                        })
                print('Epochs - ',it//iter_per_epoch,'/', tot_epochs,' time - ', time.time()-tim, ' Iterations - ', it, 
                  ' Validation loss = ', val_loss, ' Prev-val_loss - ', prev_val_loss) 
                  #' reg_loss- ', reg_loss)
#            if it % 100 == 0:
#                scores = output.eval(feed_dict={
#                        feed_x: x_batch,
#                        feed_y: y_batch
#                        })
#                visualize(scores, y_batch, it)
            if it % 1000 == 0:
                saver.save(sess, checkpoints_folder)
            #time.sleep(0.5)
                
        saver.save(sess, checkpoints_folder)
        
if __name__ == '__main__':
    main()