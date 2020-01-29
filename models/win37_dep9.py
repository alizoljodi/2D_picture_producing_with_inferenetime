import tensorflow as tf

slim = tf.contrib.slim


def create_network(state, inputs, is_training, scope="win37_dep9", reuse=False):
    num_maps = 64
    kw = 5
    kh = 5
    #print('state:',state)

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):

            has=False
            #print(state[1])
            for i in state[1]:
                if i[0]!='none':
                    has=True
            if has==False:
                #print('effefegrghrg',state)
                net=inputs
                count=0
                for i in state[0]:
                    if i[0] == 'conv2d':
                        #print(i)
                        net = slim.conv2d(net, i[1], [i[3], i[3]], padding=i[2],scope='1_conv2d_'+str(count)+str(i[1]))
                        count+=1
                        # print(net.shape)
                    elif i[0] == 'batch':
                        net = slim.batch_norm(net, is_training=is_training,scope='1_batch_'+str(count))
                        count+=1
                        # print(net.shape)
                    else:
                        pass
                        # print(f)
                        #raise ValueError('not in category1')



                net = slim.batch_norm(net, is_training=is_training,scope='1_batch_'+str(count))
                return net


            else:
                nets=[inputs,inputs]
                count1=0
                count2=0
                f1=f2=['nome',64,'none',37]
                for i in range(max(len(state[0])+1,len(state[1])+1)):
                    if len(state[0])!=0:
                        f3=f1
                        f1=state[0].pop(0)
                        if f1[0] == 'conv2d':
                            nets[0] = slim.conv2d(nets[0], f1[1], [f1[3], f1[3]], padding=f1[2],scope='1_conv2d_'+str(count1)+str(f1[1]))
                            count1+=1
                            # print(net.shape)
                        elif f1[0] == 'batch':
                            nets[0] = slim.batch_norm(nets[0], is_training=is_training,scope='1_batch_'+str(count1))
                            count1+=1
                            # print(net.shape)
                        elif f1[0]=='conc':
                            shape=nets[0].shape
                            #print(shape)
                            #print(shape[1],type(shape[1]))
                            x=int(shape[1])
                            y=int(shape[2])
                            #print(x,type(x))
                            #print(nets[1].shape)
                            #input()
                            new_image=tf.image.resize_image_with_crop_or_pad(nets[1],x,y)
                            #print(new_image)
                            #print(f3)
                            print(nets[0])
                            print(f3)
                            new_image=slim.conv2d(new_image,nets[0].shape[3],[5,5],padding='same',scope='1_conc_conv_'+str(count1)+str(nets[0].shape[3]))
                            count1+=1
                            new_image=slim.batch_norm(new_image,is_training=is_training,scope='1_conc_batch_'+str(count1))
                            count1+=1
                            #print(nets[0].shape)
                            #print(new_image.shape)
                            nets[0]=nets[0]+new_image
                        else:
                            pass
                    if len((state[1]))!=0:
                        f4=f2
                        f2 = state[1].pop(0)
                        if f2[0] == 'conv2d':
                            #print('kkkkk',f2)
                            nets[1] = slim.conv2d(nets[1], f2[1], [f2[3], f2[3]], padding=f2[2],scope='2_conv_'+str(count2)+str(f2[1]))
                            count2+=1
                            # print(net.shape)
                        elif f2[0] == 'batch':
                            nets[1] = slim.batch_norm(nets[1], is_training=is_training,scope='2_batch_'+str(count2))
                            count2+=1
                            # print(net.shape)
                        elif f2[0] == 'conc':
                            shape = nets[1].shape
                            #print(shape)
                            x=int(shape[1])
                            y=int(shape[2])
                            #input()
                            new_image = tf.image.resize_image_with_crop_or_pad(nets[0],x,y)
                            new_image = slim.conv2d(new_image, nets[1].shape[3], [5, 5], padding='same',scope='2_conc_conv_'+str(count2)+str(nets[1].shape[3]))
                            count2+=1
                            new_image = slim.batch_norm(new_image, is_training=is_training,scope='2_conc_batch_'+str(count2))
                            count2+=1
                            nets[1] = nets[1] + new_image
                        else:
                            pass
                            #print(nets[1].shape)


                #print('0',nets[0].shape)
                #print('1',nets[1].shape)
                #print(f1)
                filt=int(nets[0].shape[3])
                nets[1]=slim.conv2d(nets[1],filt,[1,1],padding='same',scope='3_conv2d'+str(filt))
                nets[1]=slim.batch_norm(nets[1],is_training=is_training,scope='3_batch')
                net=nets[0]+nets[1]
                return net