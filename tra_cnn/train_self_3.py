from __future__ import absolute_import, division, print_function
import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, save_model_dir, model_index, save_every_n_epoch,\
    train_dir,valid_dir,test_dir,NUM_CLASSES,save_img_dir
from prepare_data import generate_datasets, load_and_preprocess_image
import math
from models import mobilenet_v1, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, \
    efficientnet, resnext, inception_v4, inception_resnet_v1, inception_resnet_v2, \
    se_resnet, squeezenet, densenet, shufflenet_v2, resnet,vgg16,vgg16_mini,VGG16_self, \
    diy_residual_block,diy_resnet
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Input, regularizers
import matplotlib.pyplot as plt

def get_model():
    if model_index == 0:
        return mobilenet_v1.MobileNetV1()
    elif model_index == 1:
        return mobilenet_v2.MobileNetV2()
    elif model_index == 2:
        return mobilenet_v3_large.MobileNetV3Large()
    elif model_index == 3:
        return mobilenet_v3_small.MobileNetV3Small()
    elif model_index == 4:
        return efficientnet.efficient_net_b0()
    elif model_index == 5:
        return efficientnet.efficient_net_b1()
    elif model_index == 6:
        return efficientnet.efficient_net_b2()
    elif model_index == 7:
        return efficientnet.efficient_net_b3()
    elif model_index == 8:
        return efficientnet.efficient_net_b4()
    elif model_index == 9:
        return efficientnet.efficient_net_b5()
    elif model_index == 10:
        return efficientnet.efficient_net_b6()
    elif model_index == 11:
        return efficientnet.efficient_net_b7()
    elif model_index == 12:
        return resnext.ResNeXt50()
    elif model_index == 13:
        return resnext.ResNeXt101()
    elif model_index == 14:
        return inception_v4.InceptionV4()
    elif model_index == 15:
        return inception_resnet_v1.InceptionResNetV1()
    elif model_index == 16:
        return inception_resnet_v2.InceptionResNetV2()
    elif model_index == 17:
        return se_resnet.se_resnet_50()
    elif model_index == 18:
        return se_resnet.se_resnet_101()
    elif model_index == 19:
        return se_resnet.se_resnet_152()
    elif model_index == 20:
        return squeezenet.SqueezeNet()
    elif model_index == 21:
        return densenet.densenet_121()
    elif model_index == 22:
        return densenet.densenet_169()
    elif model_index == 23:
        return densenet.densenet_201()
    elif model_index == 24:
        return densenet.densenet_264()
    elif model_index == 25:
        return shufflenet_v2.shufflenet_0_5x()
    elif model_index == 26:
        return shufflenet_v2.shufflenet_1_0x()
    elif model_index == 27:
        return shufflenet_v2.shufflenet_1_5x()
    elif model_index == 28:
        return shufflenet_v2.shufflenet_2_0x()
    elif model_index == 29:
        return resnet.resnet_18()
    elif model_index == 30:
        return resnet.resnet_34()
    elif model_index == 31:
        return resnet.resnet_50()
    elif model_index == 32:
        return resnet.resnet_101()
    elif model_index == 33:
        return resnet.resnet_152()
    elif model_index == 34:
        return vgg16.VGG16()
    elif model_index == 35:
        return vgg16_mini.VGG16()
    elif model_index == 36:
        return VGG16_self.VGG16()
    elif model_index == 10086:
        return diy_resnet.resnet_50()
    else:
        raise ValueError("The model_index does not exist.")


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()

    return images, labels


def get_datasets():
    # Preprocess the dataset
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='constant'
    )

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        class_mode="categorical")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
    )
    valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 /255.0
    )
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                      color_mode="rgb",
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      class_mode="categorical"
                                                      )


    train_num = train_generator.samples
    valid_num = valid_generator.samples
    test_num = test_generator.samples


    return train_generator, \
           valid_generator, \
           test_generator, \
           train_num, valid_num, test_num

def getKerasModel():
    tf.keras.backend.set_learning_phase(0)
    base_model = ResNet50(
                          include_top=False,
                          input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    base_model.trainable = True
    tf.keras.backend.set_learning_phase(1)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='average_pool')(x)
    x = Flatten()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.05),activation = tf.keras.activations.relu)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model





if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the dataset
    train_generator, valid_generator, test_generator, \
    train_num, valid_num, test_num = get_datasets()

    # create model
    model = getKerasModel()
    model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    set_trainable = False
    # 加载模型权重，继续训练
    model.load_weights(filepath=save_model_dir + "model")
    model.summary()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  #=tf.keras.optimizers.SGD(lr=5e-3,decay=0.02,momentum=0.9, nesterov=True),
                  optimizer=tf.keras.optimizers.Adadelta(lr=8e-3),
                  metrics=['acc'])
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    modelcheckpoint=tf.keras.callbacks.ModelCheckpoint(filepath=save_model_dir+"model",
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max',
                                    period=1,)
    # reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.5,
    #     patience=4,
    #     verbose=1,
    #     min_lr=5e-6,
    # )
    #early_stop = tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_acc')
    callback_list = [tensorboard, modelcheckpoint]



    # start training
    history = model.fit_generator(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=valid_num // BATCH_SIZE,
                        callbacks=callback_list,
                        validation_freq=1)

    # save the whole model
    #model.save_weights(filepath=save_model_dir + "model", save_format='tf')



    fig = plt.figure()#新建一张图
    plt.plot(history.history['acc'],label='training acc')
    plt.plot(history.history['val_acc'],label='val acc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    fig.savefig('ModelTraining_acc.png')
    fig = plt.figure()
    plt.plot(history.history['loss'],label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig('modelTraining_loss.png')