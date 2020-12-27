import tensorflow as tf
from configuration import save_model_dir
from prepare_data import generate_datasets
from train_self_3 import get_model, process_features,getKerasModel
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
count = 0
if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    # load the model
    model = getKerasModel()
    #model = transfer_model()
    model.load_weights(filepath=save_model_dir+"model")
    #model = tf.saved_model.load(save_model_dir+"saved_model")
    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        return np.argmax(predictions,axis=1)

    label_list=[]
    prediction_list=[]
    for features in test_dataset:
        test_images, test_labels = process_features(features)
        predictions = test_step(test_images,test_labels)
        label_list.extend(test_labels)
        prediction_list.extend(predictions)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))
    t = classification_report(label_list, prediction_list,digits=5)
    print("label     :{}".format(label_list))
    print("prediction:{}".format(prediction_list))
    d = [y for y in prediction_list if y not in label_list]
    print(d)
    print(t)
    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))