# 분류 ANN을 위한 인공지능 모델 구현
from keras import layers, models

# 분산 방식 모델링을 포함하는 함수형 구현
def ANN_models_func(Nin, Nh, Nout):
    x = layers.Input(shape=(Nin,))
    h = layers.Activation('relu')(layers.Dense(Nh)(x))
    y = layers.Activation('softmax')(layers.Dense(Nout)(h))
    model = models.Model(x, y)
    model.compile(loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])
    return model


# 연쇄 방식 모델링을 포함하는 함수형 구현
def ANN_seq_func(Nin, Nh, Nout):
    model = models.Sequential()
    model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
    model.add(layers.Dense(Nout, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])
    return model


# 분산 방식 모델링을 포함하는 객체지향형 구현
class ANN_models_class(models.Model):
    def __init__(self, Nin, Nh, Nout):
        # Prepare network layers and activate functions
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        # Connect network elements
        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])


# 연쇄 방식 모델링을 포함하는 객체지향형 구현
class ANN_seq_class(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])


# 분류 ANN에 사용할 데이터 불러오기
import numpy as np
from keras import datasets # mnist
from keras.utils import np_utils # to_categorical
import random

def Data_func():
    # (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    source_list = range(1,1000) # 999개
    source_y = [0 if x % 2 == 0 else 1 for x in source_list] # 짝이면 0, 홀이면 1

    # source_list_np = np.asarray(source_list)
    # source_y_np = np.asarray(source_y)

    random_indexes = [random.choice(range(999)) for _ in range(120)]

    X_train = np.asarray(source_list)
    y_train = np.asarray(source_y)

    X_test = [source_list[x] for x in random_indexes]
    y_test = [source_y[x] for x in random_indexes]
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    
    X_train = np_utils.to_categorical(X_train, num_classes=1000)
    X_test = np_utils.to_categorical(X_test, num_classes=1000)
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    # L, W, H = X_train.shape # 60000, 28, 28
    # X_train = X_train.reshape(-1, W * H)
    # X_test = X_test.reshape(-1, W * H)

    # X_train = X_train / 255.0
    # X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)


# 분류 ANN 학습 결과 그래프 구현
import matplotlib.pyplot as plt

def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

def plot_acc(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0) 


# 분류 ANN 학습 및 성능 분석
def main():
    Nin = 1000
    Nh = 100
    number_of_class = 2
    Nout = number_of_class

    model = ANN_seq_class(Nin, Nh, Nout)
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    
    print(X_train.shape)
    print(Y_train.shape)

    ##########################
    # Training
    ##########################
    history = model.fit(X_train, Y_train, epochs=100,
                        batch_size=100, validation_split=0.2)
 #   performance_test = model.evaluate(X_test, Y_test, batch_size=100)
 #   print('Test Loss and Accuracy ->', performance_test)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    X_hat = np.array([1])
    X_hat = np_utils.to_categorical(X_hat, num_classes=1000)
    Y_hat = model.predict(X_hat)
    print(Y_hat)

    X_hat = np.array([1, 10, 20, 654])
    X_hat = np_utils.to_categorical(X_hat, num_classes=1000)
    print(X_hat.shape)
    Y_hat = model.predict(X_hat)
    print(Y_hat)

#    plot_loss(history)
#    plt.show()
#    plot_acc(history)
#    plt.show()

# Run code
if __name__ == '__main__':
    main()