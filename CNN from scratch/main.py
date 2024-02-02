from convolution import Convolution
from pooling import Pooling
from fullyConnected import ANN
from optimizer import Optimizer
import numpy as np
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0
conv = Convolution()
pool = Pooling()
ann = ANN(216, 10)
opt = Optimizer()


feature_map = {}
pooling = {}
index_pool = {}


def initialisation(x):
    m = {}
    v = {}
    m, v = opt.initialisation(x, "adam")
    return m, v


def forward(img):
    global feature_map
    global pooling
    global index_pool
    pooling["layer0"] = img
    for i in range(1, len(conv.layer)+1):
        feature_map["layer"+str(i)] = conv.forward(pooling["layer" +
                                                           str(i-1)], conv.filters["layer"+str(i)], i)
        pooling["layer"+str(i)], index_pool["layer"+str(i)
                                            ] = pool.max_pooling(feature_map["layer"+str(i)], 2)
    output = ann.forward(pooling["layer"+str(i)])
    return output


def bacward(out, label):
    global feature_map
    global pooling
    global index_pool
    grad_feature_map = {}
    grad_filter = {}
    grad_biases = {}
    # bacword for a fully connected layer
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    d_out = ann.backprop(gradient, 0.01)
    grad_feature_map["layer2"] = pool.backward(
        feature_map["layer2"], d_out, index_pool["layer2"])
    grad_filter["layer2"] = conv.backward(
        pooling["layer1"], grad_feature_map["layer2"], 2, feature_map)
    grad_biases["layer2"] = np.array([np.sum(
        grad_feature_map["layer2"][k]) for k in range(grad_feature_map["layer2"].shape[2])])
    for i in range(len(conv.layer)-2, -1, -1):
        dp1 = pool.grad_p(pooling, grad_feature_map, i+1, conv)
        grad_feature_map["layer"+str(i+1)] = pool.backward(
            feature_map["layer"+str(i+1)], dp1, index_pool["layer"+str(i+1)])
        grad_filter["layer"+str(i+1)] = conv.backward(pooling["layer"+str(i)],
                                                      grad_feature_map["layer"+str(i+1)], i+1, feature_map)
        grad_biases["layer"+str(i+1)] = np.array([np.sum(grad_feature_map["layer"+str(i+1)][k])
                                                  for k in range(grad_feature_map["layer"+str(i+1)].shape[2])])
    return grad_filter,grad_biases


def fit(img, lab,  epoch, m, v, m_filter, v_filter,m_biases,v_biases):
    global conv
    global pool
    global ann
    for _ in range(epoch):
        accuracy = 0
        for im, label in zip(img, lab):
            out = forward(im)
            acc = 1 if np.argmax(out) == label else 0
            accuracy += acc
            grad_filter,grad_biases = bacward(out, label)
            ann.param = opt.adam(ann.param, ann.dparam, m, v)
            conv.filters = opt.adam(
                conv.filters, grad_filter, m_filter, v_filter)
            conv.biases = opt.adam(
                conv.biases, grad_biases, m_biases, v_biases)
            # for i in range(len(conv.layer)):
            #     conv.filters["layer"+str(i+1)]=conv.filters["layer"+str(i+1)]-lr*grad_filter["layer"+str(i+1)]
            # conv.biases["layer"+str(i+1)]=conv.biases["layer"+str(i+1)]-lr*grad_biases["layer"+str(i+1)]

        print("accuracy: ", accuracy/img.shape[0])
    np.save("filter22.npy", conv.filters["layer2"])
    np.save("filter11.npy", conv.filters["layer1"])
    np.save("baises11_conv.npy", conv.biases["layer1"])
    np.save("baises22_conv.npy", conv.biases["layer2"])
    np.save("weight1.npy", ann.param["weight"])
    np.save("biases2.npy", ann.param["biases"])


def test(img, label):
    accuracy = 0
    for i in range(img.shape[0]):
        out = forward(img[i])
        #print("predicted: ",np.argmax(out),"---true: ",label[i])
        acc = 1 if np.argmax(out) == label[i] else 0
        accuracy += acc
    print("accuracy test: ", accuracy/img.shape[0]*100)


m, v = initialisation(ann.param)
m_filter, v_filter = initialisation(conv.filters)
m_biases,v_biases=initialisation(conv.biases)
permutation = np.random.permutation(len(train_images))
train_images = train_images[permutation]
train_labels = train_labels[permutation]
fit(train_images[:20000], train_labels[:20000],
     10, m, v, m_filter, v_filter,m_biases,v_biases)
test(test_images[:10000], test_labels[:10000])
