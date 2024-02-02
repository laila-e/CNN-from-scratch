import numpy as np

class Pooling:
    def __init__(self,type=["max","max"]):
        self.type={"layer"+str(i+1):type[i] for i in range(len(type))}
    def max_pooling(self, input_tensor, pool_size):
        height = input_tensor.shape[0]//pool_size
        width = input_tensor.shape[1]//pool_size
        output = np.zeros(( height, width,input_tensor.shape[2]))
        max_indices = []
        for k in range(input_tensor.shape[2]):
            tmp = []
            for i in range(height):
                for j in range(width):
                    window = input_tensor[:,:,k][i*pool_size:(i+1)*pool_size,
                                             j*pool_size:(j+1)*pool_size]
                    max_val = np.max(window)
                    max_idx = np.argmax(window)
                    tmp.append(
                        (i*pool_size + max_idx//pool_size, j*pool_size + max_idx % pool_size))
                    output[:,:,k][i, j] = max_val
            max_indices.append(tmp)
        return output, max_indices
    def backward(self, shape_origine, dp,index):
        dc = np.zeros(
            (shape_origine.shape[0], shape_origine.shape[1], shape_origine.shape[2]))
        h,w,num_channel=shape_origine.shape
        for k in range(num_channel):
                n = 0
                for i in dp[:,:,k]:
                    for j in i:
                        k1, k2 = index[k][n]
                        dc[k1, k2,k] = j
                        n += 1
        return dc
    def grad_p(self,p,grad_feature_map,num_layer,conv):
        pad_z=np.pad(grad_feature_map["layer"+str(num_layer+1)], (((1,1), (1, 1), (0, 0))))
        dp=np.zeros(p["layer"+str(num_layer)].shape)
        for i in range(conv.num_filter["layer"+str(num_layer+1)]):
            dp+=conv.conv2D(pad_z[:,:,i],np.rot90(np.rot90(conv.filters["layer"+str(num_layer+1)][i])),num_layer)
        return dp