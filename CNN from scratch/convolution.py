import numpy as np


class Convolution:
    def __init__(self, layer=[(3, 3), (6, 2)]):
        self.filters = {}
        self.biases = {}
        self.num_filter = {
            "layer"+str(i+1): layer[i][0] for i in range(len(layer))}
        self.size_filter = {
            "layer"+str(i+1): layer[i][1] for i in range(len(layer))}
        self.layer = {"layer"+str(i+1): layer[i] for i in range(len(layer))}
        self.initialisation()

    def initialisation(self):
        for i, v in enumerate(self.layer):
            if i == 0:
                 self.filters["layer"+str(i+1)] = np.random.rand(self.num_filter["layer"+str(i+1)], self.size_filter["layer"+str(
                     i+1)], self.size_filter["layer"+str(i+1)])/self.size_filter["layer"+str(i+1)]*self.size_filter["layer"+str(i+1)]
            else:
                 self.filters["layer"+str(i+1)] = np.random.rand(self.num_filter["layer"+str(i+1)], self.num_filter["layer"+str(i)], self.size_filter["layer"+str(
                     i+1)], self.size_filter["layer"+str(i+1)])/self.size_filter["layer"+str(i+1)]*self.size_filter["layer"+str(i+1)]
             self.biases["layer" +
                         str(i+1)] = np.zeros((self.num_filter["layer"+str(i+1)],))

    def iterate_regions(self, image, size_filter):
        h, w = image.shape

        for i in range(h - size_filter+1):
            for j in range(w - size_filter+1):
                im_region = image[i:(i + size_filter), j:(j + size_filter)]
                yield im_region, i, j

    def conv2D(self, input, filters, num_layer):
        self.last_input = {}
        if len(input.shape) == 2:
            self.last_input["layer"+str(num_layer)] = input
            num_filters, size_filter, size_filter = filters.shape
            h, w = input.shape
            output = np.zeros(
                (h - size_filter+1, w - size_filter+1, num_filters))

            for im_region, i, j in self.iterate_regions(input, size_filter):
                output[i, j] = np.sum(im_region * filters, axis=(1, 2))

            return output
        elif len(input.shape) == 3:
            self.last_input["layer"+str(num_layer)] = input
            h, w, chanel = input.shape
            num_filters, num_channel, size_filter, size_filter = filters.shape
            output = np.zeros(
                (h - size_filter+1, w - size_filter+1, num_filters))
            for k in range(num_filters):
                out = np.zeros((h-size_filter+1, w-size_filter+1))
                for v in range(chanel):
                    for im_region, i, j in self.iterate_regions(input[:, :, v], size_filter):
                        out[i, j] += np.sum(im_region *
                                            filters[k][v], axis=(0, 1))
                output[:, :, k] += out

            return output

    def relu(self, Z):
        A = np.maximum(0, Z)
        return A

    def forward(self, input, filters, num_layer):
        out = self.conv2D(input, filters, num_layer)
        for i in range(self.num_filter["layer"+str(num_layer)]):
            out[:, :, i] += self.biases["layer"+str(num_layer)][i]
        return self.relu(out)

    def tensor_dot_product(self, tensor1, tensor2):
        if tensor1.shape != tensor2.shape:
            raise ValueError("The two tensors must have the same shape.")
        result = np.zeros(tensor1.shape)
        for k in range(tensor1.shape[2]):
            result[:, :, k] = np.multiply(tensor1[:, :, k], tensor2[:, :, k])
        return result

    def backward(self, p, dz, num_layer, feature_map):
        grad_filter = {}
        dz = self.tensor_dot_product(dz, np.where(
            feature_map["layer"+str(num_layer)] > 0, 1, 0))
        if num_layer != 1:
            grad_filter["layer"+str(num_layer)] = np.zeros(
                self.filters["layer"+str(num_layer)].shape)
            for k in range(dz.shape[2]):
                for v in range(p.shape[2]):
                    for im_region, i, j in self.iterate_regions(self.last_input["layer"+str(num_layer)][:, :, v], self.size_filter["layer"+str(num_layer)]):
                        for f in range(self.num_filter["layer"+str(num_layer)]):
                            grad_filter["layer"+str(num_layer)
                                        ][k, v] += dz[i, j, f] * im_region

            return grad_filter["layer"+str(num_layer)]
        elif num_layer == 1:
            grad_filter["layer1"] = np.zeros(self.filters["layer1"].shape)
            for v in range(dz.shape[2]):
                for im_region, i, j in self.iterate_regions(p, self.size_filter["layer1"]):
                    for f in range(self.num_filter["layer1"]):
                        grad_filter["layer1"][v] += dz[i, j, f] * im_region
            return grad_filter["layer1"]
