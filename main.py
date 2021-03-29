import numpy as np


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def forward(wij, wjk, bias, input, output):
    # hidden layer
    I3 = input[0] * wij[0][0] + input[1] * wij[1][0] + bias[0]
    O3 = sigmod(I3)
    print("First hidden layer neuron: ", I3, O3)

    I4 = input[0] * wij[0][1] + input[1] * wij[1][1] + bias[1]
    O4 = sigmod(I4)
    print("Second hidden layer neuron: ", I4, O4)

    I5 = input[0] * wij[0][2] + input[1] * wij[1][2] + bias[2]
    O5 = sigmod(I5)
    print("Third hidden layer neuron: ", I5, O5)

    # output layer
    I6 = O3 * wjk[0][0] + O4 * wjk[0][1] + O5 * wjk[0][2] + bias[3]
    O6 = sigmod(I6)
    print("First output layer neuron: ", I6, O6)

    I7 = O3 * wjk[1][0] + O4 * wjk[1][1] + O5 * wjk[1][2] + bias[4]
    O7 = sigmod(I7)
    print("Second output layer neuron: ", I7, O7)

    # calculate error
    err6 = O6 * (1 - O6) * (output[0] - O6)
    err7 = O7 * (1 - O7) * (output[1] - O7)
    err3 = O3 * (1 - O3) * (wjk[0][0] * err6 + wjk[1][0] * err7)
    err4 = O4 * (1 - O4) * (wjk[0][1] * err6 + wjk[1][1] * err7)
    err5 = O5 * (1 - O5) * (wjk[0][2] * err6 + wjk[1][2] * err7)

    outList = np.array([O3, O4, O5, O6, O7])
    print("Output list: ", outList)
    errList = np.array([err3, err4, err5, err6, err7])
    print("Error list: ", errList)

    return outList, errList


def backpropagation(out, err, input):
    # new weights for wjk
    new_w36 = wjk[0][0] + l * out[0] * err[3]
    new_w37 = wjk[1][0] + l * out[0] * err[4]
    new_w46 = wjk[0][1] + l * out[1] * err[3]
    new_w47 = wjk[1][1] + l * out[1] * err[4]
    new_w56 = wjk[0][2] + l * out[2] * err[3]
    new_w57 = wjk[1][0] + l * out[2] * err[4]

    # new weights for wij
    new_w13 = wij[0][0] + l * input[0] * err[0]
    new_w23 = wij[1][0] + l * input[1] * err[0]
    new_w14 = wij[0][1] + l * input[0] * err[1]
    new_w24 = wij[1][1] + l * input[1] * err[1]
    new_w15 = wij[0][2] + l * input[0] * err[2]
    new_w25 = wij[1][2] + l * input[1] * err[2]

    newWeights = np.array([new_w36, new_w37, new_w46, new_w47, new_w56, new_w57,
                           new_w13, new_w23, new_w14, new_w24, new_w15, new_w25])
    print("new weights: ", newWeights)

    # calculate new bias
    b3 = bias[0] + l * err[0]
    b4 = bias[1] + l * err[1]
    b5 = bias[2] + l * err[2]
    b6 = bias[3] + l * err[3]
    b7 = bias[4] + l * err[4]

    newBias = np.array([b3, b4, b5, b6, b7])
    print("new bias", newBias)

    return newWeights, newBias


if __name__ == "__main__":
    wij = np.array([[0.1, 0, 0.3], [-0.2, 0.2, -0.4]])
    wjk = np.array([[-0.4, 0.1, 0.6], [0.2, -0.1, -0.2]])
    bias = np.array([0.1, 0.2, 0.5, -0.1, 0.6])
    l = 0.1

    input1 = np.array([0.6, 0.1])
    output1 = np.array([1, 0])
    print("Train T1: ")
    outL, errL = forward(wij, wjk, bias, input1, output1)
    newWeight, newBia = backpropagation(outL, errL, input1)

    input2 = np.array([0.2, 0.3])
    output2 = np.array([0, 1])
    print("Train T2: ")
    outL, errL = forward(wij, wjk, bias, input2, output2)
    newWeight, newBia = backpropagation(outL, errL, input2)

