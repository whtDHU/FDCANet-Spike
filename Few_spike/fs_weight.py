import numpy as np

sigmoid_h = [1.6558512, 1.0989026, 10.7739, 1.3579437, 1.0146959,
             0.8972655, 2.1207616, 1.6949341, 3.6397197, -2.1483946, 5.1177883, 0]
sigmoid_d = [0.37689912, 0.21977554, 0.40271342, 0.17005783, 0.09584032, 0.06293014,
             0.04360681, -0.0443096, 0.01999909, 0.07107325, 0.01502249, 0.01021793]
sigmoid_T = [-0.00824843, -0.9319625, 1.0080122, -1.2074932, -1.7885877, -2.7065408,
             -3.2715735, 0.07132628, -3.719372, -0.543918, -3.7693157, -5.5447803]

relu_K = 10
alpha = 4
relu_h = alpha * 2 ** (-relu_K) * np.array([float(2 ** (relu_K - i)) for i in range(1, relu_K + 1)]).astype(np.float32)
relu_d = alpha * 2 ** (-relu_K) * np.array([float(2 ** (relu_K - i)) for i in range(1, relu_K + 1)]).astype(np.float32)
relu_T = alpha * 2 ** (-relu_K) * np.array([float(2 ** (relu_K - i)) for i in range(1, relu_K + 1)]).astype(np.float32)
# print(relu_h)
