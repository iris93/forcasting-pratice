# Fetch data
from preparing_input import InitPreprocessing
import matplotlib.pyplot as plt

init_instance = InitPreprocessing()
X_train, y_train, X_test, y_test = init_instance.get_tensor_transform()

print X_train
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.grid(True)
# p1 = ax.plot(y_test)
# # p2 = ax.plot(y_predict)
# # ax.legend((p1[0], p2[0]), ('real', 'pred'), loc='best', fancybox=True, framealpha=0.5)
# plt.show(block=True)