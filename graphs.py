import matplotlib.pyplot as plt
import numpy as np
file=open("results.txt")
values=[]
lines=file.readlines()
for line in lines:
    values.append(line)
plt.plot(np.arange(1e-3,1e-5,1e-7), values)
plt.title('Accuracies for learning rates')
plt.xlabel('learning rate')
plt.ylabel('Accuracy')
plt.show()
