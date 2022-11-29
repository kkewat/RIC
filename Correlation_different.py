import numpy as np
import matplotlib.pyplot as plt
import matplotlib


np.random.seed(1)
#1000 Random integers between 0 to 50
x = np.random.randint(1,50,1000)

#Positive Correlation
print('Positive Correlation')
#Positive Correlation with some noise
y = x+np.random.normal(0,10,1000)

np.corrcoef(x,y)

matplotlib.style.use('ggplot')

plt.scatter(x,y)
plt.show()

#Negative Correlation
print('Negative Correlation')
#Negative correlation with some noise 
y1 = 100 - x + np.random.normal(0,5,1000)
np.corrcoef(x,y)
plt.scatter(x,y)
plt.show()
