import matplotlib.pyplot as plt

data1 = [1,2, 3,4]
data2 = [4 ,3 ,2 ,1]

plt.plot(data1, 'r.' , label='circle')
plt.plot(data2, 'g^' , label='triangle')

#그래프
plt.title("title of plot")
plt.legend()

plt.show()