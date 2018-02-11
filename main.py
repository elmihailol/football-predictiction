import numpy
import numpy as np
import pandas as pd
import operator
from sklearn.utils import shuffle
from keras import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


def find_max(x):
    max_index, max_value = max(enumerate(x), key=operator.itemgetter(1))
    # if max_value < 0.6:
    #     return -1
    return max_index
    # return max(x)


data = pd.read_csv('results.csv', sep=',', header=None, names=["date", "team1", "team2",
                                                               "result1", "result2", "type",
                                                               "city", "country"])
data = data[30576:]
# Если нужно смешать
# data = shuffle(data)
print("Подготовка данных")
team_data = []
team_data.extend(data['team1'].values.tolist())
team_data.extend(data['team2'].values.tolist())
team_data.extend(data['country'].values.tolist())
print("Анализ комманд")
encoder = LabelEncoder()
# Классификацируем
encoder.fit(team_data)
encoded_Y = encoder.transform(team_data)
dummy_y = np_utils.to_categorical(encoded_Y)
county_dict = {}
for i in range(len(team_data)):
    county_dict[team_data[i]] = dummy_y[i]
print("Запись победителя")

dataX = []
dataY = []
sum = 0
draw = 0
for index, row in data.iterrows():
    sum += 1
    team1 = row["team1"]
    team2 = row["team2"]
    result1 = row["result1"]
    result2 = row["result2"]
    county = row["country"]
    type = row["type"]

    team1 = county_dict[team1]
    team2 = county_dict[team2]
    county = county_dict[county]
    c_result0 = [0, 1, 0]
    if result1 > result2:
        c_result0 = [1, 0, 0]
    elif result1 < result2:
        c_result0 = [0, 0, 1]
    elif result1 == result2:
        draw += 1
        # continue
    bufX = []

    bin_type = [0, 1]
    # print(type)
    if str(type) == "Friendly":
        bin_type = [1, 0]

    bufX.extend(team1)
    bufX.extend(team2)
    bufX.extend(county)
    bufX.extend(bin_type)
    bufY = []
    bufY.extend(c_result0)
    dataX.append(np.array(bufX))
    dataY.append(np.array(bufY))

print(draw / sum)
dataX = np.array(dataX)
dataY = np.array(dataY)

persent = 0.97
len_train = int(len(dataX) * persent)
len_test = int(len(dataX) - len_train)
trainX = dataX[:len_train]
trainY = dataY[:len_train]
testX = dataX[-len_test:]
testY = dataY[-len_test:]

# exit()
model = Sequential()
model.add(Dense(len(dataX[0]) * 2, input_dim=len(dataX[0]), activation='relu'))
model.add(Dense(len(dataX[0]), activation='relu'))
# model.add(Dense(len(dataX[0]), activation='relu'))
# model.add(Dense(len(dataX[0]), activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=100, epochs=3, verbose=1)
model.save('model.h5')
print("Обучаем")
# Делаем предсказания класса на основе testX
prediction = model.predict(testX)
# Вычисляем ошибку
eval = model.evaluate(testX, testY, verbose=1)
print(eval)
# Подготовливаем данные для графика
graph_prediction = []
for i in range(len(prediction)):
    graph_prediction.append(find_max(prediction[i]))
# Просто очень нравится list'ы
graph_real = []
sum = 0
correct = 0
for i in range(len(testY)):
    graph_real.append(find_max(testY[i]))
    sum += 1
    if find_max(prediction[i]) == find_max(testY[i]):
        correct += 1

plt.plot(graph_real)
plt.plot(graph_prediction, 'ro')
plt.ylabel('Win?')
plt.xlabel('Match  ' + str(correct / sum))
plt.title('Loss value/Accuracy ' + str(eval))
plt.show()
