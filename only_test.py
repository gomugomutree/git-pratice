# import os
# import send2trash
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
# # send2trash.send2trash(r"C:\Users\hojun\Desktop\test_place\1\1.json")

# # a = "C:/Users/hojun/Desktop/test_place/1/2.jpg"
# # b = os.path.basename(a)
# # print(os.path.splitext(a))

# bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
#                 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
#                 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
# bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
#                 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
#                 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
                
# smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
# # plt.scatter(bream_length, bream_weight)
# # plt.scatter(smelt_length, smelt_weight)
# # plt.xlabel('길이')
# # plt.ylabel('무게')
# # plt.show()


# length = bream_length + smelt_length
# weight = bream_weight + smelt_weight

# fish_data = [[l,w] for l,w in zip(length,weight)]
# fish_trarget = [1]*35 + [0]*14


# kn = KNeighborsClassifier()
# kn.fit(fish_data,fish_trarget)
# print(kn.score(fish_data,fish_trarget))

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_validate
# from sklearn.ensemble import RandomForestClassifier


# wine = pd.read_csv('http://bit.ly/wine_csv_data')
# data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
# target = wine['class'].to_numpy()
# train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# rf = RandomForestClassifier(n_jobs=-1, random_state=42)
# scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
# print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# rf.fit(train_input, train_target)
# print(rf.score)
# print(rf.feature_importances_)

import numpy as np
def activation(z):
    return 1 / (1+np.exp(-z))
def AND(x,y):
    wx = 20
    wy = 20
    bias = -30

    if activation((wx*x + wy*y + bias)) < 0.5: 
        return 0
    else: 
        return 1

print(f"x=0, y=0: {AND(0,0)}")
print(f"x=0, y=1: {AND(0,1)}")
print(f"x=1, y=0: {AND(1,0)}")
print(f"x=1, y=1: {AND(1,1)}")
print()
def OR(x,y):
    # or 게이트로 작동하기위한 웨이트값 코딩 [+]
    wx = 20
    wy = 20
    bias = -10

    if activation((wx*x + wy*y + bias)) < 0.5: 
        return 0
    else: 
        return 1

print(f"x=0, y=0: {OR(0,0)}")
print(f"x=0, y=1: {OR(0,1)}")
print(f"x=1, y=0: {OR(1,0)}")
print(f"x=1, y=1: {OR(1,1)}")