import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import seaborn as sns
import numpy as np
def plotti():
    colors = sns.color_palette('pastel')[0:5]

    emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    with open('outfile', 'rb') as fp:
        hist = pickle.load(fp)
    arr = dict()
    for i in emotions.keys():
        if hist.count(emotions[i])>0:
            arr[emotions[i]] = hist.count(emotions[i])

    fig = plt.figure()
    print(arr)
    ax1 = fig.add_subplot(1, 1, 1)
    labels = arr.keys()
    explode = [0.1 for x in range(len(labels))]
    plt.title('Mood:')
    ax1.pie(arr.values(), labels=labels, shadow=True, startangle=90,  autopct=lambda p: '{:.0f}%'.format(round(p)) if p > 1 else '', colors=colors,explode=explode)
    ax1.axis('equal')
    plt.show()
plotti()