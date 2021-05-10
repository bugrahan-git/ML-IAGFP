import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from Util import get_images

colors = ['yellowgreen',
          'red',
          'gold',
          'lightskyblue',
          'white',
          'lightcoral',
          'blue',
          'pink',
          'darkgreen',
          'yellow',
          'grey',
          'violet',
          'magenta',
          'cyan'
          ]


def plot_pie_chart(data, title, isExplode=False, isShadow=True, sortLegend=True):
    counter = Counter(data)
    counts = np.array([v for v in counter.values()])
    labels = np.array([str(k).strip() for k in counter])

    porcent = 100. * counts / counts.sum()

    explode = np.zeros(len(counts))
    if isExplode:
        explode[counts.argmax()] = 0.1

    patches, texts = plt.pie(counts, colors=colors[:len(labels)], startangle=0, radius=1.2, explode=explode,
                             shadow=isShadow,
                             wedgeprops={"edgecolor": "k", 'linewidth': 2, 'antialiased': True})
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(labels, porcent)]

    if sortLegend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, counts),
                                             key=lambda x: x[2],
                                             reverse=True))
    plt.legend(patches, labels, bbox_to_anchor=(1, 0.5), loc="center right", fontsize=10,
               bbox_transform=plt.gcf().transFigure)

    plt.title(title)
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
    plt.savefig("plots/" + title + '.png', bbox_inches='tight')
    plt.show()


def pie():
    image_list_train, artist_list_train = get_images("train_artist.mat", "artists")
    image_list_test, artist_list_test = get_images("test_artist.mat", "artists")

    _, movement_list_train = get_images("train_movement.mat", "movements")
    _, movement_list_test = get_images("test_movement.mat", "movements")

    image_list = [*image_list_train, *image_list_test]
    artist_list = [*artist_list_train, *artist_list_test]
    movement_list = [*movement_list_train, *movement_list_test]

    plot_pie_chart(artist_list, "Artist Distribution After Data Augmentation")
    plot_pie_chart(movement_list, "Movement Distribution After Data Augmentation")


def line(X, y, xLabel, yLabel, title, color="yellowgreen"):
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.plot(X, y, color=color)
    plt.savefig("plots/" + title + '.png')
    plt.show()
