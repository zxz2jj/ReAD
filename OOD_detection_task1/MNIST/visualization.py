from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pylab as plt


def t_sne_visualization(data, category_number):
    colour_label = []
    combination_abstraction = []
    for category in range(category_number):
        combination_abstraction.append(data[category]['correct_pictures'])
        colour_label.extend([category for _ in range(data[category]['correct_pictures'].shape[0])])

    combination_abstraction = np.concatenate(combination_abstraction, axis=0)
    embedded = TSNE(n_components=2).fit_transform(combination_abstraction)
    x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
    embedded = embedded / (x_max - x_min)
    plt.scatter(embedded[:, 0], embedded[:, 1],
                c=(np.array(colour_label)/10.0), s=1)
    plt.axis('off')
    plt.show()
    plt.close()
