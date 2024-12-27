import matplotlib.pyplot as plt

def plot2d(samples, index_x=0, index_y=1, **kwargs):
    plt.scatter(samples[:, index_x], samples[:, index_y], **kwargs)
    plt.legend()