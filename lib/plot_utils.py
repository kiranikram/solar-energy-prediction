import matplotlib.pyplot as plt

def save_plot(data_list, ylabel, xlabel, title, path):
    plt.clf()
    plt.plot(list(range(0, len(data_list))), data_list, color="blue")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(path)


