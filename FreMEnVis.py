import matplotlib.pyplot as plt
def show(**kwargs):
    times = kwargs["times"]
    states = kwargs["states"]
    probabilities = kwargs["probabilities"]
    reconstruction_times = kwargs["reconstruction_times"]
    plt.plot(times, states, 'rx')
    plt.plot(reconstruction_times, probabilities, 'b')
    plt.show()

