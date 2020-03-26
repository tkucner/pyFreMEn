import numpy as np


class FreMen:
    def __init__(self, **kwargs):
        self.number_of_periodicities = (
            kwargs["number_of_periodicities"] if "number_of_periodicities" in kwargs else 100)
        self.period_in_seconds = (kwargs["period_in_seconds"] if "period_in_seconds" in kwargs else 604800)

        multies = range(1, self.number_of_periodicities + 1)
        self.periodicies = self.period_in_seconds * np.divide(np.ones(self.number_of_periodicities), multies)
        self.phase = np.zeros(self.number_of_periodicities)
        self.amplitude = np.zeros(self.number_of_periodicities)
        self.realStates = np.zeros(self.number_of_periodicities)
        self.imagStates = np.zeros(self.number_of_periodicities)
        self.realBalance = np.zeros(self.number_of_periodicities)
        self.imagBalance = np.zeros(self.number_of_periodicities)
        self.gain = 0
        self.points_count = 0

        self.times = np.array([])
        self.states = np.array([])

    def add_observations(self, times, states):
        self.times = np.array(times if self.times.size == 0 else np.append(self.times, times))
        self.states = np.array(states if self.states.size == 0 else np.append(self.states, states))

    def update(self):
        self.gain = (self.gain * self.points_count + np.sum(self.states)) / (self.points_count + self.states.size)
        self.points_count = self.points_count + self.states.size
        for state, time in zip(self.states, self.times):
            for periodicies_id, period in enumerate(self.periodicies):
                angle = 2 * np.pi * time / period
                self.realStates[periodicies_id] = self.realStates[periodicies_id] + state * np.cos(angle)
                self.imagStates[periodicies_id] = self.imagStates[periodicies_id] + state * np.sin(angle)
                self.realBalance[periodicies_id] = self.realBalance[periodicies_id] + self.gain * np.cos(angle)
                self.imagBalance[periodicies_id] = self.imagBalance[periodicies_id] + self.gain * np.sin(angle)

    def reconstruct(self, **kwargs):
        order = (kwargs["order"] if "order" in kwargs else 100)
        reconstruction_time = (
            kwargs["reconstruction_time"] if "reconstruction_time" in kwargs else range(0, 601200, 3600))

        amplitude = np.zeros(self.number_of_periodicities)
        phase = np.zeros(self.number_of_periodicities)
        for periodicies_id, (res, reb, ims, imb) in enumerate(
                zip(self.realStates, self.realBalance, self.imagStates, self.imagBalance)):
            re = res - reb
            im = ims - imb
            amplitude[periodicies_id] = np.sqrt(re ** 2 + im ** 2) / self.points_count
            if amplitude[periodicies_id] < 0:
                amplitude[periodicies_id] = 0
            phase[periodicies_id] = np.arctan2(im, re)
        amplitude_indices = np.argsort(amplitude)
        p = np.repeat(self.gain, len(reconstruction_time))
        for rt_id, rt in enumerate(reconstruction_time):
            for i in range(order):
                p[rt_id] = p[rt_id] + 2 * amplitude[amplitude_indices[i]] * np.cos(
                    rt / self.periodicies[amplitude_indices[i]] * 2 * np.pi - phase[amplitude_indices[i]])

            if p[rt_id] > 1:
                p[rt_id] = 1.0
            if p[rt_id] < 0:
                p[rt_id] = 0.0
        return p,reconstruction_time
