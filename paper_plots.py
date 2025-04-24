import matplotlib.pyplot as plt
import numpy as np


def batched_data(data, samples):
    data_mean = []
    data_deviation = []
    for i in range(0, len(data), samples):
        data_mean.append(np.mean(data[i:i + samples]))
        data_deviation.append(np.std(data[i:i + samples]))

    m, d = np.array(data_mean), np.array(data_deviation)
    print(m.shape, d.shape)
    return m, d


def basic_plot_error(data_mean, data_deviation, subplot, data_label, y_label, title, color="b", linestyle="solid"):
    subplot.plot(data_mean, label=data_label, color=color, linestyle=linestyle)
    std_err = data_deviation / np.sqrt(np.size(data_deviation, axis=0))

    err_plus = data_mean + std_err
    err_minus = data_mean - std_err

    subplot.fill_between(range(np.size(data_deviation, axis=0)), err_minus, err_plus, alpha=0.4, color=color)
    subplot.legend()
    subplot.set_xlabel("Iterations")
    subplot.set_ylabel(y_label)
    subplot.set_title(title)


def basic_plot(data, subplot, data_label, y_label, title, color="b", linestyle="solid"):
    subplot.plot(data, label=data_label, color=color, linestyle=linestyle)
    subplot.legend()
    subplot.set_xlabel("Iterations")
    subplot.set_ylabel(y_label)
    subplot.set_title(title)


total_loss = []
policy_loss = []
value_loss = []
episode_len = []
win = []
draw = []
loss = []

order = [total_loss, policy_loss, value_loss, episode_len, win, draw, loss]
models = [0, 1, 2, 3]
for m in models:
    with open(f"models/config_{m}/run_data_.txt", "r") as f:
        for o in order:
            line = f.readline()
            elements = line.split(",")
            elements.pop(0)
            x = [float(i) for i in elements]
            o.append(np.array(x))

colours = ["royalblue", "green", "sienna", "orange", "grey"]
# Losses
_, axs = plt.subplots(1)
for m in models:
    t = batched_data(total_loss[m], 8)
    p = batched_data(policy_loss[m], 8)
    v = batched_data(value_loss[m], 8)

    basic_plot_error(t[0], t[1], axs, f"Total loss - C{m}", "Loss", "Training loss per iteration", color=colours[m], linestyle="dotted")
    basic_plot_error(p[0], p[1], axs, f"Policy loss - C{m}", "Loss", "Training loss per iteration", color=colours[m], linestyle="solid")
    basic_plot_error(v[0], v[1], axs, f"Value loss - C{m}", "Loss", "Training loss per iteration", color=colours[m], linestyle="dashed")
plt.legend(fontsize="small", ncols=4)
plt.ylim(0, 4)
plt.show()

# Episode len
_, axs = plt.subplots(1)
for m in models:
    e = batched_data(episode_len[m], 30)
    basic_plot(e[0], axs, f"Configuration {m}", "Episode length", "Average episode length per iteration", color=colours[m])
plt.ylim(0, 42)
plt.show()

# Win rate
_, axs = plt.subplots(1)
for m in models:
    w = batched_data(win[m], 30)
    basic_plot(w[0], axs, f"Configuration {m}", "% of wins", "Average win rate per iteration", color=colours[m])
plt.ylim(0, 1)
plt.show()
