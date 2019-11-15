"""
Sentdex text: https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/?completed=/q
-learning-algorithm-reinforcement-learning-python-tutorial/


"""
import cv2
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

EPISODES = 10000
STEPS = 100

folder_path = "qtables/mountain_car/"

style.use("ggplot")


def get_q_colr(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3


def make_images():

    fig = plt.figure(figsize=(12, 9))

    for i in range(0, EPISODES, STEPS):
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        file_path = folder_path + f"{i}-qtable.npy"
        q_table = np.load(file_path)

        for x, x_vals in enumerate(q_table):
            for y, y_vals in enumerate(x_vals):
                ax1.scatter(x, y, c=get_q_colr(y_vals[0], y_vals)[0], marker="o", alpha=get_q_colr(y_vals[0], y_vals)[1])
                ax2.scatter(x, y, c=get_q_colr(y_vals[1], y_vals)[0], marker="o", alpha=get_q_colr(y_vals[1], y_vals)[1])
                ax3.scatter(x, y, c=get_q_colr(y_vals[2], y_vals)[0], marker="o", alpha=get_q_colr(y_vals[2], y_vals)[1])

                ax1.set_ylabel("Action 0")
                ax2.set_ylabel("Action 1")
                ax3.set_ylabel("Action 2")

        plt.savefig(folder_path + "q_table_charts/" + f"{i}.png")
        plt.clf()


def make_video():
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("qlearn.avi", fourcc, 60.0, (1200, 900))

    for i in range(0, EPISODES, STEPS):
        img_path = folder_path + "q_table_charts/" + f"{i}.png"
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()


make_images()
make_video()
