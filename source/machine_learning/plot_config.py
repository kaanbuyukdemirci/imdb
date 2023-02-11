import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def config():
    style = "seaborn-v0_8-paper"
    plt.style.use(style)
    #SMALL_SIZE = 16
    #MEDIUM_SIZE = 18
    #BIGGER_SIZE = 20
    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def smooth_same(y, box_pts):
    box = np.ones(box_pts)/box_pts
    checker = np.ones(y.shape[0])
    checker = np.convolve(checker, box, 'same')
    checker = 1/checker
    y_smooth = np.convolve(y, box, mode='same')*checker
    return y_smooth

# plots
if __name__ == "__main__":
    # example
    config()
    x1 = np.array([1,2,3])
    y1 = np.array([1,2,3])
    x2 = np.array([1,2,3])
    y2 = np.array([3,2,1])
    plt.figure(0, figsize=(9,6))
    plt.clf()
    plt.plot(x1, y1, linewidth=3)
    plt.plot(x2, y2, linewidth=3)
    plt.legend(["legend 1", "legend 2"])
    plt.title("title")
    plt.xlabel("x label")
    plt.ylabel("y label")
    plt.grid(True, color=mcolors.CSS4_COLORS["grey"], linestyle='--', linewidth=0.5)
    plt.locator_params(axis="both", nbins=10, tight=True)
    plt.show()