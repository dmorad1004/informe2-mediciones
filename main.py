from distutils.log import error
from black import err
import matplotlib.pyplot as plt
import numpy as np
import math


# prepare some data
turns_15 = [[0, 20, 40, 60, 80, 140, 180, 200], [0, 8, 16, 22, 28, 38, 44, 46]]
turns_10 = [[0, 100, 120, 150, 170, 180, 200, 300], [0, 24, 28, 32, 34, 36, 40, 50]]
turns_5 = [
    [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
    [0, 16, 22, 28, 32, 38, 40, 44, 46, 48, 50],
]

Bt_real = 0.000026980


def plotWithLineaRegression(x, y, N):
    y_err = np.array(
        list(map(lambda x: (1 / math.cos(x * np.pi / 180) ** 2) * np.pi / 90, x))
    )

    y = np.array(list(map(lambda y: y / 1000, y)))
    x = np.array(list(map(lambda x: math.tan(x * math.pi / 180), x)))

    plt.figure(figsize=(7, 7))

    b, a = np.polyfit(y, x, 1)
    p, V = np.polyfit(y, x, 1, cov=True)

    print(np.polyfit(y, x, 1, full=True))

    r2 = np.corrcoef(y, x)[0, 1] ** 2

    plt.scatter(y, x, s=50, alpha=0.7, edgecolors="k")

    ax = plt.subplot()
    ax.set_box_aspect(0.7)

    xseq = np.linspace(0, max(y) + 0.05, num=100)

    # Plot regression line
    plt.plot(xseq, a + b * xseq, label=f"y={np.round(b,5)}x+{np.round(a,5)}")

    plt.xlabel("Corriente I (mA)", fontsize=18, fontweight="bold")
    plt.ylabel("tan(Θ)", fontsize=18, fontweight="bold")
    plt.title(f"N={N}", fontweight="bold")

    slope_err = np.sqrt(V[0][0])
    print("slope: {} +/- {}".format(p[0], slope_err))
    print("intercept: {} +/- {}".format(p[1], np.sqrt(V[1][1])))

    plt.errorbar(y, x, yerr=y_err, fmt=".")

    bt, _ = calculeMagneticField(b, N, slope_err)

    err_absoluto = abs(float(bt) - Bt_real)
    err_porcentual = (err_absoluto / Bt_real) * 100

    print(f"err_absoluto:{err_absoluto}", f"err_porcentual:{err_porcentual} %")

    plt.legend()
    plt.text(0, 1.358, rf"$R^{2}={r2}$")
    plt.show()


def calculeMagneticField(slope, N, err):
    U0 = (4 * np.pi) * 10**-7
    R = 0.06

    bt = (U0 * N) / (2 * R * slope)
    bt_err = bt * (err / slope)

    return ("{:.20f}".format(bt), "{:.20f}".format(bt_err))


plotWithLineaRegression(turns_15[1], turns_15[0], 15)
plotWithLineaRegression(turns_10[1], turns_10[0], 10)
plotWithLineaRegression(turns_5[1], turns_5[0], 5)
