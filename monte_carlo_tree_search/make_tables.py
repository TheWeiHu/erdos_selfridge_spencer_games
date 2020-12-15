import os
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt


def variance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)


f = []
for (dirpath, dirnames, filenames) in os.walk("./benchmark"):
    f.extend(filenames)
f = [i for i in f if re.match(r"(attacker_|defender_)(\d)+\.(\d)+\.csv", i)]
print(f)

# matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
#     color=["#010101", "#011926", "#003E5C", "#016293"]
# )

result = {}
for file in f:
    df = pd.read_csv("./benchmark/" + file)
    df["version"] = df["version"].apply(lambda x: re.sub("[^0-9]", "", x))
    df["version"] = df["version"].apply(lambda x: int(x) if x else 400)
    df = df.sort_values("version")
    df = df[
        [
            "version",
            "optimal_player",
            "mostly_optimal_player",
            "mostly_random_player",
            "random_player",
        ]
    ]

    df = df.rename(
        columns={
            "optimal_player": "Theorem 2.3 Player",
            "mostly_optimal_player": "Mostly Theorem 2.3 Player",
            "mostly_random_player": "Mostly Random Player",
            "random_player": "Random Player",
        }
    )

    for name in list(df):
        if name == "version":
            continue
        df[name] = df[name].apply(
            lambda x: (
                (eval(x)[0] - eval(x)[1]) / 50,
                variance(
                    [1] * eval(x)[0]
                    + [-1] * eval(x)[1]
                    + [0] * (50 - eval(x)[0] - eval(x)[1])
                ),
            ),
        )

    print(df)
    df = df.set_index("version").T

    fig = plt.figure()

    for i in range(len(df)):
        plt.plot(
            [k for k in df.columns], [df[y].iloc[i][0] for y in df.columns], linewidth=3
        )
        plt.fill_between(
            [k for k in df.columns],
            [df[y].iloc[i][0] - df[y].iloc[i][1] for y in df.columns],
            [df[y].iloc[i][0] + df[y].iloc[i][1] for y in df.columns],
            alpha=0.25,
            edgecolor="#FFFFFF",
        )
    potential = file.lstrip("attacker_").lstrip("defender_").rstrip(".csv")
    fig.suptitle(
        "Trained "
        + file.split("_")[0].capitalize()
        + " Against K = 20, Potential = "
        + potential,
        fontsize=14,
    )
    plt.xlabel("Iterations", fontsize=10)
    plt.ylabel("Average Score", fontsize=10)
    plt.legend(df.index, title="Opponents", loc="center right", prop={"size": 6})
    plt.axes().set_ylim([-1, 1])
    plt.savefig(file + 'visualize.png')
    print(df)
