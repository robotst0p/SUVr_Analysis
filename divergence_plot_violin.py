import pandas as pd
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import markers

# load in whole divergence frame of 800 rows
# frame is 800 rows by 4 columns
divergence_frame = pd.read_pickle("./whole_divergence_frame.pkl")

# adjust dataframe for region column with list of regions
region_list = [
    "ctx_lh_caudalanteriorcingulate",
    "ctx_lh_isthmuscingulate",
    "ctx_lh_posteriorcingulate",
    "ctx_lh_rostralanteriorcingulate",
    "ctx_rh_caudalanteriorcingulate",
    "ctx_rh_isthmuscingulate",
    "ctx_rh_posteriorcingulate",
    "ctx_rh_rostralanteriorcingulate",
]

# begin constructing frame for plotting data and converting divergence frame data from wide to long format for violin plot
plot_frame = pd.DataFrame(columns=["KL Type", "Region", "Divergence"])


def kl_type(divergence_frame, plot_frame):
    l = 0
    k = 800

    for column in divergence_frame.columns:
        for i in range(l, k):
            plot_frame.loc[i, "KL Type"] = column

        l += 800
        k += 800
    return plot_frame


plot_frame = kl_type(divergence_frame, plot_frame)

# fill in brain region for each KL divergence value point
def region_type(divergence_frame, plot_frame):
    for i in range(0, len(plot_frame.index)):
        if i < 8:
            plot_frame.loc[i, "Region"] = region_list[i]
        else:
            plot_frame.loc[i, "Region"] = region_list[i % 8]

    return plot_frame


plot_frame = region_type(divergence_frame, plot_frame)

# fill in the kl divergence values from the original divergence frame for each kl type and brain region
def divergence_values(divergence_frame, plot_frame):
    kl_point_list = []
    for column in divergence_frame.columns:
        current_list = divergence_frame.loc[:, column].tolist()
        for i in range(0, len(current_list)):
            kl_point_list.append(current_list[i])

    for i in range(0, len(plot_frame.index)):
        plot_frame.loc[i, "Divergence"] = kl_point_list[i]

    return plot_frame


plot_frame = divergence_values(divergence_frame, plot_frame)

# need different data frame for each region
# take the matching indexes from each frame and combine the rows into one frame: this will give us one frame for each of the 8 brain regions
def create_region_frame(
    plot_frame1,
    region,
    new_frame=pd.DataFrame(),
    new_frame2=pd.DataFrame(),
):
    new_frame = plot_frame1.loc[plot_frame1["Region"] == region]

    final_frame = new_frame.reset_index()

    return final_frame


lh_caudalanterior = create_region_frame(plot_frame, "ctx_lh_caudalanteriorcingulate")
lh_isthmus = create_region_frame(plot_frame, "ctx_lh_isthmuscingulate")
lh_posterior = create_region_frame(plot_frame, "ctx_lh_posteriorcingulate")
lh_rostralanterior = create_region_frame(plot_frame, "ctx_lh_rostralanteriorcingulate")
rh_caudalanterior = create_region_frame(plot_frame, "ctx_rh_caudalanteriorcingulate")
rh_isthmus = create_region_frame(plot_frame, "ctx_rh_isthmuscingulate")
rh_posterior = create_region_frame(plot_frame, "ctx_rh_posteriorcingulate")
rh_rostralanterior = create_region_frame(plot_frame, "ctx_rh_rostralanteriorcingulate")

brain_frames = [
    lh_caudalanterior,
    lh_isthmus,
    lh_posterior,
    lh_rostralanterior,
    rh_caudalanterior,
    rh_isthmus,
    rh_posterior,
    rh_rostralanterior,
]


# need to convert frame column datatypes to float
def to_float(plot_frame):
    plot_frame["Divergence"] = plot_frame["Divergence"].astype(float)

    return plot_frame


for frame in brain_frames:
    frame = to_float(frame)

print(rh_rostralanterior.dtypes)
# create violin plots for divergence values by region, grouped by kl type
f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows=2, ncols=4)
plt.title("KL Divergence Values Across Cingulate Regions")
# plot kl divergence values
sns.set(font_scale=0.8)
sns.set_style(rc={"axes.facecolor": sns.color_palette("pastel")[8]})
sns.violinplot(
    data=lh_caudalanterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "synthaud_originalaud": "orange",
        "synthaud_originalcontrol": "red",
        "synthcontrol_originalcontrol": "blue",
        "synthcontrol_originalaud": "cyan"
    },
    order=[
        "synthaud_originalaud",
        "synthaud_originalcontrol",
        " ",
        "synthcontrol_originalcontrol",
        "synthcontrol_originalaud"
    ],
    ax=ax1,
).set(title="LH Caudalanterior Cingulate", xticklabels=[], xlabel=None)
# ax1.legend(bbox_to_anchor=(-0.74, 1), loc="upper left", borderaxespad=0, fontsize=7.6)
sns.violinplot(
    data=lh_isthmus,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "synthaud_originalaud": "orange",
        "synthaud_originalcontrol": "red",
        "synthcontrol_originalcontrol": "blue",
        "synthcontrol_originalaud": "cyan"
    },
    order=[
        "synthaud_originalaud",
        "synthaud_originalcontrol",
        " ",
        "synthcontrol_originalcontrol",
        "synthcontrol_originalaud"
    ],
    ax=ax2,
    legend=False,
).set(title="LH Isthmus Cingulate", xticklabels=[], xlabel=None)
sns.violinplot(
    data=lh_posterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "synthaud_originalaud": "orange",
        "synthaud_originalcontrol": "red",
        "synthcontrol_originalcontrol": "blue",
        "synthcontrol_originalaud": "cyan"
    },
    order=[
        "synthaud_originalaud",
        "synthaud_originalcontrol",
        " ",
        "synthcontrol_originalcontrol",
        "synthcontrol_originalaud"
    ],
    ax=ax3,
    legend=False,
).set(title="LH Posterior Cingulate", xticklabels=[], xlabel=None)
sns.violinplot(
    data=lh_rostralanterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "synthaud_originalaud": "orange",
        "synthaud_originalcontrol": "red",
        "synthcontrol_originalcontrol": "blue",
        "synthcontrol_originalaud": "cyan"
    },
    order=[
        "synthaud_originalaud",
        "synthaud_originalcontrol",
        " ",
        "synthcontrol_originalcontrol",
        "synthcontrol_originalaud"
    ],
    ax=ax4,
    legend=False,
).set(title="LH Rostralanterior Cingulate", xticklabels=[], xlabel=None)
sns.violinplot(
    data=rh_caudalanterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "synthaud_originalaud": "orange",
        "synthaud_originalcontrol": "red",
        "synthcontrol_originalcontrol": "blue",
        "synthcontrol_originalaud": "cyan"
    },
    order=[
        "synthaud_originalaud",
        "synthaud_originalcontrol",
        " ",
        "synthcontrol_originalcontrol",
        "synthcontrol_originalaud"
    ],
    ax=ax5,
    legend=False,
).set(title="RH Caudalanterior Cingulate", xticklabels=[], xlabel=None)
sns.violinplot(
    data=rh_isthmus,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "synthaud_originalaud": "orange",
        "synthaud_originalcontrol": "red",
        "synthcontrol_originalcontrol": "blue",
        "synthcontrol_originalaud": "cyan"
    },
    order=[
        "synthaud_originalaud",
        "synthaud_originalcontrol",
        " ",
        "synthcontrol_originalcontrol",
        "synthcontrol_originalaud"
    ],
    ax=ax6,
    legend=False,
).set(title="RH Isthmus Cingulate", xticklabels=[], xlabel=None)
sns.violinplot(
    data=rh_posterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "synthaud_originalaud": "orange",
        "synthaud_originalcontrol": "red",
        "synthcontrol_originalcontrol": "blue",
        "synthcontrol_originalaud": "cyan"
    },
    order=[
        "synthaud_originalaud",
        "synthaud_originalcontrol",
        " ",
        "synthcontrol_originalcontrol",
        "synthcontrol_originalaud"
    ],
    ax=ax7,
    legend=False,
).set(title="RH Posterior Cingulate", xticklabels=[], xlabel=None)
sns.violinplot(
    data=rh_rostralanterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    width = 1,
    palette={
        "synthaud_originalaud": "orange",
        "synthaud_originalcontrol": "red",
        "synthcontrol_originalcontrol": "blue",
        "synthcontrol_originalaud": "cyan"
    },
    order=[
        "synthaud_originalaud",
        "synthaud_originalcontrol",
        " ",
        "synthcontrol_originalcontrol",
        "synthcontrol_originalaud"
    ],
    ax=ax8,
    legend=False,
).set(title="RH Rostralanterior Cingulate", xticklabels=[], xlabel=None)

plt.show()

# clear plot for individual 8 region graphs
plt.clf()
