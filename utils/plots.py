import matplotlib.pyplot as plt
import numpy as np

def plot_runner_method(runners, method_name="plot_feature_importance", selected_targets=None, **kwargs):
    """
    Calls a specified plotting method (like 'plot_feature_importance', 'confusion_matrix', etc.)
    on selected runner objects and displays them in subplots.

    Parameters:
    - runners: dict of runner objects
    - method_name: str, the method to call on each runner (must accept an Axes object as a parameter)
    - selected_targets: list of keys in runners to plot. If None, plots all.
    - **kwargs: extra keyword arguments to pass to the method
    """
    if selected_targets is None:
        selected_targets = list(runners.keys())

    n = len(selected_targets)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    for i, target in enumerate(selected_targets):
        runner = runners[target]
        method = getattr(runner, method_name) # equivalent to for example runner.plot_confusion_matrix()
        method(axes[i], **kwargs)
        axes[i].set_title(target)

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()