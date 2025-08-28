# Contains utilities for plotting Risk-Abstention Curves and ROC Curves
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================================
# Risk-Abstention Curve utils
# =============================================================================================
# make data for risk abstention curves


def compute_risk(scores, correctness, thres):
    """
    Computes the estimated risk given a threshold: total loss / number of samples

    Note that loss = 1 if and only if thres < score and correctness is False
    """
    total_loss = 0

    for idx, score in enumerate(scores):
        if (score > thres) and (not correctness[idx]):
            total_loss += 1

    return total_loss / len(scores)


def compute_abstention_rate(scores, thres):
    """
    Computes the proportion of scores < thres
    """
    return np.sum(scores < thres) / len(scores)


def prepare_risk_abstention(scores, correctness, n_steps: int):
    """
    Creates a dictionary that is ready to be fed into `plot_risk_abstention_tradeoff()`

    ## Parameters
    > **scores**
    >> A list or numpy array of computed (or generated) scores

    > **correctness**
    >> A corresponding list or numpy array of binary yes/no scores for caption correctness

    > **n_steps**
    >> The number of lambda values to try. These will be separated linearly.

    ## Returns
    > **results**: *dict*
    >> Each of keywords "lambdas", "risks", and "abstention_rates" is associated with an array of equal length.
        The "lambdas" array has the different tested thresholds.
        The "risks" array has computed risks for each threshold.
        The "abstention_rates" array has computed abstention rates for each threshold
    """
    # prepare thresholds
    min_score = np.min(scores)
    max_score = np.max(scores)

    lambdas = np.linspace(min_score, max_score, n_steps)

    # get risks and abstention rates for each lambda
    risks = np.zeros(n_steps)
    abstention_rates = np.zeros(n_steps)

    for idx, thres in enumerate(lambdas):
        risks[idx] = compute_risk(scores, correctness, thres)
        abstention_rates[idx] = compute_abstention_rate(scores, thres)

    # create the dict
    results = {"lambdas": lambdas, "risks": risks, "abstention_rates": abstention_rates}
    return results


def plot_risk_abstention_tradeoff(results: dict, annotate: bool = True):
    """
    Visualize how Risk and Abstention change with thresholded lambda values as well as the Risk-Abstention Curve

    ## Parameters:
    > **results**: *dict*
    >> Each of keywords "lambdas", "risks", and "abstention_rates" is associated with an array of equal length.
        The "lambdas" array has the different tested thresholds.
        The "risks" array has computed risks for each threshold.
        The "abstention_rates" array has computed abstention rates for each threshold

    > **annotate**: *bool, optional (default True)*
    >> Decides if different thresholds should be shown on the ROC curve.

    ## Returns
    > **fig**: *matplotlib.pyplot.figure object*

    > **axes**: *matplotlib.pyplot.axes object*
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    lambdas = results["lambdas"]
    risks = results["risks"]
    abstention_rates = results["abstention_rates"]

    # rescale lambdas to fall between 0 and 1
    min_lambda = np.min(lambdas)
    max_lambda = np.max(lambdas)
    lambdas = (np.array(lambdas) - min_lambda) / (max_lambda - min_lambda)

    # Plot 1: Risk vs λ
    axes[0].plot(lambdas, risks, "b-o", linewidth=2, markersize=6)
    axes[0].set_xlabel("λ (Abstention Threshold)")
    axes[0].set_ylabel("Risk R(λ)")
    axes[0].set_title("Risk vs Threshold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    # Plot 2: Abstention Rate vs λ
    axes[1].plot(lambdas, abstention_rates, "r-o", linewidth=2, markersize=6)
    axes[1].set_xlabel("λ (Abstention Threshold)")
    axes[1].set_ylabel("Abstention Rate T(λ)")
    axes[1].set_title("Abstention Rate vs Threshold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    # Plot 3: Risk vs Abstention Rate
    axes[2].plot(risks, abstention_rates, "g-o", linewidth=2, markersize=6)
    axes[2].set_xlabel("Risk R(λ)")
    axes[2].set_ylabel("Abstention Rate T(λ)")
    axes[2].set_title("Risk vs Abstention Trade-off")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(-0.05, 1.05)
    axes[2].set_ylim(-0.05, 1.05)

    # Add annotations for key points
    for i, lambda_val in enumerate(lambdas[::2]):  # Annotate every other point
        if i * 2 < len(lambdas):
            idx = i * 2
            axes[2].annotate(
                f"λ={lambda_val:.4}",
                (risks[idx], abstention_rates[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    plt.tight_layout()
    plt.show()

    return fig, axes


# =============================================================================================
# ROC Curve utils
# =============================================================================================


def compute_roc(computed_scores, correct_scores, thres):
    """
    Computes the false and true positive rates. Assumes that computed_score < thres is a negative.

    ## Parameters:
    > **computed_scores**: *list, numpy array*
    >> Contains computed scores

    > **correct_scores**: *list, numpy array*
    >> Contains baseline binary values for knowing if we have true or false positives or negatives

    > **thres**: *numeric*
    >> A threshold for comparing the computed scores against.

    # Returns:
    > **tp_rate**: *float*
    >> The true positive rate. This is the number of true positives divided by the total number of positives in `correct_scores`.

    > **fp_rate**: *float*
    >> The false positive rate. This is the number of false positives divided by the total number of negatives in `correct_scores`.

    """
    # First, get positives and negatives
    abstain_policy = np.array(computed_scores) < thres

    # compute true positive rate
    # positive means abstain, but correct means it should not have abstained
    n_true_positives = np.sum(
        np.logical_and(abstain_policy, np.logical_not(correct_scores))
    )
    n_real_positives = np.sum(np.logical_not(correct_scores))
    tp_rate = n_true_positives / n_real_positives

    # compute false positive rate
    # positive abstain, not correct means it should have abstained
    n_false_positives = np.sum(np.logical_and(abstain_policy, correct_scores))
    n_real_negatives = np.sum(correct_scores)
    fp_rate = n_false_positives / n_real_negatives

    return tp_rate, fp_rate


def plot_roc(
    computed_scores, correct_scores, n_steps: int, label: str, annotate: bool = True
):
    """
    Visualizes the ROC curve for the computed scores with a certain number of steps.

    ## Parameters:
    > **computed_scores**: *list, numpy array*
    >> Contains computed scores

    > **correct_scores**: *list, numpy array*
    >> Contains baseline binary values for knowing if we have true or false positives or negatives

    > **n_steps**: *int*
    >> Controls the roughness of the ROC curve

    > **label**: *str*
    >> The label for the ROC curve (usually the score function)

    > **annotate**: *bool, optional (default True)*
    >> Decides if different thresholds should be shown on the ROC curve.

    ## Returns
    > **fig**: *matplotlib.pyplot.figure object*

    > **axes**: *matplotlib.pyplot.axes object*
    """
    # set up thresholds
    min_val = np.min(computed_scores)
    max_val = np.max(computed_scores)
    thresholds = np.linspace(min_val, max_val, n_steps)

    # Pre-allocate numpy arrays for the true/false positive rates
    # We add a +1 because we want it to touch the top right corner
    tp_rates = np.ones(n_steps + 1)
    fp_rates = np.ones(n_steps + 1)

    # fill in the values
    for idx, thres in enumerate(thresholds):
        # First, we need false positive/negative rates
        tp_rate, fp_rate = compute_roc(computed_scores, correct_scores, thres)
        # Append
        tp_rates[idx] = tp_rate
        fp_rates[idx] = fp_rate

    # rescale thresholds to fall between 0 and 1
    min_thres = np.min(thresholds)
    max_thres = np.max(thresholds)
    thresholds = (np.array(thresholds) - min_thres) / (max_thres - min_thres)

    # Plot
    fig, axes = plt.subplots(1, 1, figsize=(12, 12))
    axes.plot(fp_rates, tp_rates, "b-o", linewidth=2, label=label, alpha=0.8)

    axes.set_xlabel(
        "False Positive Rate (FPR)\nRate at which policy incorrectly abstains when model was correct.",
        fontsize=12,
    )
    axes.set_ylabel(
        "True Positive Rate (TPR)\nRate at which policy correctly abstains when model was incorrect",
        fontsize=12,
    )
    axes.set_title(
        f"ROC Curve (N = {len(computed_scores)}) using Abstention Policy with {n_steps} Lambda thresholds",
        fontsize=14,
        pad=20,
    )

    # Set ticks
    major_ticks = np.arange(0, 1.1, 0.1)
    minor_ticks = np.arange(0, 1.1, 0.05)
    axes.set_xticks(major_ticks)
    axes.set_yticks(major_ticks)
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(minor_ticks, minor=True)

    # Grid
    axes.grid(True, which="major", alpha=0.6)
    axes.grid(True, which="minor", alpha=0.3)

    # Set axis limits
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)

    if annotate:
        # Add lambda annotations
        for fpr, tpr, lam in zip(
            fp_rates,
            tp_rates,
            thresholds,
        ):
            axes.annotate(
                f"λ={lam:.4}",
                (fpr, tpr),
                xytext=(0, 15),
                textcoords="offset points",
                fontsize=9,
                color="black",
                alpha=0.8,
            )

    # Add random classifier
    axes.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="Random Classifier")

    # Legend
    axes.legend(loc="lower right", fontsize=11)

    return fig, axes
