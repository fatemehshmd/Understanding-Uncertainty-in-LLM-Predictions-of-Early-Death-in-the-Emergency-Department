"""
Utility functions for conformal prediction analysis with LLM predictions.

This module contains the core functionality extracted from the Jupyter notebook
for better version control and modularity.
"""

import numpy as np
import pandas as pd
from collections import Counter
import ast
import math
from matplotlib import pyplot as plt
import seaborn as sns


def str_to_list(x):
    """Convert string representation of list to actual list."""
    return ast.literal_eval(x)


def most_comm(lst):
    """Find the most common element in a list."""
    data = Counter(lst)
    return data.most_common(1)[0][0]


def least_common(x):
    """Find the least common element between Yes/No and return complement."""
    c = x.most_common(1)[0]
    if c[0].lower() == 'yes':
        return ('No', 30 - c[1])
    else:
        return ('Yes', 30 - c[1])


def normalized_yes_probability(lst):
    """
    Calculate normalized probabilities for Yes/No/Other from logprob entries.
    
    Args:
        lst: List of logprob entries for each sample
        
    Returns:
        List of [no_prob, yes_prob, other_prob] for each sample
    """
    y_n_o_probs_all = []
    
    for entries in lst:
        y_n_o_probs = []
        # Convert logprobs to standard probabilities
        probs = [(text, math.exp(logprob)) for text, logprob in entries]
        
        # Total probability for normalization
        total_prob = sum(prob for _, prob in probs)
        
        # Sum of probabilities for entries containing "yes" (case-insensitive)
        yes_prob = sum(prob for text, prob in probs if 'yes' in text.lower())
        
        # Sum of probabilities for entries containing "no" (case-insensitive)
        no_prob = sum(prob for text, prob in probs if 'no' in text.lower())
        
        # Normalized probability
        y_n_o_probs.append(no_prob / total_prob if total_prob > 0 else 0.0)
        y_n_o_probs.append(yes_prob / total_prob if total_prob > 0 else 0.0)
        o_prob = 1 - y_n_o_probs[0] - y_n_o_probs[1]
        y_n_o_probs.append(o_prob)
        
        y_n_o_probs_all.append(y_n_o_probs)
    
    return y_n_o_probs_all


def sample_final_prediction(lst):
    """Get the class index with highest probability for each sample."""
    l = []
    for item in lst:
        l.append(item.index(max(item)))
    return l


def true_class_prob(x):
    """Extract true class probabilities from probability array and true labels."""
    return [sublist[x[1]] for sublist in x[0]]


def calc_non_conformity(data):
    """
    Calculate non-conformity scores for the dataset.
    
    Args:
        data: DataFrame with prediction_logprob and HOSPITAL_EXPIRE_FLAG columns
        
    Returns:
        DataFrame with additional columns for probabilities and non-conformity scores
    """
    data['prediction_logprob'] = data['prediction_logprob'].apply(str_to_list)
    data['y_n_o_probs_all'] = data['prediction_logprob'].apply(normalized_yes_probability)
    data['sample_final_prediction'] = data['y_n_o_probs_all'].apply(sample_final_prediction)
    data['final_prediction'] = data['sample_final_prediction'].apply(most_comm)
    data['HOSPITAL_EXPIRE_FLAG_CAT'] = data['HOSPITAL_EXPIRE_FLAG'].apply(lambda x: 'Yes' if x == 1 else 'No')
    data['true_class_prob'] = data[['y_n_o_probs_all', 'HOSPITAL_EXPIRE_FLAG']].apply(true_class_prob, axis=1)
    data['nonconformity_scores'] = data['true_class_prob'].apply(lambda lst: [1 - p for p in lst])
    return data


class PredictionClass:
    """Class for handling conformal prediction results."""
    
    def __init__(self, p, eps):
        self.p = p
        self.eps = eps

    def classes(self, eps=None):
        """Return classes that meet the epsilon threshold."""
        if eps is None:
            eps = self.eps
            assert(eps is not None)
        return [y for p_y, y in self.p if p_y > eps]

    def verdict(self, ref, eps=None):
        """Check if reference class is in the prediction set."""
        return ref in self.classes(eps)

    def confidence(self):
        """Calculate confidence score."""
        return 1 - sorted([p_y for p_y, y in self.p], reverse=True)[1]

    def credibility(self):
        """Calculate credibility score."""
        return max(p_y for p_y, y in self.p)


def conformal_prediction(cal_scores, test_in, alpha=0.1, verbose=True):
    """
    Perform conformal prediction on test data.
    
    Args:
        cal_scores: Calibration non-conformity scores
        test_in: Test dataframe with probability columns
        alpha: Significance level (default 0.1)
        verbose: Whether to print coverage statistics
        
    Returns:
        DataFrame with conformal prediction results
    """
    # Make a copy of test_in to not mutate input df
    test = test_in.copy()
    
    # Convert test probabilities to a numpy array
    preds = test[['no', 'yes', 'other']].to_numpy()
    
    # Build prediction results with a nested list comprehension
    prediction_results = [
        PredictionClass(
            [
                (
                    # Using inverse probability nonconformity measure
                    (np.sum(cal_scores >= (1 - pred)) + 1) / (len(cal_scores) + 1),
                    cls
                )
                for cls, pred in enumerate(example)
            ],
            eps=alpha
        )
        for example in preds
    ]
    
    # Assign the PredictionClass results and compute the other columns
    test['confidence'] = [cp.confidence() for cp in prediction_results]
    test['credibility'] = [cp.credibility() for cp in prediction_results]
    test['classes'] = [cp.classes() for cp in prediction_results]
    test['verdict'] = [cp.verdict(cls) for cp, cls in zip(prediction_results, test['class'])]
    test['cp'] = prediction_results

    # Calculate and print empirical coverage and avg prediction set size if verbose is enabled
    if verbose:        
        empirical_coverage = test['verdict'].mean()
        print(f"The empirical coverage is: {100 * empirical_coverage:.2f}%")
        
        # Compute empirical coverage per class
        coverage_by_class = test.groupby('class')['verdict'].mean()
        for cls, coverage in coverage_by_class.items():
            print(f"Empirical coverage for class {cls}: {100 * coverage:.2f}%")
    
        pred_set_size = test['classes'].apply(len).mean()
        print(f"Mean prediction set size is: {pred_set_size:.2f}")
        
        # Compute mean prediction set size per class
        pred_set_size_by_class = test.groupby('class')['classes'].apply(
            lambda x: x.map(len).mean()
        )
        for cls, ps_size in pred_set_size_by_class.items():
            print(f"Mean prediction set size for class {cls}: {ps_size:.2f}")
        print()
    
    return test


def conformal_prediction_quantile_based(cal_scores, test, alpha=0.1, verbose=True):
    """
    Perform quantile-based conformal prediction.
    
    Args:
        cal_scores: Calibration non-conformity scores
        test: Test dataframe with probability columns
        alpha: Significance level (default 0.1)
        verbose: Whether to print coverage statistics
        
    Returns:
        DataFrame with prediction sets
    """
    # Get the adjusted quantile
    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
        
    # Form conformal prediction sets
    pred_sets = test[['no', 'yes', 'other']] >= (1 - qhat)
        
    # Calculate empirical coverage
    empirical_coverage = pred_sets.to_numpy()[np.arange(pred_sets.shape[0]), test['class']].mean()

    if verbose:
        print(f'qhat: {qhat}\n')
        print(pred_sets.apply(lambda x: int(x['no']) + int(x['yes']) + int(x['other']), axis=1).value_counts())
        print(f"\nThe empirical coverage is: {empirical_coverage:.3f}\n\n")
        
    return pred_sets


def plot_nonconformity_histograms(cal_data, save_path='Mean_non_conform_flag.jpg'):
    """
    Plot histograms of mean nonconformity scores by mortality flag.
    
    Args:
        cal_data: Calibration dataframe with mean_nonconformity_scores and HOSPITAL_EXPIRE_FLAG
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)
    sns.histplot(cal_data[cal_data['HOSPITAL_EXPIRE_FLAG'] == 0]['mean_nonconformity_scores'], 
                 bins=20, kde=True, ax=axes[0], color='C0')
    axes[0].set_title('In-hospital survived', fontdict={'size': 14, 'weight': 'bold'})
    axes[0].set_xlabel('Nonconformity Score', fontdict={'size': 14})
    axes[0].set_ylabel('Count', fontdict={'size': 14})
    
    sns.histplot(cal_data[cal_data['HOSPITAL_EXPIRE_FLAG'] == 1]['mean_nonconformity_scores'], 
                 bins=20, kde=True, ax=axes[1], color='C1')
    axes[1].set_title('In-hospital death', fontdict={'size': 14, 'weight': 'bold'})
    axes[1].set_xlabel('Nonconformity Score', fontdict={'size': 14})
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_note_length_vs_nonconformity(cal_data, save_path='note_length_non_conform_flag.jpg'):
    """
    Plot scatter of note length vs mean nonconformity score.
    
    Args:
        cal_data: Calibration dataframe
        save_path: Path to save the plot
    """
    plt.figure(figsize=(7, 5), dpi=300)

    # Scatter plot for HOSPITAL_EXPIRE_FLAG = 0
    plt.scatter(
        cal_data[cal_data['HOSPITAL_EXPIRE_FLAG'] == 0]['mean_nonconformity_scores'],
        cal_data[cal_data['HOSPITAL_EXPIRE_FLAG'] == 0]['len_text_2'],
        alpha=0.6,
        label='In-hospital survived'
    )

    # Scatter plot for HOSPITAL_EXPIRE_FLAG = 1
    plt.scatter(
        cal_data[cal_data['HOSPITAL_EXPIRE_FLAG'] == 1]['mean_nonconformity_scores'],
        cal_data[cal_data['HOSPITAL_EXPIRE_FLAG'] == 1]['len_text_2'],
        alpha=0.6,
        label='In-hospital death'
    )

    plt.xlabel('Mean Non-conformity Score')
    plt.ylabel('Note Length')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_confidence_credibility(cp_data, save_path='Credibility vs. Confidence in GPT Predictions.jpg'):
    """
    Plot credibility vs confidence scatter plot.
    
    Args:
        cp_data: Conformal prediction results dataframe
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 5), dpi=300)
    sns.scatterplot(data=cp_data, x='confidence', y='credibility', hue='verdict', ax=ax)
    ax.legend(prop={'size': 14})
    plt.xlabel("Confidence", fontdict={'fontsize': 16})
    plt.ylabel("Credibility", fontdict={'fontsize': 16})
    plt.tight_layout()
    plt.savefig(save_path)


def plot_confidence_true_class_prob(cp_data, save_path='Confidence vs. True Class Probability in GPT Predictions.jpg'):
    """
    Plot confidence vs true class probability scatter plot.
    
    Args:
        cp_data: Conformal prediction results dataframe
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 5), dpi=300)
    sns.scatterplot(data=cp_data, x='confidence', y='true_class_prob', hue='verdict', alpha=0.25, ax=ax)
    ax.legend(prop={'size': 14})
    plt.xlabel("Confidence", fontdict={'fontsize': 16})
    plt.ylabel("True Class Probability", fontdict={'fontsize': 16})
    plt.savefig(save_path)


def prepare_test_dataframe(test_data):
    """
    Prepare test dataframe for conformal prediction.
    
    Args:
        test_data: Raw test dataframe
        
    Returns:
        Prepared dataframe with required columns
    """
    df = pd.DataFrame({'true_class_prob': test_data['true_class_prob']})
    df['true_class_prob'] = df['true_class_prob'].apply(np.mean)
    df[['no', 'yes', 'other']] = test_data['y_n_o_probs_all'].apply(
        lambda probs: pd.Series(np.mean(probs, axis=0))
    )
    df['class'] = test_data['HOSPITAL_EXPIRE_FLAG']
    df['predicted_class'] = test_data['final_prediction']
    return df