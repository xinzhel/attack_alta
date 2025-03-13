import numpy as np
from statsmodels.stats.proportion import proportions_ztest

def two_proportion_z_test(x_A, n_A, x_B, n_B, alpha=0.05):
    """
    Conducts a two-proportion z-test to check if there is a significant 
    difference between the success rates of two methods A and B.

    Parameters
    ----------
    x_A : int
        Number of successful attacks for method A.
    n_A : int
        Total number of samples tested by method A.
    x_B : int
        Number of successful attacks for method B.
    n_B : int
        Total number of samples tested by method B.
    alpha : float, optional
        Significance level, default is 0.05.

    Returns
    -------
    z_stat : float
        The computed z statistic.
    p_value : float
        The two-tailed p-value.
    is_significant : bool
        True if p_value < alpha, otherwise False.
    """
    # counts of successes for each group
    count = np.array([x_A, x_B])
    # sample sizes for each group
    nobs = np.array([n_A, n_B])
    
    z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
    is_significant = (p_value < alpha)
    
    return z_stat, p_value, is_significant


if __name__ == "__main__":
    n_A = 960
    n_B = n_A

    asr_A = 0
    asr_B = 0.02

    # Example usage:
    # Suppose method A was tested on 100 samples with 35 successes.
    x_A = n_A * asr_A
    
    # Suppose method B was tested on 120 samples with 60 successes.
    x_B = n_B * asr_B
    
    alpha = 0.05
    
    z_stat, p_val, significant = two_proportion_z_test(x_A, n_A, x_B, n_B, alpha)
    
    print(f"Z-statistic: {z_stat:.4f}")
    print(f"p-value: {p_val:.4f}")
    
    if significant:
        print(f"Reject the null hypothesis at alpha={alpha}. The difference is statistically significant.")
    else:
        print(f"Fail to reject the null hypothesis at alpha={alpha}. No significant difference.")
