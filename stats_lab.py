import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    
    # Generate samples
    data = np.random.normal(0, 1, n)
    
    # Plot histogram
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal(0,1) Distribution")
    plt.show()
    
    return data
    

def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
     # Generate samples
    data = np.random.uniform(0, 10, n)
    
    # Plot histogram
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10) Distribution")
    plt.show()
    
    return data



def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    # Generate samples
    data = np.random.binomial(1, 0.5, n)
    
    # Plot histogram
    plt.hist(data, bins=10)
    plt.xlabel("Value (0 or 1)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5) Distribution")
    plt.show()
    
    return data

normal_data = normal_histogram(1000)
uniform_data = uniform_histogram(1000)
bernoulli_data = bernoulli_histogram(1000)


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    # 1️⃣ Sample Mean Function
    n = len(data)
    total = 0
    
    for value in data:
        total += value
        
    mean = total / n
    return mean


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    n = len(data)
    
    # Step 1: Compute mean
    mean = sample_mean(data)
    
    # Step 2: Compute squared differences
    squared_diff_sum = 0
    for value in data:
        squared_diff_sum += (value - mean) ** 2
        
    # Step 3: Divide by (n - 1)
    variance = squared_diff_sum / (n - 1)
    
    return variance

data = [2, 4, 6, 8]

mean_value = sample_mean(data)
variance_value = sample_variance(data)

print("Sample Mean:", mean_value)
print("Sample Variance:", variance_value)

# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    # Step 1: Sort the data
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Minimum and Maximum
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    
    # Median
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    # Quartiles (Autograder Style)
    q1 = sorted_data[n // 4]
    q3 = sorted_data[(3 * n) // 4]
    
    return (minimum, maximum, median, q1, q3)

data = [5,1,3,2,4]

result = order_statistics(data)
print(result)



# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    n = len(x)
    
    # Step 1: Compute means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Step 2: Compute sum of products
    cov_sum = 0
    for i in range(n):
        cov_sum += (x[i] - mean_x) * (y[i] - mean_y)
    
    # Step 3: Divide by (n - 1)
    covariance = cov_sum / (n - 1)
    
    return covariance

x = [1, 2, 3]
y = [4, 5, 6]

print(sample_covariance(x, y))


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)
    
    return [
        [var_x, cov_xy],
        [cov_xy, var_y]
    ]


x = [1, 2, 3]
y = [4, 5, 6]

matrix = covariance_matrix(x, y)
print(matrix)

