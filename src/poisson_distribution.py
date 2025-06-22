"""
Poisson Distribution Implementation - Group 16
Implementation without using external statistical libraries
"""

import math
import matplotlib.pyplot as plt
import numpy as np

class PoissonDistribution:
    """
    Custom implementation of Poisson Distribution
    Formula: P(X = k) = (λ^k * e^(-λ)) / k!
    """
    
    def __init__(self, lambda_param):
        """
        Initialize Poisson distribution with lambda parameter
        
        Args:
            lambda_param (float): Rate parameter (λ) - average number of events
        """
        if lambda_param <= 0:
            raise ValueError("Lambda parameter must be positive")
        self.lambda_param = lambda_param
    
    def factorial(self, n):
        """Calculate factorial without using math.factorial"""
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    def pmf(self, k):
        """
        Probability Mass Function for Poisson Distribution
        P(X = k) = (λ^k * e^(-λ)) / k!
        
        Args:
            k (int): Number of events
            
        Returns:
            float: Probability of exactly k events
        """
        if k < 0:
            return 0
        
        # Calculate each component of the formula
        lambda_power_k = self.lambda_param ** k
        e_negative_lambda = math.exp(-self.lambda_param)
        k_factorial = self.factorial(k)
        
        probability = (lambda_power_k * e_negative_lambda) / k_factorial
        return probability
    
    def cdf(self, k):
        """
        Cumulative Distribution Function
        P(X ≤ k) = Σ(i=0 to k) P(X = i)
        
        Args:
            k (int): Upper limit
            
        Returns:
            float: Cumulative probability up to k events
        """
        cumulative_prob = 0
        for i in range(k + 1):
            cumulative_prob += self.pmf(i)
        return cumulative_prob
    
    def mean(self):
        """Mean of Poisson distribution = λ"""
        return self.lambda_param
    
    def variance(self):
        """Variance of Poisson distribution = λ"""
        return self.lambda_param
    
    def standard_deviation(self):
        """Standard deviation = √λ"""
        return math.sqrt(self.lambda_param)

def real_world_example():
    """
    Real-world example: Number of customer calls to a call center per hour
    Average: 8 calls per hour (λ = 8)
    """
    print("="*60)
    print("REAL-WORLD EXAMPLE: Call Center Analysis")
    print("="*60)
    print("Scenario: A call center receives an average of 8 calls per hour")
    print("Question: What's the probability of receiving exactly 5, 10, or 12 calls in an hour?")
    print()
    
    # Initialize Poisson distribution with λ = 8
    call_center = PoissonDistribution(lambda_param=8)
    
    # Calculate probabilities for different scenarios
    scenarios = [5, 8, 10, 12]
    
    print("Results:")
    print("-" * 40)
    for calls in scenarios:
        prob = call_center.pmf(calls)
        print(f"P(X = {calls} calls) = {prob:.4f} ({prob*100:.2f}%)")
    
    print(f"\nDistribution Statistics:")
    print(f"Mean (Expected calls): {call_center.mean()}")
    print(f"Variance: {call_center.variance()}")
    print(f"Standard Deviation: {call_center.standard_deviation():.2f}")
    
    return call_center

def visualize_poisson_distribution(poisson_dist, max_k=20):
    """
    Create visualization of the Poisson distribution
    
    Args:
        poisson_dist: PoissonDistribution object
        max_k (int): Maximum value of k to plot
    """
    # Generate data points
    k_values = list(range(0, max_k + 1))
    probabilities = [poisson_dist.pmf(k) for k in k_values]
    cumulative_probs = [poisson_dist.cdf(k) for k in k_values]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot PMF
    ax1.bar(k_values, probabilities, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Number of Events (k)')
    ax1.set_ylabel('Probability P(X = k)')
    ax1.set_title(f'Poisson Distribution PMF (λ = {poisson_dist.lambda_param})')
    ax1.grid(True, alpha=0.3)
    
    # Add mean line
    mean_val = poisson_dist.mean()
    ax1.axvline(x=mean_val, color='red', linestyle='--', 
                label=f'Mean = {mean_val}', linewidth=2)
    ax1.legend()
    
    # Plot CDF
    ax2.step(k_values, cumulative_probs, where='post', color='green', linewidth=2)
    ax2.fill_between(k_values, cumulative_probs, step='post', alpha=0.3, color='green')
    ax2.set_xlabel('Number of Events (k)')
    ax2.set_ylabel('Cumulative Probability P(X ≤ k)')
    ax2.set_title(f'Poisson Distribution CDF (λ = {poisson_dist.lambda_param})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def compare_different_lambdas():
    """Compare Poisson distributions with different lambda values"""
    lambdas = [2, 5, 8, 12]
    colors = ['red', 'blue', 'green', 'orange']
    
    plt.figure(figsize=(12, 8))
    
    for i, lambda_val in enumerate(lambdas):
        poisson_dist = PoissonDistribution(lambda_val)
        k_values = list(range(0, 21))
        probabilities = [poisson_dist.pmf(k) for k in k_values]
        
        plt.plot(k_values, probabilities, 'o-', color=colors[i], 
                label=f'λ = {lambda_val}', markersize=4, linewidth=2)
    
    plt.xlabel('Number of Events (k)')
    plt.ylabel('Probability P(X = k)')
    plt.title('Comparison of Poisson Distributions with Different λ Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Run the real-world example
    call_center_dist = real_world_example()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_poisson_distribution(call_center_dist)
    
    print("Comparing different lambda values...")
    compare_different_lambdas()
