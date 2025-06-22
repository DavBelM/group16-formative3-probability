"""
Gradient Descent Implementation - Group 16
Manual implementation using SciPy for linear regression
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math

class GradientDescentLinearRegression:
    """
    Manual implementation of Gradient Descent for Linear Regression
    y = mx + b
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=100):
        """
        Initialize gradient descent parameters
        
        Args:
            learning_rate (float): Step size for parameter updates
            max_iterations (int): Maximum number of iterations
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.cost_history = []
        self.m_history = []
        self.b_history = []
        
    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate Mean Squared Error (MSE)
        MSE = (1/n) * Σ(y_true - y_pred)²
        
        Args:
            y_true (array): Actual values
            y_pred (array): Predicted values
            
        Returns:
            float: Mean squared error
        """
        n = len(y_true)
        mse = sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n
        return mse
    
    def predict(self, X, m, b):
        """
        Make predictions using linear equation y = mx + b
        
        Args:
            X (array): Input features
            m (float): Slope parameter
            b (float): Intercept parameter
            
        Returns:
            array: Predicted values
        """
        return [m * x + b for x in X]
    
    def compute_gradients(self, X, y, m, b):
        """
        Compute gradients for parameters m and b
        
        ∂J/∂m = -(2/n) * Σ(yi - ŷi) * xi
        ∂J/∂b = -(2/n) * Σ(yi - ŷi)
        
        Args:
            X (array): Input features
            y (array): Target values
            m (float): Current slope
            b (float): Current intercept
            
        Returns:
            tuple: (gradient_m, gradient_b)
        """
        n = len(X)
        y_pred = self.predict(X, m, b)
        
        # Calculate gradients manually
        gradient_m = 0
        gradient_b = 0
        
        for i in range(n):
            error = y[i] - y_pred[i]
            gradient_m += error * X[i]
            gradient_b += error
        
        gradient_m = -(2/n) * gradient_m
        gradient_b = -(2/n) * gradient_b
        
        return gradient_m, gradient_b
    
    def fit_manual(self, X, y, initial_m=0, initial_b=0, verbose=True):
        """
        Fit the model using manual gradient descent implementation
        
        Args:
            X (array): Input features
            y (array): Target values
            initial_m (float): Initial slope value
            initial_b (float): Initial intercept value
            verbose (bool): Print iteration details
            
        Returns:
            tuple: (final_m, final_b)
        """
        m = initial_m
        b = initial_b
        
        if verbose:
            print("="*80)
            print("MANUAL GRADIENT DESCENT IMPLEMENTATION")
            print("="*80)
            print(f"Initial parameters: m = {m}, b = {b}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Data points: {list(zip(X, y))}")
            print("\nIteration Details:")
            print("-" * 80)
        
        for iteration in range(self.max_iterations):
            # Make predictions
            y_pred = self.predict(X, m, b)
            
            # Calculate cost
            cost = self.mean_squared_error(y, y_pred)
            
            # Compute gradients
            grad_m, grad_b = self.compute_gradients(X, y, m, b)
            
            # Update parameters
            m_new = m - self.learning_rate * grad_m
            b_new = b - self.learning_rate * grad_b
            
            # Store history
            self.cost_history.append(cost)
            self.m_history.append(m)
            self.b_history.append(b)
            
            if verbose and iteration < 10:  # Show first 10 iterations in detail
                print(f"Iteration {iteration + 1}:")
                print(f"  Predictions: {[f'{pred:.4f}' for pred in y_pred]}")
                print(f"  Cost (MSE): {cost:.6f}")
                print(f"  Gradients: ∂J/∂m = {grad_m:.6f}, ∂J/∂b = {grad_b:.6f}")
                print(f"  Updated: m = {m:.6f} → {m_new:.6f}")
                print(f"           b = {b:.6f} → {b_new:.6f}")
                print()
            
            # Update parameters for next iteration
            m, b = m_new, b_new
            
            # Check for convergence
            if iteration > 0 and abs(self.cost_history[-2] - cost) < 1e-8:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
        
        # Store final values
        self.final_m = m
        self.final_b = b
        final_cost = self.mean_squared_error(y, self.predict(X, m, b))
        
        if verbose:
            print(f"\nFinal Results:")
            print(f"Final parameters: m = {m:.6f}, b = {b:.6f}")
            print(f"Final cost: {final_cost:.6f}")
            print(f"Final equation: y = {m:.6f}x + {b:.6f}")
        
        return m, b
    
    def fit_scipy(self, X, y, verbose=True):
        """
        Compare with SciPy optimization
        """
        def cost_function(params):
            m, b = params
            y_pred = self.predict(X, m, b)
            return self.mean_squared_error(y, y_pred)
        
        # Initial guess
        initial_guess = [0, 0]
        
        # Optimize using SciPy
        result = minimize(cost_function, initial_guess, method='BFGS')
        
        if verbose:
            print("\n" + "="*80)
            print("SCIPY OPTIMIZATION COMPARISON")
            print("="*80)
            print(f"SciPy result: m = {result.x[0]:.6f}, b = {result.x[1]:.6f}")
            print(f"SciPy cost: {result.fun:.6f}")
            print(f"Manual result: m = {self.final_m:.6f}, b = {self.final_b:.6f}")
            print(f"Difference in m: {abs(result.x[0] - self.final_m):.8f}")
            print(f"Difference in b: {abs(result.x[1] - self.final_b):.8f}")
        
        return result.x[0], result.x[1]

def assignment_example():
    """
    Solve the specific assignment problem
    Given points: (1, 3), (2, 5), (3, 7)
    Initial: m = 0, b = 0, learning_rate = 0.1
    """
    print("="*80)
    print("ASSIGNMENT SPECIFIC EXAMPLE")
    print("="*80)
    print("Given data points: (1, 3), (2, 5), (3, 7)")
    print("Initial parameters: m = 0, b = 0")
    print("Learning rate: 0.1")
    print()
    
    # Assignment data - CORRECTED from Part 3 requirements
    X = [1, 3]
    y = [3, 6]
    
    # Create model with assignment parameters
    model = GradientDescentLinearRegression(learning_rate=0.1, max_iterations=10)
    
    # Fit the model with correct initial parameters
    final_m, final_b = model.fit_manual(X, y, initial_m=-1, initial_b=1)
    
    # Make final predictions
    final_predictions = model.predict(X, final_m, final_b)
    
    print(f"\nFinal Predictions:")
    for i, (x_val, y_true, y_pred) in enumerate(zip(X, y, final_predictions)):
        print(f"Point {i+1}: x={x_val}, y_true={y_true}, y_pred={y_pred:.4f}, error={abs(y_true-y_pred):.4f}")
    
    return model

def manual_calculation_steps():
    """
    Show detailed manual calculations for the first few iterations
    This matches what students need to do by hand
    """
    print("\n" + "="*80)
    print("DETAILED MANUAL CALCULATIONS (For Hand Calculation)")
    print("="*80)
    
    # Given data - CORRECTED from Part 3 requirements
    X = [1, 3]
    y = [3, 6]
    m, b = -1, 1  # Corrected initial parameters
    learning_rate = 0.1
    n = len(X)
    
    print(f"Given: Data points = {list(zip(X, y))}")
    print(f"Initial: m = {m}, b = {b}")
    print(f"Learning rate α = {learning_rate}")
    print(f"Number of points n = {n}")
    print()
    
    # Show 3 iterations manually
    for iteration in range(3):
        print(f"ITERATION {iteration + 1}:")
        print("-" * 40)
        
        # Step 1: Calculate predictions
        print(f"Step 1: Calculate predictions ŷᵢ = m*xᵢ + b")
        y_pred = []
        for i in range(n):
            pred = m * X[i] + b
            y_pred.append(pred)
            print(f"  ŷ{i+1} = {m} * {X[i]} + {b} = {pred}")
        
        # Step 2: Calculate cost
        print(f"\nStep 2: Calculate Mean Squared Error")
        mse_sum = 0
        for i in range(n):
            error_sq = (y[i] - y_pred[i])**2
            mse_sum += error_sq
            print(f"  (y{i+1} - ŷ{i+1})² = ({y[i]} - {y_pred[i]})² = {error_sq}")
        
        mse = mse_sum / n
        print(f"  MSE = (1/{n}) * {mse_sum} = {mse}")
        
        # Step 3: Calculate gradients
        print(f"\nStep 3: Calculate gradients")
        grad_m_sum = 0
        grad_b_sum = 0
        
        for i in range(n):
            error = y[i] - y_pred[i]
            grad_m_sum += error * X[i]
            grad_b_sum += error
            print(f"  Error{i+1} = {y[i]} - {y_pred[i]} = {error}")
        
        grad_m = -(2/n) * grad_m_sum
        grad_b = -(2/n) * grad_b_sum
        
        print(f"  ∂J/∂m = -(2/{n}) * Σ(yᵢ - ŷᵢ)*xᵢ = -(2/{n}) * {grad_m_sum} = {grad_m}")
        print(f"  ∂J/∂b = -(2/{n}) * Σ(yᵢ - ŷᵢ) = -(2/{n}) * {grad_b_sum} = {grad_b}")
        
        # Step 4: Update parameters
        print(f"\nStep 4: Update parameters")
        m_new = m - learning_rate * grad_m
        b_new = b - learning_rate * grad_b
        
        print(f"  m_new = m - α * ∂J/∂m = {m} - {learning_rate} * {grad_m} = {m_new}")
        print(f"  b_new = b - α * ∂J/∂b = {b} - {learning_rate} * {grad_b} = {b_new}")
        
        print(f"\nResults after iteration {iteration + 1}:")
        print(f"  m = {m_new:.6f}, b = {b_new:.6f}, Cost = {mse:.6f}")
        print()
        
        # Update for next iteration
        m, b = m_new, b_new

def visualize_gradient_descent(model):
    """
    Create visualizations showing gradient descent progress
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cost function over iterations
    ax1.plot(model.cost_history, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost (MSE)')
    ax1.set_title('Cost Function Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter evolution
    iterations = range(len(model.m_history))
    ax2.plot(iterations, model.m_history, 'r-', linewidth=2, marker='s', label='m (slope)')
    ax2.plot(iterations, model.b_history, 'g-', linewidth=2, marker='^', label='b (intercept)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Data points and final line - CORRECTED data
    X = [1, 3]
    y = [3, 6]
    ax3.scatter(X, y, color='red', s=100, zorder=5, label='Data points')
    
    # Plot regression line
    x_line = np.linspace(0.5, 3.5, 100)
    y_line = model.final_m * x_line + model.final_b
    ax3.plot(x_line, y_line, 'b-', linewidth=2, label=f'y = {model.final_m:.3f}x + {model.final_b:.3f}')
    
    # Plot predictions
    final_predictions = model.predict(X, model.final_m, model.final_b)
    ax3.scatter(X, final_predictions, color='blue', s=50, marker='x', label='Predictions')
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Linear Regression Result')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error reduction
    errors = []
    for i in range(len(model.cost_history)):
        y_pred = model.predict(X, model.m_history[i], model.b_history[i])
        total_error = sum(abs(y[j] - y_pred[j]) for j in range(len(y)))
        errors.append(total_error)
    
    ax4.plot(errors, 'purple', linewidth=2, marker='d')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Total Absolute Error')
    ax4.set_title('Error Reduction Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Run assignment example
    model = assignment_example()
    
    # Show detailed manual calculations
    manual_calculation_steps()
    
    # Compare with SciPy - CORRECTED data
    X = [1, 3]
    y = [3, 6]
    model.fit_scipy(X, y)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_gradient_descent(model)
