"""
Bayesian Inference Implementation - Group 16
Real-world application: Medical Diagnosis using Bayes' Theorem
"""

import numpy as np
import matplotlib.pyplot as plt

class BayesianInference:
    """
    Implementation of Bayes' Theorem for real-world applications
    P(A|B) = P(B|A) * P(A) / P(B)
    """
    
    def __init__(self):
        self.history = []  # Store updates for visualization
    
    def bayes_theorem(self, prior, likelihood, evidence):
        """
        Calculate posterior probability using Bayes' theorem
        
        Args:
            prior (float): P(A) - Prior probability
            likelihood (float): P(B|A) - Likelihood
            evidence (float): P(B) - Evidence/Marginal probability
            
        Returns:
            float: P(A|B) - Posterior probability
        """
        if evidence == 0:
            raise ValueError("Evidence probability cannot be zero")
        
        posterior = (likelihood * prior) / evidence
        return posterior
    
    def calculate_evidence(self, likelihood_positive, prior_positive, 
                          likelihood_negative, prior_negative):
        """
        Calculate evidence P(B) using law of total probability
        P(B) = P(B|A) * P(A) + P(B|¬A) * P(¬A)
        """
        evidence = (likelihood_positive * prior_positive + 
                   likelihood_negative * prior_negative)
        return evidence

class MedicalDiagnosis(BayesianInference):
    """
    Medical diagnosis using Bayesian inference
    Scenario: COVID-19 rapid test accuracy
    """
    
    def __init__(self):
        super().__init__()
        # Initialize with realistic COVID-19 test parameters
        self.disease_prevalence = 0.05  # 5% of population has COVID
        self.test_sensitivity = 0.95    # 95% true positive rate
        self.test_specificity = 0.98    # 98% true negative rate
        
        # Calculate derived probabilities
        self.prior_healthy = 1 - self.disease_prevalence
        self.false_positive_rate = 1 - self.test_specificity
        
    def diagnose_single_test(self, test_result="positive"):
        """
        Calculate probability of having disease given test result
        
        Args:
            test_result (str): "positive" or "negative"
            
        Returns:
            dict: Contains all probability calculations
        """
        print("="*60)
        print("MEDICAL DIAGNOSIS: COVID-19 Rapid Test Analysis")
        print("="*60)
        print("Scenario Parameters:")
        print(f"• Disease prevalence in population: {self.disease_prevalence*100:.1f}%")
        print(f"• Test sensitivity (true positive rate): {self.test_sensitivity*100:.1f}%")
        print(f"• Test specificity (true negative rate): {self.test_specificity*100:.1f}%")
        print()
        
        if test_result == "positive":
            # P(Disease | Positive Test)
            likelihood = self.test_sensitivity  # P(Positive | Disease)
            prior = self.disease_prevalence     # P(Disease)
            
            # Calculate evidence using law of total probability
            evidence = self.calculate_evidence(
                likelihood_positive=self.test_sensitivity,
                prior_positive=self.disease_prevalence,
                likelihood_negative=self.false_positive_rate,
                prior_negative=self.prior_healthy
            )
            
            posterior = self.bayes_theorem(prior, likelihood, evidence)
            
            print("POSITIVE TEST RESULT ANALYSIS:")
            print("-" * 40)
            print(f"Prior P(Disease) = {prior:.4f}")
            print(f"Likelihood P(Positive|Disease) = {likelihood:.4f}")
            print(f"Evidence P(Positive) = {evidence:.4f}")
            print(f"Posterior P(Disease|Positive) = {posterior:.4f}")
            print(f"\nResult: {posterior*100:.2f}% chance of having COVID-19")
            
        else:  # negative test
            # P(Disease | Negative Test)
            likelihood = 1 - self.test_sensitivity  # P(Negative | Disease)
            prior = self.disease_prevalence          # P(Disease)
            
            # Calculate evidence
            evidence = self.calculate_evidence(
                likelihood_positive=1 - self.test_sensitivity,
                prior_positive=self.disease_prevalence,
                likelihood_negative=self.test_specificity,
                prior_negative=self.prior_healthy
            )
            
            posterior = self.bayes_theorem(prior, likelihood, evidence)
            
            print("NEGATIVE TEST RESULT ANALYSIS:")
            print("-" * 40)
            print(f"Prior P(Disease) = {prior:.4f}")
            print(f"Likelihood P(Negative|Disease) = {likelihood:.4f}")
            print(f"Evidence P(Negative) = {evidence:.4f}")
            print(f"Posterior P(Disease|Negative) = {posterior:.4f}")
            print(f"\nResult: {posterior*100:.4f}% chance of having COVID-19")
        
        results = {
            'test_result': test_result,
            'prior': prior,
            'likelihood': likelihood,
            'evidence': evidence,
            'posterior': posterior
        }
        
        self.history.append(results)
        return results
    
    def sequential_testing(self, test_results):
        """
        Demonstrate sequential Bayesian updating with multiple tests
        
        Args:
            test_results (list): List of test results ["positive", "negative", ...]
        """
        print("\n" + "="*60)
        print("SEQUENTIAL BAYESIAN UPDATING")
        print("="*60)
        print("Demonstrating how multiple tests update our belief about disease status")
        print()
        
        current_prior = self.disease_prevalence
        priors = [current_prior]
        posteriors = []
        
        for i, test_result in enumerate(test_results):
            print(f"Test {i+1}: {test_result.upper()} result")
            print("-" * 30)
            
            if test_result == "positive":
                likelihood = self.test_sensitivity
                evidence = self.calculate_evidence(
                    self.test_sensitivity, current_prior,
                    self.false_positive_rate, 1 - current_prior
                )
            else:
                likelihood = 1 - self.test_sensitivity
                evidence = self.calculate_evidence(
                    1 - self.test_sensitivity, current_prior,
                    self.test_specificity, 1 - current_prior
                )
            
            posterior = self.bayes_theorem(current_prior, likelihood, evidence)
            
            print(f"Prior: {current_prior:.4f}")
            print(f"Likelihood: {likelihood:.4f}")
            print(f"Evidence: {evidence:.4f}")
            print(f"Posterior: {posterior:.4f}")
            print(f"Probability of disease: {posterior*100:.2f}%")
            print()
            
            posteriors.append(posterior)
            current_prior = posterior  # Today's posterior becomes tomorrow's prior
            priors.append(current_prior)
        
        return priors, posteriors
    
    def visualize_sequential_updates(self, test_results):
        """Visualize how probabilities change with each test"""
        priors, posteriors = self.sequential_testing(test_results)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot probability updates
        test_numbers = range(len(test_results) + 1)
        ax1.plot(test_numbers, priors, 'o-', linewidth=2, markersize=8, 
                label='Prior Probability', color='blue')
        ax1.plot(range(1, len(posteriors) + 1), posteriors, 's-', 
                linewidth=2, markersize=8, label='Posterior Probability', color='red')
        
        ax1.set_xlabel('Test Number')
        ax1.set_ylabel('Probability of Disease')
        ax1.set_title('Bayesian Updates: Prior vs Posterior Probabilities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add test result annotations
        for i, result in enumerate(test_results):
            ax1.annotate(f'{result}', 
                        xy=(i+1, posteriors[i]), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.7))
        
        # Plot test characteristics
        characteristics = ['Sensitivity', 'Specificity', 'Prevalence']
        values = [self.test_sensitivity, self.test_specificity, self.disease_prevalence]
        colors = ['green', 'orange', 'purple']
        
        bars = ax2.bar(characteristics, values, color=colors, alpha=0.7)
        ax2.set_ylabel('Probability')
        ax2.set_title('Test Characteristics')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig

def spam_filter_example():
    """
    Alternative example: Email spam filtering using Bayesian inference
    """
    print("\n" + "="*60)
    print("ALTERNATIVE EXAMPLE: Email Spam Filtering")
    print("="*60)
    
    # Spam filter parameters
    spam_rate = 0.3  # 30% of emails are spam
    word_in_spam = 0.8  # 80% of spam emails contain the word "FREE"
    word_in_ham = 0.05  # 5% of legitimate emails contain "FREE"
    
    bayesian = BayesianInference()
    
    # Calculate evidence
    evidence = bayesian.calculate_evidence(
        likelihood_positive=word_in_spam,
        prior_positive=spam_rate,
        likelihood_negative=word_in_ham,
        prior_negative=1 - spam_rate
    )
    
    # Calculate posterior
    posterior = bayesian.bayes_theorem(spam_rate, word_in_spam, evidence)
    
    print("Email contains the word 'FREE'")
    print("-" * 30)
    print(f"Prior P(Spam) = {spam_rate:.2f}")
    print(f"Likelihood P('FREE'|Spam) = {word_in_spam:.2f}")
    print(f"Evidence P('FREE') = {evidence:.4f}")
    print(f"Posterior P(Spam|'FREE') = {posterior:.4f}")
    print(f"\nConclusion: {posterior*100:.1f}% chance this email is spam")

if __name__ == "__main__":
    # Medical diagnosis example
    diagnosis = MedicalDiagnosis()
    
    # Single test analysis
    diagnosis.diagnose_single_test("positive")
    print("\n" + "="*60)
    diagnosis.diagnose_single_test("negative")
    
    # Sequential testing example
    test_sequence = ["positive", "positive", "negative"]
    diagnosis.visualize_sequential_updates(test_sequence)
    
    # Alternative example
    spam_filter_example()
