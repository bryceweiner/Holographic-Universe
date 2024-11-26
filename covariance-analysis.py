import numpy as np
from typing import Dict, List

class CovarianceAnalysis:
    def __init__(self):
        # Observed data for covariance analysis
        self.observed_data = {
            'matter_density': [0.267, 0.298, 0.315],
            'h0': [73.2, 69.8, 67.36],
            'bao': [19.05, 18.92, 18.80],
            's8': [0.773, 0.781, 0.834]
        }

        # Predicted values for ΛCDM and HU models
        self.lcdm_predictions = {
            'matter_density': [0.315, 0.315, 0.315],
            'h0': [67.36, 67.36, 67.36],
            'bao': [20.10, 19.85, 19.60],
            's8': [0.834, 0.834, 0.834]
        }

        self.hu_predictions = {
            'matter_density': [0.298, 0.298, 0.298],
            'h0': [73.2, 73.2, 73.2],
            'bao': [18.80, 18.80, 18.80],
            's8': [0.781, 0.781, 0.781]
        }

    def calculate_covariance_with_observed(self, model_predictions: Dict[str, List[float]]) -> np.ndarray:
        """Calculate the covariance matrix between model predictions and observed data."""
        observed_matrix = np.array([self.observed_data[key] for key in self.observed_data])
        model_matrix = np.array([model_predictions[key] for key in model_predictions])
        return np.cov(observed_matrix, model_matrix)

    def interpret_covariance_matrix(self, covariance_matrix: np.ndarray, model_name: str):
        """Interpret the covariance matrix."""
        print(f"\nInterpretation of Covariance Matrix for {model_name}:")
        parameter_names = list(self.observed_data.keys())
        for i in range(len(parameter_names)):
            for j in range(len(parameter_names), len(parameter_names) * 2):
                cov_value = covariance_matrix[i, j]
                if cov_value > 0:
                    relation = "positively correlated"
                elif cov_value < 0:
                    relation = "negatively correlated"
                else:
                    relation = "uncorrelated"
                print(f"{parameter_names[i]} and {parameter_names[j - len(parameter_names)]} are {relation} (covariance = {cov_value:.4f})")

    def compare_models(self, lcdm_covariance: np.ndarray, hu_covariance: np.ndarray):
        """Compare the models based on their covariance with observed data."""
        parameter_names = list(self.observed_data.keys())
        lcdm_total_cov = np.sum(np.abs(lcdm_covariance))
        hu_total_cov = np.sum(np.abs(hu_covariance))

        print("\nModel Comparison:")
        print(f"Total absolute covariance for ΛCDM: {lcdm_total_cov:.4f}")
        print(f"Total absolute covariance for HU: {hu_total_cov:.4f}")

        for i, param in enumerate(parameter_names):
            lcdm_param_cov = np.sum(np.abs(lcdm_covariance[i, len(parameter_names):]))
            hu_param_cov = np.sum(np.abs(hu_covariance[i, len(parameter_names):]))
            print(f"\n{param.upper()} Comparison:")
            print(f"ΛCDM {param} covariance: {lcdm_param_cov:.4f}")
            print(f"HU {param} covariance: {hu_param_cov:.4f}")
            if lcdm_param_cov < hu_param_cov:
                print(f"Conclusion: ΛCDM model more closely matches the observed {param} data.")
            else:
                print(f"Conclusion: HU model more closely matches the observed {param} data.")

        if lcdm_total_cov < hu_total_cov:
            print("\nOverall Conclusion: ΛCDM model more closely matches the observed data.")
        else:
            print("\nOverall Conclusion: HU model more closely matches the observed data.")

    def print_covariance_matrices(self):
        """Print the covariance matrices and their interpretations."""
        # ΛCDM model covariance with observed data
        lcdm_covariance_matrix = self.calculate_covariance_with_observed(self.lcdm_predictions)
        print("ΛCDM Model Covariance with Observed Data:")
        print(lcdm_covariance_matrix)
        self.interpret_covariance_matrix(lcdm_covariance_matrix, "ΛCDM Model")

        # HU model covariance with observed data
        hu_covariance_matrix = self.calculate_covariance_with_observed(self.hu_predictions)
        print("\nHU Model Covariance with Observed Data:")
        print(hu_covariance_matrix)
        self.interpret_covariance_matrix(hu_covariance_matrix, "HU Model")

        # Compare models
        self.compare_models(lcdm_covariance_matrix, hu_covariance_matrix)

def main():
    covariance_analysis = CovarianceAnalysis()
    covariance_analysis.print_covariance_matrices()

if __name__ == "__main__":
    main() 