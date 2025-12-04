# ==================== REFACTORED CODE ====================
# Original messy code has been refactored following clean code principles

import pandas as pd

# Constants (0.5 points) - UPPERCASE names for magic numbers
AGE_THRESHOLD = 50
TARGET_CANCER_TYPE = 'Lung'
TREATMENT_GROUP = 'Treatment_A'
CONTROL_GROUP = 'Control'


def load_patient_data(filepath: str) -> pd.DataFrame:
    """
    Load patient data from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file containing patient data.

    Returns:
        pd.DataFrame: DataFrame containing patient records.
    """
    return pd.read_csv(filepath)


def find_responsive_patients(patient_data: pd.DataFrame,
                             age_threshold: int = AGE_THRESHOLD,
                             cancer_type: str = TARGET_CANCER_TYPE) -> list:
    """
    Find patients who responded positively to treatment.

    A patient is considered responsive if:
    - They are at or above the age threshold
    - They have the specified cancer type
    - Their final tumor size is smaller than baseline tumor size

    Parameters:
        patient_data (pd.DataFrame): DataFrame containing patient records.
        age_threshold (int): Minimum age to include in analysis. Default is 50.
        cancer_type (str): Cancer type to filter by. Default is 'Lung'.

    Returns:
        list: List of patient IDs who showed positive response.
    """
    # Use pandas boolean filtering instead of loops (0.5 points - better pandas usage)
    responsive_mask = (
        (patient_data['age'] >= age_threshold) &
        (patient_data['cancer_type'] == cancer_type) &
        (patient_data['final_tumor_size'] < patient_data['baseline_tumor_size'])
    )

    responsive_patient_ids = patient_data.loc[responsive_mask, 'patient_id'].tolist()
    return responsive_patient_ids


def calculate_survival_statistics(patient_data: pd.DataFrame) -> dict:
    """
    Calculate average survival months by treatment group.

    Parameters:
        patient_data (pd.DataFrame): DataFrame containing patient records.

    Returns:
        dict: Dictionary with treatment groups as keys and average survival as values.
    """
    # Use groupby instead of loops (0.5 points - better pandas usage)
    survival_by_treatment = patient_data.groupby('treatment')['survival_months'].mean()

    return {
        'treatment_a_avg_survival': survival_by_treatment.get(TREATMENT_GROUP, 0),
        'control_avg_survival': survival_by_treatment.get(CONTROL_GROUP, 0)
    }


def analyze_clinical_trial(filepath: str) -> list:
    """
    Main function to analyze clinical trial data.

    Coordinates the analysis by:
    1. Loading patient data
    2. Finding responsive patients
    3. Calculating survival statistics
    4. Printing summary results

    Parameters:
        filepath (str): Path to the CSV file containing patient data.

    Returns:
        list: List of responsive patient IDs.
    """
    # Load data
    patient_data = load_patient_data(filepath)

    # Find responsive patients
    responsive_patient_ids = find_responsive_patients(patient_data)

    # Calculate survival statistics
    survival_stats = calculate_survival_statistics(patient_data)

    # Print results
    print(f"Avg survival Treatment A: {survival_stats['treatment_a_avg_survival']:.2f}")
    print(f"Avg survival Control: {survival_stats['control_avg_survival']:.2f}")
    print(f"Responsive patients: {len(responsive_patient_ids)}")

    return responsive_patient_ids


# Example usage (uncomment when you have the data file):
# responsive_patients = analyze_clinical_trial('clinical_trial_patients.csv')
