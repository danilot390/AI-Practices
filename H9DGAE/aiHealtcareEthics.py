import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1. Simulate Dataset
data = {
    'PatientId' :['P001','P002','P003','P004','P005','P010'],
    'Age' : [65, 50, 80, 45, 30, 80],
    'UrgencyScore' : [9, 7, 10, 6, 5, 10],
    'SurvivalProbability' : [0.8, 0.6, 0.3, 0.7, 0.9, 1]
}
df = pd.DataFrame(data)

# Step 2: Calculate Allocation Scores
# Formula: Score = (0.5 * Urgency) + (0.3 * Survival Probability) + (0.2 * Age Factor)
scaler = MinMaxScaler()
df['AgeFactor'] = scaler.fit_transform(df[['Age']]) # Normalize age
df['UrgencyScore'] = scaler.fit_transform(df[['UrgencyScore']])

df['AllocationScore'] = (
0.5 * df['UrgencyScore'] +
0.3 * df['SurvivalProbability'] +
0.2 * df['AgeFactor'] 
)

# Step 3: Sort patients by Allocation Score
df = df.sort_values(by='AllocationScore', ascending=False)

# Step 4: Generate Explanability Report
def generate_explanability_report(row):
    return(
        f"Patient {row['PatientId']}: \n"
        f"  Urgency Score Contribution: {0.5 * row['UrgencyScore']:.2f}\n"
        f"  Survival Probability Contribution: {0.3 * row['SurvivalProbability']:.2f}\n"
        f"  Age Factor Contribution: {0.2 * row['AgeFactor']:.2f}\n"
        f"  Total Allocation Score: {row['AllocationScore']:.2f}\n"
    )

df['Explainability'] = df.apply(generate_explanability_report, axis=1)

# Output Results
print('Sorted Patients bu Allocation Score: \n', df[['PatientId', 'AllocationScore']])
print('\n Explainability Reports:')
for report in df['Explainability']:
    print(report)

# Ethical Safeguards
print('\n Ethical Safeguards: ')
print('- Ensure data is anonymized and consented in real-world applications')
print('- Regular audits for bias and fairness.')
print('- Incllude a manual override for healthcare professionals.')