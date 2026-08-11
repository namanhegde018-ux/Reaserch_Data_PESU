import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

# -----------------------------
# File paths
# -----------------------------
input_csv = "mill.csv"
output_csv = "accepted_rejected.csv"

# -----------------------------
# Parameters
# -----------------------------
alpha = 0.99
dof = 3
threshold = chi2.ppf(alpha, dof)

# -----------------------------
# Read CSV
# -----------------------------
df = pd.read_csv(input_csv)

# Name of Mahalanobis distance column
distance_col = "mahalanobis_distance_squared"

# -----------------------------
# Classify
# -----------------------------
df["Status"] = df[distance_col].apply(
    lambda x: "Accepted" if x <= threshold else "Rejected"
)

# Save CSV
df.to_csv(output_csv, index=False)

# -----------------------------
# Separate Accepted & Rejected
# -----------------------------
accepted = df[df["Status"] == "Accepted"]
rejected = df[df["Status"] == "Rejected"]

# X-axis = Observation Number
accepted_x = accepted.index + 1
accepted_y = accepted[distance_col]

rejected_x = rejected.index + 1
rejected_y = rejected[distance_col]

# Total observations
total_obs = len(df)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(13, 6))

# Accepted
if not accepted.empty:
    markerline, stemlines, baseline = plt.stem(
        accepted_x,
        accepted_y,
        linefmt='g-',
        markerfmt='go',
        basefmt=' ',
        label='Accepted'
    )
    plt.setp(stemlines, linewidth=0.7)
    plt.setp(markerline, markersize=4)

# Rejected
if not rejected.empty:
    markerline, stemlines, baseline = plt.stem(
        rejected_x,
        rejected_y,
        linefmt='r-',
        markerfmt='rx',
        basefmt=' ',
        label='Rejected'
    )
    plt.setp(stemlines, linewidth=0.7)
    plt.setp(markerline, markersize=6)

# Chi-Square Threshold
plt.axhline(
    y=threshold,
    color='blue',
    linestyle='--',
    linewidth=2,
    label=f'χ² Threshold (α={alpha}, df={dof}) = {threshold:.4f}'
)

# Axes
plt.xlim(1, max(total_obs, 170))
plt.ylim(0, 170)

plt.xlabel("Observation Number", fontsize=12)
plt.ylabel("Mahalanobis Distance Squared (D²)", fontsize=12)

plt.title(
    "Classification of Observations Using Mahalanobis Distance Squared (D²)\n"
    "Based on Chi-Square Distribution",
    fontsize=14,
    fontweight='bold'
)

# Summary Box
summary = (
    f"Total Observations : {total_obs}\n"
    f"Accepted : {len(accepted)}\n"
    f"Rejected : {len(rejected)}"
)

plt.text(
    0.02,
    0.97,
    summary,
    transform=plt.gca().transAxes,
    fontsize=10,
    va='top',
    bbox=dict(facecolor='white',
              edgecolor='black',
              boxstyle='round')
)

# Legend
plt.legend(loc='upper right')
plt.tight_layout()

# Save Figure
plt.savefig("Mahalanobis_Distance_Classification.png", dpi=300)

plt.show()

print(f"Chi-Square Threshold (99%): {threshold:.4f}")
print(f"Accepted: {len(accepted)}")
print(f"Rejected: {len(rejected)}")
print(f"Results saved to: {output_csv}")