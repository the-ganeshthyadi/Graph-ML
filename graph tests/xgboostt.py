import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Sample data (increased size for better results)
X = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [0.2, 0.4, 0.6],
    [0.5, 0.7, 0.9],
    [0.1, 0.3, 0.5],
    [0.3, 0.5, 0.7],
    [0.6, 0.8, 1.0],
    [0.4, 0.6, 0.8],
    [0.7, 0.9, 1.1]
])
y = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])

# Function to train XGBoost model and get feature importances
def xgboost_classifier(X, y, n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1):
    model = XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        min_child_weight=min_child_weight, 
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        random_state=0
    )
    model.fit(X, y)
    feature_importances = model.feature_importances_
    return model, feature_importances

# Train the model and get feature importances
model, importances = xgboost_classifier(X, y, n_estimators=150, learning_rate=0.05)

# Debugging: Print feature importances
print("Feature Importances:", importances)

# Feature names
feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]

# Plotting
def plot_feature_importances(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    
    # Debugging: Print indices and sorted importances
    print("Sorted Importances:", importances[indices])
    print("Feature Indices:", indices)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    colors = sns.color_palette("crest", as_cmap=True)
    
    # Create the bar plot
    plt.bar(range(len(importances)), importances[indices], align='center', color=colors(0.5))
    
    # Add labels and title
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.title('Feature Importances (XGBoost)', fontsize=16, fontweight='bold')
    
    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Beautify the plot
    sns.despine()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Plot the feature importances
plot_feature_importances(importances, feature_names)
