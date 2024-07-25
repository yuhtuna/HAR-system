import time
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import butter, filtfilt, welch

# Load the .mat file
print("Loading data...")
mat = scipy.io.loadmat('BufferedHumanActivity.mat')

# Extract data from the loaded mat file
print("Extracting data from .mat file...")
atx = mat['atx']  # 44 by 7776 matrix
actid = mat['actid'].squeeze() - 1  # Zero-indexed
actnames = [name[0] for name in mat['actnames'].squeeze()]  # Converting actnames to a list of strings
fs = mat['fs'].squeeze()  # Sample rate

# Define the high-pass filter
def highpass_filter(data, cutoff, fs):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

print("Applying high-pass filter...")
start_time = time.time()
atx_filtered = highpass_filter(atx, 0.7, fs)
print(f"Filtering completed in {time.time() - start_time:.2f} seconds")

# Define feature extraction functions
def extract_mean_features(data):
    return np.mean(data, axis=0)

def extract_RMS_features(data):
    return np.sqrt(np.mean(data**2, axis=0))

def extract_std_features(data):
    return np.std(data, axis=0)

def extract_median_features(data):
    return np.median(data, axis=0)

def extract_variance_features(data):
    return np.var(data, axis=0)

def extract_skewness_features(data):
    return skew(data, axis=0)

def extract_kurtosis_features(data):
    return kurtosis(data, axis=0)

# Add additional features
def extract_autocorrelation_features(data, lag=1):
    autocorr = np.correlate(data, data, mode='full')
    return autocorr[autocorr.size // 2 + lag]

def extract_mean_crossing_rate(data):
    return np.mean(np.diff(data > np.mean(data)) != 0)
# Extract features
print("Extracting features...")
start_time = time.time()
meanFeatures = extract_mean_features(atx_filtered)
rmsFeatures = extract_RMS_features(atx_filtered)
stdFeatures = extract_std_features(atx_filtered)
medianFeatures = extract_median_features(atx_filtered)
varianceFeatures = extract_variance_features(atx_filtered)
skewnessFeatures = extract_skewness_features(atx_filtered)
kurtosisFeatures = extract_kurtosis_features(atx_filtered)

autocorrelationFeatures = np.array([extract_autocorrelation_features(signal, lag=1) for signal in atx_filtered.T])
meanCrossingRateFeatures = np.array([extract_mean_crossing_rate(signal) for signal in atx_filtered.T])

print("Creating feature table...")
featureTable = pd.DataFrame({
    'MeanFeature': meanFeatures,
    'RMSFeature': rmsFeatures,
    'StdFeature': stdFeatures,
    'MedianFeature': medianFeatures,
    'VarianceFeature': varianceFeatures,
    'SkewnessFeature': skewnessFeatures,
    'KurtosisFeature': kurtosisFeatures,
    'AutocorrelationFeature': autocorrelationFeatures,
    'MeanCrossingRateFeature': meanCrossingRateFeatures,
    'ActivityID': actid
})

# Extract predictors and response
predictors = featureTable.drop(columns=['ActivityID'])
response = featureTable['ActivityID']

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

# Partition the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(predictors, response, test_size=0.2, random_state=2)

# Hyperparameter tuning using Grid Search
print("Performing grid search for hyperparameter tuning...")
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Train the classifier with the best parameters
print("Training SVM model...")
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict on test data
print("Predicting on test data...")
y_pred = best_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"The classification accuracy on the test partition is {accuracy*100:.2f}%")

# Plot confusion matrix
print("Plotting confusion matrix...")
unique_labels = np.unique(response)
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(actnames))
plt.xticks(tick_marks, actnames, rotation=45)
plt.yticks(tick_marks, actnames)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
