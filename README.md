# LSTM-Autoencoder-for-Early-Failure-Detection-in-NASA-Turbofan-Engines-C-MAPSS-

# Problem Statement

Predictive maintenance is critical in aerospace systems.
This project applies an LSTM Autoencoder to detect anomalous behavior in turbofan engines using the NASA C-MAPSS dataset.

The goal is to identify early-stage engine degradation before failure occurs.

# Dataset

Dataset: NASA C-MAPSS (FD001 subset)

Source: Kaggle – behrad3d/nasa-cmaps

Engines: 100

Sensors: 21 sensor measurements

Operating settings: 3

We compute Remaining Useful Life (RUL):

𝑅𝑈𝐿=𝑚𝑎𝑥(𝑐𝑦𝑐𝑙𝑒)−𝑐𝑢𝑟𝑟𝑒𝑛𝑡(𝑐𝑦𝑐𝑙𝑒)RUL=max(cycle)−current(cycle)

Engines with RUL ≤ 30 are treated as "near failure" (anomaly condition).

# Data Preprocessing

Removed extra blank columns

MinMax Scaling

Sequence window size = 30 cycles

Input shape: (30 timesteps × 21 sensors)

🧠 Model Architecture
# Encoder:

1) LSTM(64)

2) LSTM(32)

# Decoder:

RepeatVector

1) LSTM(32)

2) LSTM(64)

3) TimeDistributed(Dense)

Loss: Mean Squared Error
Optimizer: Adam
Epochs: 10
Batch Size: 128

# Anomaly Detection Strategy

Reconstruction Error:
MSE=mean((X−X(reconstructed))**2)
	​
Threshold:
μ+3σ

Sequences above threshold are classified as anomalies.

# Results

Metric	Value
Accuracy	83%
ROC-AUC	0.41
F1 (Failure Class)	0.00

Confusion Matrix shows strong bias toward normal class.

# Analysis

The autoencoder reconstructs both normal and degraded sequences similarly, leading to poor anomaly separation.

ROC-AUC < 0.5 indicates the reconstruction error is not a strong discriminator for failure state.

This suggests:

Reconstruction-based anomaly detection may not align with RUL thresholding.
Failure progression may not be sharply separable via MSE alone.

# Conclusion

This project demonstrates an end-to-end deep learning pipeline for predictive maintenance.
