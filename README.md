# PBE-GovernanceAlgorithms

CUSUM drift detection and changepoint (PELT) detection for indicator time series.
Designed to run in Microsoft Fabric / Synapse notebooks using Spark.

Contents
- nb_07_signals.py — CUSUM + PELT detection; writes Delta tables  and .

Usage (Fabric)
- Upload  into your Fabric workspace and run it inside a Python notebook with Spark.
- Changepoint detection uses  (if missing, the script will skip PELT and still run CUSUM).

Configuration
- Top constants: , , , , 

License: Add your preferred license.
