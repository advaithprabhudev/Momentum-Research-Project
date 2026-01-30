import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_ingestion import fetch_data
from feature_generation import *
from label_generator import *

feat = features()

# Feature Time-Series
plt.figure(1)
feat.plot(subplots=True, figsize=(12, 8))
plt.tight_layout()
plt.close()

# Feature Distributions
feat.hist(bins=50, figsize=(12, 8))
plt.close()

# Correlation Heatmap

corr = feat.corr()

plt.figure(figsize = (8, 6))
plt.imshow(corr, vmin = -1, vmax = 1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation = 90)
plt.yticks(range(len(corr)), corr.columns)
plt.show()

#