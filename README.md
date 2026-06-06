# IBM Stock Data Mining

End-to-end data mining pipeline on 5 years of IBM stock data, covering time series decomposition, segment clustering, motif discovery, and anomaly detection.

---

The pipeline starts by downloading OHLCV data via `yfinance` and engineering a set of technical features — log returns, daily returns, SMA/EMA (20 and 50-day), Bollinger Bands, rolling annualized volatility, momentum, rate of change, and high-low range. The price series is then decomposed into trend, seasonal, and residual components using a manual STL-style approach built on a Savitzky-Golay smoother, with a 252-day period for the seasonal extraction.

Segmentation splits the full series into fixed 20-day windows. Each segment is summarized into a feature vector (mean + std of the six core features), scaled, and fed into both K-Means and Agglomerative (Ward linkage) clustering. Optimal K is selected automatically from the Silhouette score, with Elbow and Davies-Bouldin plots generated alongside for reference. Cluster labels are mapped back onto the original price chart so each market regime is visible in context.

Motif discovery runs DTW pairwise on a capped sample of segments within each cluster to find the two most shape-similar windows — giving a sense of the recurring price pattern that defines each regime. The DTW distance matrix over the full segment sample is also computed and plotted as a heatmap.

Anomaly detection combines three methods: Z-score on log returns (threshold ±3), IQR on rolling volatility, and Isolation Forest on a three-feature matrix. Days flagged by two or more methods are marked as strong anomalies and printed with their date, return, volatility, and combined score.

Everything outputs as 12 dark-themed PNGs saved to the working directory — price and indicator dashboard, STL decomposition, DTW heatmap, clustering evaluation metrics, cluster labels on price, segment shape profiles, PCA projection, dendrogram, motif pairs, anomaly overlay, return distribution with Q-Q plot, and a cluster feature heatmap.

---

## Setup

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn scipy
```

Run all cells in `DataMiningProject.ipynb`. Stock data downloads automatically on first run. To change the ticker, lookback period, or segment window, edit the config at the top of the notebook:

```python
stock       = "IBM"  # any yfinance ticker
years       = 5
window      = 20     # segment size in trading days
sample_size = 60     # segments used for DTW matrix (runtime scales quadratically)
```
