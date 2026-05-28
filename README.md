<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Comparative Analysis of Dimensionality Reduction Techniques for GMM Clustering in Human Activity Recognition</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --color-bg: #0f172a;
      --color-card: #1e293b;
      --color-card-hover: #334155;
      --color-text: #f1f5f9;
      --color-text-muted: #94a3b8;
      --color-primary: #3b82f6;
      --color-primary-hover: #2563eb;
      --color-secondary: #8b5cf6;
      --color-accent: #06b6d4;
      --color-success: #10b981;
      --color-warning: #f59e0b;
      --color-danger: #ef4444;
      --color-border: #334155;
      --color-code-bg: #0b1220;
      --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
      --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
      --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
      --radius-sm: 0.375rem;
      --radius-md: 0.5rem;
      --radius-lg: 0.75rem;
      --radius-xl: 1rem;
      --transition: all 0.2s ease-in-out;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--color-bg);
      color: var(--color-text);
      line-height: 1.7;
      padding: 2rem 1rem;
    }

    @media (min-width: 768px) {
      body {
        padding: 3rem 2rem;
      }
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
    }

    /* Header Styles */
    header {
      text-align: center;
      padding: 2.5rem 1.5rem;
      margin-bottom: 2rem;
      background: linear-gradient(135deg, var(--color-card), var(--color-card-hover));
      border-radius: var(--radius-xl);
      border: 1px solid var(--color-border);
      box-shadow: var(--shadow-lg);
    }

    h1 {
      font-size: 1.75rem;
      font-weight: 700;
      margin-bottom: 1rem;
      background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.3;
    }

    @media (min-width: 768px) {
      h1 {
        font-size: 2.25rem;
      }
    }

    .subtitle {
      color: var(--color-text-muted);
      font-size: 1.1rem;
      max-width: 800px;
      margin: 0 auto 1.5rem;
    }

    /* Content Sections */
    section {
      background: var(--color-card);
      border-radius: var(--radius-lg);
      padding: 1.75rem;
      margin-bottom: 1.5rem;
      border: 1px solid var(--color-border);
      box-shadow: var(--shadow-md);
      transition: var(--transition);
    }

    section:hover {
      border-color: var(--color-primary);
      transform: translateY(-2px);
    }

    h2 {
      font-size: 1.5rem;
      font-weight: 600;
      margin: 0 0 1.25rem;
      padding-bottom: 0.75rem;
      border-bottom: 2px solid var(--color-border);
      color: var(--color-text);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    h2::before {
      content: "";
      display: inline-block;
      width: 4px;
      height: 1.25rem;
      background: linear-gradient(to bottom, var(--color-primary), var(--color-accent));
      border-radius: 2px;
    }

    h3 {
      font-size: 1.25rem;
      font-weight: 600;
      margin: 1.5rem 0 0.75rem;
      color: var(--color-primary);
    }

    p {
      margin-bottom: 1rem;
      color: var(--color-text-muted);
    }

    a {
      color: var(--color-primary);
      text-decoration: none;
      font-weight: 500;
      transition: var(--transition);
    }

    a:hover {
      color: var(--color-primary-hover);
      text-decoration: underline;
    }

    /* Dataset Info Card */
    .dataset-info {
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
      border-left: 4px solid var(--color-primary);
      padding: 1.25rem;
      border-radius: 0 var(--radius-md) var(--radius-md) 0;
      margin: 1rem 0;
    }

    .dataset-info strong {
      color: var(--color-text);
    }

    /* Code Blocks */
    pre {
      background: var(--color-code-bg);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: 1.25rem;
      overflow-x: auto;
      margin: 1rem 0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9rem;
      line-height: 1.6;
    }

    code {
      font-family: 'JetBrains Mono', monospace;
      background: rgba(59, 130, 246, 0.15);
      padding: 0.2rem 0.4rem;
      border-radius: var(--radius-sm);
      font-size: 0.95em;
      color: var(--color-accent);
    }

    pre code {
      background: none;
      padding: 0;
      color: inherit;
    }

    /* Directory Tree */
    .tree {
      font-family: 'JetBrains Mono', monospace;
      background: var(--color-code-bg);
      padding: 1.25rem;
      border-radius: var(--radius-md);
      border: 1px solid var(--color-border);
      overflow-x: auto;
      margin: 1rem 0;
      line-height: 1.8;
      font-size: 0.9rem;
    }

    .tree span.dir { color: var(--color-success); }
    .tree span.file { color: var(--color-primary); }
    .tree span.comment { color: var(--color-text-muted); font-style: italic; }

    /* Badges */
    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin: 1.5rem 0;
      justify-content: center;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.4rem 0.9rem;
      background: var(--color-card-hover);
      border: 1px solid var(--color-border);
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 500;
      transition: var(--transition);
      text-decoration: none;
      color: var(--color-text);
    }

    .badge:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
      border-color: var(--color-primary);
    }

    .badge.github { background: linear-gradient(135deg, #24292e, #1a1e22); }
    .badge.x { background: linear-gradient(135deg, #000, #1a1a1a); }
    .badge.linkedin { background: linear-gradient(135deg, #0077b5, #005885); }
    .badge.website { background: linear-gradient(135deg, #4285f4, #3367d6); }

    /* Lists */
    ul, ol {
      margin: 0.75rem 0 0.75rem 1.5rem;
      color: var(--color-text-muted);
    }

    li {
      margin-bottom: 0.4rem;
    }

    li strong {
      color: var(--color-text);
    }

    /* Module Cards */
    .modules-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1rem;
      margin: 1rem 0;
    }

    .module-card {
      background: var(--color-card-hover);
      border-radius: var(--radius-md);
      padding: 1.25rem;
      border: 1px solid var(--color-border);
      transition: var(--transition);
    }

    .module-card:hover {
      border-color: var(--color-secondary);
      transform: translateX(4px);
    }

    .module-card h4 {
      color: var(--color-secondary);
      margin: 0 0 0.5rem;
      font-size: 1.1rem;
    }

    .module-card p {
      margin-bottom: 0.5rem;
      font-size: 0.95rem;
    }

    .module-card code {
      font-size: 0.9rem;
    }

    /* Footer */
    footer {
      text-align: center;
      padding: 2rem;
      margin-top: 2rem;
      color: var(--color-text-muted);
      font-size: 0.9rem;
      border-top: 1px solid var(--color-border);
    }

    footer a {
      color: var(--color-primary);
    }

    /* Responsive */
    @media (max-width: 640px) {
      h1 { font-size: 1.5rem; }
      h2 { font-size: 1.3rem; }
      section { padding: 1.5rem; }
      pre, .tree { font-size: 0.85rem; padding: 1rem; }
    }

    /* Highlight callouts */
    .callout {
      display: flex;
      gap: 0.75rem;
      padding: 1rem;
      border-radius: var(--radius-md);
      margin: 1rem 0;
      align-items: flex-start;
    }

    .callout.info {
      background: rgba(59, 130, 246, 0.15);
      border-left: 4px solid var(--color-primary);
    }

    .callout.success {
      background: rgba(16, 185, 129, 0.15);
      border-left: 4px solid var(--color-success);
    }

    .callout.warning {
      background: rgba(245, 158, 11, 0.15);
      border-left: 4px solid var(--color-warning);
    }

    .callout-icon {
      font-size: 1.25rem;
      flex-shrink: 0;
      margin-top: 0.1rem;
    }
  </style>
</head>
<body>
  <div class="container">
    
    <!-- Header -->
    <header>
      <h1>Comparative Analysis of Dimensionality Reduction Techniques for GMM Clustering in Human Activity Recognition</h1>
      <p class="subtitle">
        A comprehensive analysis pipeline for Human Activity Recognition (HAR) using sensor data, featuring dimensionality reduction with PCA and UMAP to optimize Gaussian Mixture Model clustering performance.
      </p>
      <div class="badges">
        <a href="https://github.com/hafezi-ali" class="badge github" target="_blank" rel="noopener">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
          GitHub
        </a>
        <a href="https://x.com/hafezi_alii" class="badge x" target="_blank" rel="noopener">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
          X (Twitter)
        </a>
        <a href="https://www.linkedin.com/in/alihafezii/" class="badge linkedin" target="_blank" rel="noopener">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
          LinkedIn
        </a>
        <a href="https://alihafezi.site" class="badge website" target="_blank" rel="noopener">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
          Website
        </a>
      </div>
    </header>

    <!-- Introduction -->
    <section>
      <p>
        This project presents a comprehensive analysis pipeline for Human Activity Recognition (HAR) using sensor data. The pipeline includes processing raw sensor data, feature engineering, and clustering, along with exploratory data analysis (EDA). Key results, such as visualizations and trained models, are provided for in-depth analysis.
      </p>
      <p>
        Furthermore, the project investigates and compares the impact of dimensionality reduction on the effectiveness of Gaussian Mixture Models (GMMs) in high-dimensional spaces. Two methods, <strong>Principal Component Analysis (PCA)</strong> and <strong>Uniform Manifold Approximation and Projection (UMAP)</strong>, are employed to generate lower-dimensional representations of the data, thereby optimizing GMM performance while mitigating the curse of dimensionality.
      </p>
      
      <div class="dataset-info">
        <strong>📊 Dataset Information:</strong><br>
        Smartphone sensor data from 30 volunteers (aged 19–48) performing six activity classes—<strong>WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING</strong>—collected using a Samsung Galaxy S II on the waist. Accelerometer and gyroscope signals (50Hz) were filtered, segmented into 2.56s windows, and processed to extract time/frequency-domain features. The data is split 70/30 for training/testing.
      </div>

      <p>
        <strong>🔗 Download:</strong> 
        <a href="https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones" target="_blank" rel="noopener">UCI HAR Dataset on Kaggle</a>
      </p>
      
      <div class="callout info">
        <span class="callout-icon">📄</span>
        <div>
          <strong>For detailed results and insights:</strong><br>
          <a href="https://alihafezi.site/uci-har-gaussian-mixture-models/" target="_blank" rel="noopener">GMM HAR Analysis Report</a>
        </div>
      </div>
    </section>

    <!-- Project Overview -->
    <section>
      <h2>🗂️ Project Overview</h2>
      
      <h3>Directory Structure</h3>
      <div class="tree">
<span class="dir">.</span>
├── <span class="dir">Data</span>
│   ├── <span class="dir">processed</span>
│   │
│   └── <span class="dir">raw</span>
│       └── <span class="comment">(contains raw HAR dataset files)</span>
├── <span class="file">main.py</span>
├── <span class="dir">modules</span>
│   ├── <span class="file">clustering.py</span>
│   ├── <span class="file">data_loading.py</span>
│   ├── <span class="file">data_processing.py</span>
│   ├── <span class="file">eda.py</span>
│   ├── <span class="file">feature_engineering.py</span>
│   └── <span class="file">utils.py</span>
├── <span class="file">README.md</span>
├── <span class="file">requirements.txt</span>
└── <span class="dir">results</span>
    └── <span class="dir">plots</span>
        └── <span class="comment">(contains visualization outputs)</span>
      </div>
    </section>

    <!-- Setup Instructions -->
    <section>
      <h2>⚙️ Setup Instructions</h2>
      
      <h3>Step 1: Clone the Repository</h3>
      <pre><code>git clone https://github.com/yourusername/har_analysis_pipeline.git
cd har_analysis_pipeline</code></pre>

      <h3>Step 2: Create and Activate a Virtual Environment</h3>
      <pre><code>python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate</code></pre>

      <h3>Step 3: Install Dependencies</h3>
      <pre><code>pip install -r requirements.txt</code></pre>

      <h3>Step 4: Run the Main Script</h3>
      <pre><code>python main.py</code></pre>

      <div class="callout success">
        <span class="callout-icon">✅</span>
        <div>
          <strong>Tip:</strong> Ensure Python 3.8+ is installed. For GPU-accelerated UMAP, install <code>cudatoolkit</code> via conda.
        </div>
      </div>
    </section>

    <!-- Data Details -->
    <section>
      <h2>📁 Data Details</h2>
      
      <h3>Raw Data Files</h3>
      <ul>
        <li><strong>activity_labels.txt</strong>: Maps activity IDs to activity names</li>
        <li><strong>data_uci.csv</strong>: Original HAR dataset</li>
        <li><strong>subject_test.txt</strong>, <strong>subject_train.txt</strong>: Subject identifiers</li>
        <li><strong>X_test.txt</strong>, <strong>X_train.txt</strong>: Sensor readings</li>
        <li><strong>y_test.txt</strong>, <strong>y_train.txt</strong>: Activity labels</li>
      </ul>

      <h3>Processed Data</h3>
      <ul>
        <li><strong>data_uci_handled_outliers.csv</strong>: Data with outliers addressed</li>
        <li><strong>x_uci_handled_outliers_iqr.csv</strong>: Outliers handled using IQR method</li>
        <li><strong>x_uci_handled_outliers_zs.csv</strong>: Outliers handled using Z-scores</li>
      </ul>

      <h3>Results</h3>
      <p>Contains visualization outputs such as activity distributions, comparison plots, and dimensionality reduction visualizations (PCA/UMAP embeddings).</p>
    </section>

    <!-- Key Components -->
    <section>
      <h2>🧩 Key Components</h2>
      
      <h3><code>main.py</code> – Pipeline Orchestrator</h3>
      <p>The primary script coordinating the entire workflow:</p>
      <ol>
        <li><strong>Setup:</strong> Initializes result directories</li>
        <li><strong>Data Loading & Preprocessing:</strong> Handles missing values, outliers, and saves cleaned datasets</li>
        <li><strong>EDA:</strong> Generates summaries and visualizations</li>
        <li><strong>Feature Engineering:</strong> Scaling + dimensionality reduction (PCA/UMAP)</li>
        <li><strong>Clustering:</strong> GMM with BIC/AIC optimization + cluster visualization</li>
        <li><strong>Completion:</strong> Outputs results and success confirmation</li>
      </ol>

      <h3>Modular Architecture</h3>
      <div class="modules-grid">
        <div class="module-card">
          <h4><code>data_loading.py</code></h4>
          <p>Loads raw and processed data into pandas DataFrames.</p>
          <code>load_data()</code>, <code>load_cleaned_data()</code>
        </div>
        <div class="module-card">
          <h4><code>clustering.py</code></h4>
          <p>Implements GMM clustering with BIC/AIC evaluation.</p>
          <code>criteria_values()</code>, <code>perform_gmm_clustering()</code>
        </div>
        <div class="module-card">
          <h4><code>data_processing.py</code></h4>
          <p>Cleans data and handles outliers via IQR/Z-score methods.</p>
          <code>handle_outliers()</code>, <code>plot_outlier_handling()</code>
        </div>
        <div class="module-card">
          <h4><code>eda.py</code></h4>
          <p>Performs exploratory data analysis and generates summaries.</p>
          <code>perform_eda()</code>
        </div>
        <div class="module-card">
          <h4><code>feature_engineering.py</code></h4>
          <p>Prepares features: scaling, PCA, and UMAP embeddings.</p>
          <code>feature_scaling()</code>, <code>apply_optimal_pca()</code>
        </div>
        <div class="module-card">
          <h4><code>utils.py</code></h4>
          <p>Utility functions for visualization and helpers.</p>
          <code>plot_2d_scatter()</code>
        </div>
      </div>
    </section>

    <!-- License & Contributing -->
    <section>
      <h2>📜 License & Contributing</h2>
      <p>
        This project is licensed under the <a href="./LICENSE" target="_blank">MIT License</a>. 
        See the <code>LICENSE</code> file for details.
      </p>
      
      <div class="callout info">
        <span class="callout-icon">🤝</span>
        <div>
          <strong>Contributions welcome!</strong><br>
          Feel free to fork the repository and submit pull requests. Please follow the existing code style and include tests for new features.
        </div>
      </div>
    </section>

    <!-- Acknowledgements -->
    <section>
      <h2>🙏 Acknowledgements</h2>
      <ul>
        <li>
          <a href="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones" target="_blank" rel="noopener">
            UCI HAR Dataset
          </a> – Primary data source for this project
        </li>
        <li>Researchers and contributors who made sensor-based activity recognition datasets publicly available</li>
        <li>Open-source libraries: scikit-learn, umap-learn, pandas, matplotlib, seaborn</li>
      </ul>
    </section>

    <!-- Future Work -->
    <section>
      <h2>🚀 Future Work</h2>
      <ul>
        <li>Integration of additional machine learning models (e.g., deep autoencoders, t-SNE)</li>
        <li>Exploration of advanced clustering methods (HDBSCAN, spectral clustering)</li>
        <li>Expansion to real-time activity recognition with streaming sensor data</li>
        <li>Cross-device generalization studies and domain adaptation techniques</li>
      </ul>
    </section>

    <!-- Credits -->
    <section id="credits">
      <h2>📜 Credits</h2>
      <p style="text-align: center; font-size: 1.1rem; margin-bottom: 1.5rem;">
        <strong>Ali Hafezi</strong><br>
        <span style="color: var(--color-text-muted);">Project Author & Maintainer</span>
      </p>
      
      <div class="badges" style="justify-content: center;">
        <a href="https://github.com/hafezi-ali" class="badge github" target="_blank" rel="noopener">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
          GitHub
        </a>
        <a href="https://x.com/hafezi_alii" class="badge x" target="_blank" rel="noopener">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
          X (Twitter)
        </a>
        <a href="https://www.linkedin.com/in/alihafezii/" class="badge linkedin" target="_blank" rel="noopener">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
          LinkedIn
        </a>
        <a href="https://alihafezi.site" class="badge website" target="_blank" rel="noopener">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
          Portfolio
        </a>
      </div>
    </section>

    <!-- Footer -->
    <footer>
      <p>
        © 2026 Ali Hafezi. Built with ❤️ for the open-source community.<br>
        <a href="#top">↑ Back to Top</a>
      </p>
    </footer>

  </div>

  <!-- Smooth scroll for anchor links -->
  <script>
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  </script>
</body>
</html>
