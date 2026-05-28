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
      background: linear-gradient(to bottom, var(--
