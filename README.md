# Principal-Component-Analysis-PCA-
Hands-on Jupyter Notebook assignment applying eigenvalues, eigenvectors, and PCA to model webpage navigation and perform image compression using Python and NumPy.

📊 Eigenvalues, Eigenvectors, and Principal Component Analysis (PCA)
Welcome to the Final Assignment!
Congratulations on making it this far! This repository contains the culminating assignment for the course, where you'll apply your knowledge of linear algebra, Python, and NumPy to solve real-world problems. This assignment will solidify your understanding of eigenvalues, eigenvectors, and their practical applications, particularly in the fascinating areas of webpage navigation modeling and dimensionality reduction with Principal Component Analysis (PCA).

🎯 What You'll Learn & Achieve
Upon completing this assignment, you will be proficient in:

Applying Linear Transformations: Understand and implement linear transformations in the context of discrete dynamical systems.

Webpage Navigation Modeling: Use transition matrices, eigenvalues, and eigenvectors to model and predict long-term user behavior on a network of webpages (a simplified PageRank concept!).

Principal Component Analysis (PCA): Apply PCA for dimensionality reduction on a real-world image dataset.

Covariance Matrix Calculation: Implement functions to center data and compute covariance matrices, essential steps for PCA.

Eigen-Decomposition for Large Datasets: Efficiently compute eigenvalues and eigenvectors for large matrices using scipy.sparse.linalg.eigsh.

Image Compression: See how PCA can be used to compress images by reducing their dimensionality while retaining essential information.

Understanding Explained Variance: Analyze the contribution of each principal component to the total variance in the dataset.

📚 Table of Contents
Project Overview

Prerequisites

Getting Started

Assignment Structure

Part 1: Webpage Navigation Model

Part 2: Principal Component Analysis (PCA)

Important Notes for Graded Cells

Packages

Dataset

Contributing

License

🚀 Project Overview
This repository hosts a Jupyter Notebook (C1W4_Assignment.ipynb) that guides you through two major applications of eigenvalues and eigenvectors:

Webpage Navigation Modeling (Markov Chains): You'll build a simple model to predict which webpages a user is most likely to visit in the long run, demonstrating the power of eigenvalues in analyzing discrete dynamical systems. This section provides a foundational understanding of concepts similar to those behind Google's original PageRank algorithm.

Image Compression with Principal Component Analysis (PCA): You'll apply PCA to a dataset of cat images to reduce their dimensionality, effectively compressing them. This involves computing covariance matrices, finding principal components, and transforming the data.

📋 Prerequisites
To successfully complete this assignment, ensure you have:

Python 3.x: Basic to intermediate proficiency in Python.

NumPy Fundamentals: A solid understanding of NumPy arrays, matrix operations, and common functions.

Linear Algebra Concepts: Familiarity with vectors, matrices, matrix multiplication, and the theoretical definitions of eigenvalues and eigenvectors.

Jupyter Notebook: Ability to navigate and execute cells within a Jupyter environment.

🛠️ Getting Started
Follow these steps to set up and run the assignment on your local machine:

Clone the Repository:

git clone https://github.com/junaidshah2001/Principal-Component-Analysis-PCA-

Install Dependencies:
This project requires numpy, matplotlib, and scipy. You can install them using pip:

pip install numpy matplotlib scipy jupyter

Data Directory:
Ensure you have a data/ directory in the root of the repository containing the necessary image files for the PCA section. The notebook expects this structure. (If the images are not included in the repo, you might need to add instructions on how to download them, e.g., from Kaggle).

Helper Files:
Make sure utils.py and w4_unittest.py are in the same directory as the Jupyter Notebook. These files contain helper functions and unit tests for the assignment.

Launch Jupyter Notebook:

jupyter notebook C1W4_Assignment.ipynb

This command will open the Jupyter interface in your web browser.

Execute Cells:
Open C1W4_Assignment.ipynb and run all cells sequentially. Pay close attention to the instructions and complete the exercises in the designated code blocks.

📖 Assignment Structure
The notebook is divided into two main application-focused sections:

Part 1: Webpage Navigation Model
This section introduces discrete dynamical systems and their application in modeling webpage navigation.

Transition Matrices: Learn how to define and interpret transition matrices where entries represent probabilities of moving between pages.

State Vectors: Understand how state vectors represent the probabilities of a browser being on a particular page at a given time.

Long-Run Probabilities: Discover how the eigenvector corresponding to the eigenvalue of 1 (for Markov matrices) can efficiently predict long-term page traffic, avoiding computationally expensive iterative multiplications.

Exercises 1 & 2: Practical tasks involving matrix multiplication and verifying eigenvector properties.

Part 2: Principal Component Analysis (PCA)
This section delves into PCA as a powerful dimensionality reduction technique, applied to image compression.

2.1 - Load the Data: Load and prepare a dataset of 64x64 pixel cat images, transforming them into a flattened format suitable for PCA.

2.2 - Get the Covariance Matrix:

Exercise 3: Implement a function to center the image data by subtracting the mean of each pixel.

Exercise 4: Compute the covariance matrix from the centered data.

2.3 - Compute the Eigenvalues and Eigenvectors: Calculate the eigenvalues and eigenvectors of the large covariance matrix efficiently using scipy.sparse.linalg.eigsh. These eigenvectors are your principal components.

2.4 - Transform the Centered Data with PCA:

Exercise 5: Project the original data onto the principal components to reduce its dimensionality.

2.5 - Analyzing Dimensionality Reduction: Visualize and understand the impact of reducing dimensions.

2.6 - Reconstructing Images: Learn how to reconstruct images from their reduced-dimension representations, showcasing the effectiveness of PCA for compression.

2.7 - Explained Variance: Analyze the proportion of variance captured by each principal component, helping you decide on the optimal number of components for your task.

⚠️ Important Notes for Graded Cells
This notebook is designed as an assignment, likely with an autograder. Please adhere to the following critical guidelines:

DO NOT delete or add exercise cells.

Keep your solutions within the original cells provided.

DO NOT import any new libraries.

Avoid importing libraries within any cell designated for grading.

Failing to follow these instructions will interfere with the autograder's functionality.

📦 Packages
The following Python packages are used in this assignment:

numpy: The fundamental package for numerical computing with Python, essential for array operations and linear algebra.

matplotlib.pyplot: A comprehensive library for creating static, animated, and interactive visualizations in Python.

scipy.sparse.linalg: A sub-package of SciPy providing advanced linear algebra routines, including efficient eigenvalue solvers for sparse matrices.

utils: A custom module (utils.py) providing helper functions for data loading and plotting.

w4_unittest: A custom module (w4_unittest.py) containing unit tests to verify your solutions.

🖼️ Dataset
The PCA section of this assignment utilizes a portion of the Cat and Dog Face dataset from Kaggle, specifically focusing on the cat images. Ensure this data is accessible in a data/ directory relative to your notebook.

🤝 Contributing
While this is an assignment, if you discover any typos, bugs, or have suggestions for clearer explanations, feel free to open an issue or submit a pull request. Your feedback is valuable!

Embark on this final journey to master linear algebra's real-world impact!
