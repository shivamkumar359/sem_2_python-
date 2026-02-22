# ============================================================
# NumPy and Matplotlib Basics
# ------------------------------------------------------------
# This script demonstrates:
# 1. Vectors and matrices using NumPy
# 2. Understanding array shapes
# 3. Line plots and scatter plots
# 4. Hit-and-trial lines
# 5. Best-fit lines (linear regression)
# 6. Real-world examples
#
# Purpose:
# Build strong intuition for data representation and
# visualization, which is essential for Machine Learning.
# ============================================================


# =========================
# IMPORT REQUIRED LIBRARIES
# =========================
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. VECTORS AND MATRICES
# =========================

# Vector: Marks of a single student
# A vector is a 1D array
marks = np.array([85, 78, 95, 60])
print("Vector (Student Marks):", marks)
print("Shape:", marks.shape)   # (4,)

# Matrix: Marks of multiple students
# Rows -> students, Columns -> subjects
marks_matrix = np.array([
    [85, 78, 95],
    [60, 72, 88],
    [91, 84, 89]
])

print("\nMatrix (Multiple Students):")
print(marks_matrix)
print("Shape:", marks_matrix.shape)   # (3, 3)


# =========================
# 2. STUDY HOURS DATASET
# =========================

# Each row represents one student
# Each column represents one day of the week
study_hours = np.array([
    [2, 3, 4, 5, 6, 4, 3],
    [1, 2, 3, 2, 3, 2, 1],
    [4, 5, 6, 5, 4, 5, 5]
])

print("\nStudy Hours Matrix:")
print(study_hours)
print("Shape:", study_hours.shape)   # (3 students, 7 days)


# =========================
# 3. LINE PLOT
# =========================

# Line plots show continuous trends
# Example: mathematical function y = x^2
x = np.arange(1, 6)
y = x ** 2

plt.plot(x, y, label="y = x²")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Line Plot Example")
plt.legend()
plt.show()


# =========================
# 4. SCATTER PLOT
# =========================

# Scatter plots show relationship between two variables
maths = [60, 70, 80, 90, 100]
science = [65, 68, 78, 85, 95]

plt.scatter(maths, science)
plt.xlabel("Maths Marks")
plt.ylabel("Science Marks")
plt.title("Maths vs Science Marks")
plt.show()


# =========================
# 5. HIT AND TRIAL LINE
# =========================

# Dataset: house size vs price
sizes = np.array([650, 800, 1200, 1500, 2000])   # sq ft
prices = np.array([50, 70, 110, 140, 200])       # lakhs

# Scatter plot of actual data
plt.scatter(sizes, prices, label="Actual Data")

# Guess values of slope (m) and intercept (c)
m, c = 0.1089, -19.97
predicted_prices = m * sizes + c

plt.plot(sizes, predicted_prices, color="green", label="Hit & Trial Line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price (Lakhs)")
plt.legend()
plt.show()


# =========================
# 6. BEST FIT LINE (NUMPY)
# =========================

# Instead of guessing, NumPy calculates best m and c
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y_data = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 14.0, 15.8])

# Linear regression using least squares
m_best, c_best = np.polyfit(x_data, y_data, 1)

# Generate smooth line
x_line = np.linspace(1, 8, 100)
y_line = m_best * x_line + c_best

plt.scatter(x_data, y_data, label="Data Points")
plt.plot(x_line, y_line, color="green", label="Best Fit Line")
plt.legend()
plt.show()

print("\nBest Fit Line Parameters:")
print("Slope (m):", m_best)
print("Intercept (c):", c_best)


# =========================
# 7. TEMPERATURE vs SALES
# =========================

# Real-world positive correlation example
temperature = np.array([20, 22, 24, 26, 28, 30, 32, 34])
sales = np.array([80, 85, 88, 95, 100, 105, 110, 120])

plt.scatter(temperature, sales)
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales")
plt.title("Temperature vs Ice Cream Sales")
plt.show()


# =========================
# 8. BUDGET vs REVENUE
# =========================

# Comparing multiple lines
budgets = np.array([20, 35, 50, 65, 80, 100, 120, 150])
revenues = np.array([45, 60, 90, 110, 140, 180, 210, 260])

plt.scatter(budgets, revenues, label="Actual Data")

# Line A
plt.plot(budgets, 2 * budgets + 10, label="Line A (m=2, c=10)")

# Line B
plt.plot(budgets, 1.8 * budgets + 20, label="Line B (m=1.8, c=20)")

plt.xlabel("Movie Budget")
plt.ylabel("Revenue")
plt.title("Movie Budget vs Revenue")
plt.legend()
plt.show()


# =========================
# CONCLUSION
# =========================

# Line B fits better because it stays closer to most data points.
# Line A slightly overestimates revenue at higher budgets.
# The best-fit line concept is the foundation of linear regression in ML.
