# ============================================================
# GRADIENT DESCENT VISUALIZATION (BASICS)
# ------------------------------------------------------------
# This script demonstrates:
# 1. Gradient Descent on a simple quadratic function
# 2. Effect of learning rate and starting point
# 3. Visualizing optimization steps
# 4. Plotting multiple models on the same graph
# ============================================================


# =========================
# IMPORT LIBRARIES
# =========================
import numpy as np
import matplotlib.pyplot as plt


# =========================
# PART 1: GRADIENT DESCENT ON f(x) = x^2
# =========================

# Define the function f(x) = x^2
def f(x):
    return x**2

# Define the gradient (derivative) of f(x)
# Gradient tells the direction of steepest increase
def grad(x):
    return 2 * x


# Starting point for gradient descent
x = -7

# Learning rate controls step size
learning_rate = 0.05

# Number of gradient descent steps
steps = 20

# Store x values to visualize the path taken
history = [x]


# Gradient Descent Loop
for i in range(steps):
    slope = grad(x)                      # Compute gradient at current x
    x = x - learning_rate * slope        # Update x in opposite direction
    history.append(x)                    # Save updated x


# =========================
# PLOT FUNCTION AND STEPS
# =========================

# Create smooth x values for plotting the function curve
x_vals = np.linspace(-10, 10, 200)
y_vals = f(x_vals)

# Plot the function y = x^2
plt.plot(x_vals, y_vals, label="y = x²")

# Plot gradient descent steps
plt.scatter(history, [f(i) for i in history],
            color="red", label="Gradient Descent Steps")

plt.legend()
plt.title("Gradient Descent on f(x) = x²")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()


# EXPERIMENT SUGGESTIONS:
# - Try different starting points: x = 10, x = -7
# - Try learning rates: 0.01 (slow), 0.5 (may overshoot)
# - Observe behavior when slope becomes zero (minimum reached)


# =========================
# PART 2: MULTIPLE LINE PLOTS
# =========================

# Sample x values
x = [1, 2, 3, 4, 5]

# Different models / trends
y1 = [1, 4, 9, 16, 25]   # Quadratic growth
y2 = [1, 2, 3, 4, 5]    # Linear growth
y3 = [25, 25, 15, 10, 5]  # Decreasing trend

# Plot all models on the same graph
plt.plot(x, y1, label="Model-A")
plt.plot(x, y2, label="Model-B")
plt.plot(x, y3, label="Model-C")

plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.title("Multiple Models Comparison")
plt.legend()
plt.show()


# =========================
# PART 3: GRADIENT DESCENT ON f(x) = (x - 3)^2 + 1
# =========================

# Define new function
def f(x):
    return (x - 3)**2 + 1

# Derivative of the function
def df_dx(x):
    return 2 * (x - 3)


# Starting point
x_start = 0.5

# Learning rate
learning_rate = 0.05

# Number of iterations
iterations = 50

# Initialize x
x = x_start

# Store x and f(x) history for visualization
x_history = [x]
y_history = [f(x)]


# =========================
# PLOTTING SETUP
# =========================

plt.figure(figsize=(8, 6))

# Create smooth curve for the function
x_plot = np.linspace(-1, 6, 100)
y_plot = f(x_plot)

# Plot the function curve
plt.plot(x_plot, y_plot,
         label=r"$f(x) = (x-3)^2 + 1$",
         color="blue")


# =========================
# GRADIENT DESCENT LOOP
# =========================

for i in range(iterations):
    gradient = df_dx(x)                      # Compute gradient
    x = x - learning_rate * gradient         # Update x
    x_history.append(x)                      # Save x
    y_history.append(f(x))                   # Save f(x)

    # Print step-by-step updates
    print(f"Step {i+1}: x = {x:.3f}, "
          f"f(x) = {f(x):.3f}, "
          f"gradient = {gradient:.3f}")

    # Plot current point
    plt.scatter(x, f(x), color="red", zorder=5)

    # Plot path taken so far
    plt.plot(x_history, y_history,
             color="green", linestyle="--", alpha=0.5)


# =========================
# FINAL PLOT FORMATTING
# =========================

plt.title("Gradient Descent on $f(x) = (x-3)^2 + 1$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
