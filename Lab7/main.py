import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Task1
# create two lists filled with random numbers
list1 = [np.random.rand() for _ in range(1000000)]
list2 = [np.random.rand() for _ in range(1000000)]

# measure the execution time of the list multiplication operation
start_time_list = time.perf_counter()
result_list = [list1[i] * list2[i] for i in range(len(list1))]
end_time_list = time.perf_counter()

# create two NumPy arrays filled with random numbers
arr1 = np.random.rand(1000000)
arr2 = np.random.rand(1000000)

# measure the execution time of the NumPy array multiplication operation
start_time_np = time.perf_counter()
result_np = np.multiply(arr1, arr2)
end_time_np = time.perf_counter()

# calculate the elapsed time for both operations and print the results
elapsed_time_lst = end_time_list - start_time_list
elapsed_time_np = end_time_np - start_time_np
print(f"Execution time for list multiplication: {elapsed_time_lst:.6f} seconds")
print(f"Execution time for NumPy array multiplication: {elapsed_time_np:.6f} seconds")

# Task 2
# read the CSV file using pandas
data = pd.read_csv("data2.csv")
col1 = data["Sulfate"]
plt.hist(col1, bins=16, density=True, rwidth=0.5)
plt.title('Normalized histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# calculate and print standard deviation
std_dev = np.std(col1)
print('Standard deviation:', std_dev)

# Task3

# define x
x = np.linspace(-3*np.pi, 3*np.pi, 500)

# define y and z
y = x * np.cos(x)
z = np.sin(x)

# create 3d figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the curve
ax.plot(x, y, z)
ax.set_title('Three-dimensional graph')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# additional task
fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i / 30))  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=10, blit=True, save_count=20)

plt.show()