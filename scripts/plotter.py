import matplotlib.pyplot as plt 

lines = []
with open("dm_info_2.txt", "r+") as file:
    for line in file:
        lines.append(int(line))


plt.hist(lines)
plt.title("Fiber diameter measurements (100)")
plt.ylabel("Frequency")
plt.xlabel("Fiber diameter (nm)")
plt.show()