import math

def find_point_on_line(m, b, x1, y1, distance):
    # Using the slope (m) and y-intercept (b) to find x2 and y2
    x2 = x1 + distance / math.sqrt(1 + m**2)
    y2 = m * x2 + b

    return x2, y2

# Example usage:
m = -0.121
b = 487
x1 = 254
y1 = 456
distance = float(input("Enter the distance: "))

x2, y2 = find_point_on_line(m, b, x1, y1, distance)

print(f"The coordinates of the second point (x2, y2) are: ({x2}, {y2})")

