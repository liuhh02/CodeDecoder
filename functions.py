# Return the absolute value of x .
def abs_value(x):
    if (x > 0) return x
    return -x

# greets the user with the given name
def greet(name):
    print("Hello, " + name + ". Good morning!")

# Calculate the youngest age
def youngest(age1, age2, age3):
    age = min(age1, age2, age3)
    print("The youngest age is " + age)

# Returns num raised to the given exponent .
def power(num, exponent):
    return num**exponent 