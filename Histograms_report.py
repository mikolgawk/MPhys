# import matplotlib.pyplot as plt
# compound_score_list = []
# with open('hitograms_compound_scores.txt', 'r') as file:
#     lines = file.readlines()
#     compound_score_list = [line.strip() for line in lines]
# list = compound_score_list
#
# float_list = [float(element) for element in list]
#
#
# float_list2 = [element for element in float_list if element >= 0]
#
# print(len(float_list))
#
# plt.hist(float_list, bins=50, color='blue', edgecolor='black')
#
# title = 'Compound Flatness Score Histogram - 60548 materials'
# # Add title and labels
# # plt.title(title)
# plt.xlabel('Compound Flatness Score')
# plt.ylabel('Number of materials')
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.55)
# # Show the plot
# plt.show()



import matplotlib.pyplot as plt

compound_score_list = []

# Assuming 'hitograms_compound_scores.txt' contains one score per line
with open('hitograms_compound_scores.txt', 'r') as file:
    lines = file.readlines()
    compound_score_list = [line.strip() for line in lines]

float_list = [float(element) for element in compound_score_list]

# Separate the values based on the condition
float_list2 = [element for element in float_list if element >= 0.9]
float_list3 = [element for element in float_list if element >= 0.99]

print(len(float_list3))

# Create a histogram
plt.hist([float_list3, float_list2], bins=50, color=['yellow', 'blue'], edgecolor='black', label=['>= 0.99','>= 0.9'])

# Customize labels and title
title = 'Compound Flatness Score Histogram - [0.9,1]'
# plt.title(title)
plt.xlabel('Compound Flatness Score')
plt.ylabel('Number of materials')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.55)
# Add a legend
plt.legend()

# Show the plot
plt.show()

# import matplotlib.pyplot as plt
#
# compound_score_list = []
#
# # Assuming 'hitograms_compound_scores.txt' contains one score per line
# with open('hitograms_compound_scores.txt', 'r') as file:
#     lines = file.readlines()
#     compound_score_list = [line.strip() for line in lines]
#
# float_list = [float(element) for element in compound_score_list]
#
# # Separate the values based on the condition
# float_list2 = [element for element in float_list if element >= 0.9]
#
# # Create a histogram
# plt.hist([float_list2], bins=50, color='blue', edgecolor='black', label=['>= 0.9'])
#
# # Customize labels and title
# title = 'Compound Flatness Score >= 0.9 Histogram'
# plt.title(title)
# plt.xlabel('Compound Flatness Score')
# plt.ylabel('Number of materials')
#
# # Add a legend
# plt.legend()
#
# # Show the plot
# plt.show()
