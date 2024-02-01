# '''
# Use this code to compare two lists containing ICSD entries.
# In this case, the list is compared with the list of the flat-band materials found by the Bernevig's group
# '''


with open('common_elements_compound.txt', 'r') as file1:
    lines1 = file1.readlines()
    # Remove newline characters and create a list
    bernevig = [line.strip() for line in lines1]

with open('common_elements_row.txt', 'r') as file2:
    lines2 = file2.readlines()
    # Remove newline characters and create a list
    us = [line.strip() for line in lines2]

set1 = set(bernevig)
set2 = set(us)

common_elements = list(set2.intersection(set1))
# with open('common_elements_row.txt', 'w') as file3:
#     for element in common_elements:
#         file3.write(element)
#         file3.write('\n')
