

file_1 = open('result_pages.csv', 'r')
file_2 = open('result_fixed.txt', 'r')

file_1_lines = file_1.readlines()
file_2_lines = file_2.readlines()

result = open('megaresult.txt', 'w')

for i in range(len(file_1_lines)):
    line_id = (int(file_1_lines[i].split('|||')[0].split('.')[-1]))
    result.write('|||'.join((file_1_lines[i][:-1], file_2_lines[line_id][:-1])) + '\n')

