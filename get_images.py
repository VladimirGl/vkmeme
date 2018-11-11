

f = open('result_fixed.csv', 'r')
o = open('urls_only.txt', 'w')
for line in f:
	o.write(line.split('|||')[1] + '\n')
