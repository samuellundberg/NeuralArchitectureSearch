import sys
import csv

def line_reader(line):
    if len(line) == 10000:
        sys.exit()
    if line[:2] == '20':
        # print(line)
        commas = 0
        val_str = str()
        for char in line:
            if char == ',':
                commas += 1
            elif commas == 9:
                val_str += char
            if commas == 10:
                break

        value = float(val_str)
        data_array['Value'].append(value)
    elif line[:3] == 'Now':
        print(line, ' We have added ', len(data_array['Value']), ' points to data array')


def save_file():
    path = 'reconstructed_evolution.csv'
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(list(data_array.keys()))
        tmp_list = [param_space.convert_types_to_string(j, data_array) for j in list(data_array.keys())]
        tmp_list = list(zip(*tmp_list))
        for i in range(len(evolution_data_array[optimization_metrics[0]])):
            w.writerow(tmp_list[i])


def text_reader(file):
    f = open(file, 'r')
    text = f.readlines()
    # c = 1
    for line in text:
        # if c == 53:
        #     if line:
        #         print(len(line))
        #     print(line)
        # c += 1
        line_reader(line)

data_array = dict()
data_array['Value'] = list()
file = sys.argv[1]
text_reader(file)
print(len(data_array['Value']))

res = sorted(data_array['Value'])
print(res)
print(len(res))
