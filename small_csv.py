import csv
csv_file = open('E:\\sl2021Project\\protein data\\pdb_data_no_dups.csv',
                mode='r', encoding='utf-8')
reader = csv.reader(csv_file)
num = 0
f = open('E:\\sl2021Project\\protein data\\1.csv',
         'w', encoding='utf-8', newline='')
writer = csv.writer(f)
for item in reader:
    if(item):
        num = num+1
        writer.writerow(item)
    if(num > 5000):
        break
f.close()
csv_file.close()
csv_file = open('E:\\sl2021Project\\protein data\\pdb_data_seq.csv',
                mode='r', encoding='utf-8')
reader = csv.reader(csv_file)
num = 0
f = open('E:\\sl2021Project\\protein data\\2.csv',
         'w', encoding='utf-8', newline='')
writer = csv.writer(f)
for item in reader:
    if(item):
        num = num+1
        writer.writerow(item)
    if(num > 5000):
        break
f.close()
csv_file.close()
