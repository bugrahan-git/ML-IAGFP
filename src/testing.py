import csv
from main import run


with open('../test_lenet5.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    ctr = False
    for row in csv_reader:
        if not ctr:
            ctr = True
        else:
            run(EPOCHS=int(row[1]), BATCH_SIZE=int(row[0]), ACTIVATION_FUNCTION=row[2], MODEL="LeNet-5")




