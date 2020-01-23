import sys

dataPath = './data/train'
dataCsvPath = './data/train_csv'
ttestPath = './data/ttest'
ttestCsvPath = './data/ttest_csv'
out_path = './models'

model_path = './models/118.pkl'
test_path = './data/test/'
test_csv_path = './data/test_csv'
result_path = './results'

EPOCH = 200
LR = 0.00001
min_mae = sys.maxsize