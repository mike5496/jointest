import pandas as pd

rows = 1000
inpath = 'data/'
outpath = 'data/sample/'

files = {'application_train.csv', 'bureau.csv', 'bureau_balance.csv', 'credit_card_balance.csv', 'installments_payments.csv', 'POS_CASH_balance.csv', 'previous_application.csv', }

for file in files:
    print('Reading file: ' + file)
    file_data = pd.read_csv(inpath + file)

    file_data = file_data.sample(rows)

    print('Writing file: ' + file)
    file_data.to_csv(outpath + file)
    #pd.DataFrame.to_csv(outpath + file)
