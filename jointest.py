import pandas as pd

inpath = 'data/sample'

'''
application_train_data = pd.read_csv('data/application_train.csv')
bureau_new_data = pd.read_csv('data/bureau.csv')
'''

def dojoin(left, right, key):
    result = pd.merge(left, right, on=key, how='left')


previous_application_data = pd.read_csv('data/previous_application.csv')
credit_card_balance_data = pd.read_csv('data/credit_card_balance.csv')
installments_payments_data = pd.read_csv('data/installments_payments.csv')

print(previous_application_data)

previous_application_data = pd.merge(previous_application_data, credit_card_balance_data, on='SK_ID_PREV', how='left')

print(previous_application_data)

'''
previous_application_data = pd.merge(previous_application_data, installments_payments_data, on='SK_ID_PREV', how='left')

print(previous_application_data)

newjoined = pd.merge(joined, previous_application_data, on='SK_ID_CURR', how='left')

print(newjoined)
'''
