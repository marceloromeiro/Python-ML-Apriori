import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_excel('C:\\Users\\marce\\Desktop\\Trabalho\\Online_Retail.xlsx')

data.head()

data.columns

data.shape

data.isnull().values.any()

data.isnull().sum()

desired_width = 320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 10)

data['Description'] = data['Description'].str.strip()

data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

data = data[~data['InvoiceNo'].str.contains('C')]


# Inserir o pais ---------------->
basket = (data[data['Country'] == "France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


def hot_encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


basket_encoded = basket.applymap(hot_encode)
basket = basket_encoded

frq_items = apriori(basket, min_support=0.1, use_colnames=True)

rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

print(rules)
