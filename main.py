import pandas as pd
import openpyxl
from apyori import apriori

store_data = pd.read_csv('C:\\Users\\marce\\Desktop\\Trabalho\\store_data.csv', header=None)

store_data.head()

records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i, j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

print(len(association_results))

print(association_results[0])

for item in association_results:
    pair = item[0]
    items = [x for x in pair]
    print("Regra: " + items[0] + " -> " + items[1])

    print("Suporte: " + str(item[1]))
    print("Confianca: " + str(item[2][0][2]))
    print("Probabilidade de Compra em conjunto: " + str(item[2][0][3]))
    print("=====================================")
