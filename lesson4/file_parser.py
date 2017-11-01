import pandas as pd 

with open('house prices dataset.txt') as f:
    data = f.readlines()

def proc(string):
    return string.split(',')[1].strip().replace(';', '')

columns = [proc(data[i]) for i in range(42,54)]
columns[0] += ' in hundreds of dollars'

df = pd.DataFrame([data[i].split()[1:] for i in range(54,82)], columns=columns)

df.to_csv('train.csv', index=False)
