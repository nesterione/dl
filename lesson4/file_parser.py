import pandas as pd 

# чтение файла с данными
with open('house prices dataset.txt') as f:
    # data - список из строк файла
    data = f.readlines()

def proc(string):
    # функция обработки строки с названием колонки
    return string.split(',')[1].strip().replace(';', '')

# создание списка с именами столбцов
columns = [proc(data[i]) for i in range(42,54)]
columns[0] += ' in hundreds of dollars'

# создание структуры pandas DataFrame
df = pd.DataFrame([data[i].split()[1:] for i in range(54,82)], columns=columns)

# запись в csv файл
df.to_csv('train.csv', index=False)
