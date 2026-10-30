import pandas as pd

data = pd.read_csv('assets/internship_candidates_final_numeric.csv')
data['EnglishLevel'] = data['EnglishLevel'].replace({
    1: 'Intermediate',
    2: 'Upper-Intermediate',
    3: 'Advanced'
})

data.to_csv('assets/internship_candidates_final_numeric_str.csv', index=False, encoding="utf-8-sig")