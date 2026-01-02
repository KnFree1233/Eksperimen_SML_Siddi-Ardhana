from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Memuat Dataset
curr_dir = Path(__file__).parent.absolute()
dataset_path = curr_dir.parent / 'covid19-patient-symptoms-diagnosis_raw.csv'
df = pd.read_csv(dataset_path)

# Menghapus patient_id
df_cleaned = pd.DataFrame(df)
df_cleaned.drop("patient_id", axis=1, inplace=True)
numeric_columns = df_cleaned.select_dtypes(include=['number']).columns

#  Menangani Data Kosong
df_cleaned["comorbidity"] = df_cleaned["comorbidity"].fillna("None")
df_cleaned.dropna(inplace=True)

# Menangani Data Duplikat
df_cleaned.drop_duplicates(inplace=True)

# Menangani Outlier
for column in numeric_columns:
    Q1 = df_cleaned[column].quantile(0.25)
    Q3 = df_cleaned[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_cleaned = df_cleaned[(df_cleaned[column] >= lower) & (df_cleaned[column] <= upper)]

# Standarisasi Data Numerik
df_encoded = pd.DataFrame(df_cleaned)
selected_features = ["age", "oxygen_level", "body_temperature"]
selected_numeric_columns = pd.Index(selected_features)

df_encoded[selected_numeric_columns] = StandardScaler().fit_transform(df_encoded[selected_numeric_columns])

# Encoding Data Kategori
category_columns = df_encoded.select_dtypes(include=['object']).columns
for column in category_columns:
    df_encoded[column] = LabelEncoder().fit_transform(df_encoded[column])

# Menyimpan Dataset Hasil Preprocessing
df_encoded.to_csv('covid19-patient-symptoms-diagnosis_preprocessing.csv', index=False)