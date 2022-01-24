SEASON = "SPRING"

SEASONS = {
    'SPRING': ['01/03', '31/05'],
    'SUMMMER': ['01/06', '31/08'],
    'AUTUMN': ['01/09', '30/11'],
    'SPRING': ['01/12', '29/02']
    }

df = pd.read_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_raw.csv'), header=0)
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')


# GET SPRING DATA - SPRING
df_season = df[(df['DateTime'].dt.month == 3) | (df['DateTime'].dt.month == 4) | (df['DateTime'].dt.month == 5)]
df_season = df_season.sort_values(by="DateTime")

print(df_season.head())