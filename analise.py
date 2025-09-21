import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_2024 = 'formula1-2024.csv'
FILE_2025 = 'formula1-2025.csv'
sns.set_theme(style="whitegrid", palette="viridis")

try:
    df_2024 = pd.read_csv(FILE_2024)
    print(f"Arquivo '{FILE_2024}' carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{FILE_2024}' não encontrado. Verifique o caminho e o nome do arquivo.")
    exit()

try:
    df_2025 = pd.read_csv(FILE_2025)
    print(f"Arquivo '{FILE_2025}' carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{FILE_2025}' não encontrado. Verifique o caminho e o nome do arquivo.")
    exit()

# Converter a coluna 'Position' para um formato numérico
df_2024['Position_numeric'] = pd.to_numeric(df_2024['Position'], errors='coerce')
df_2025['Position_numeric'] = pd.to_numeric(df_2025['Position'], errors='coerce')

# Equipes com mais vitórias
plt.figure(figsize=(12, 7))
vitorias_equipe = df_2024[df_2024['Position_numeric'] == 1]['Team'].value_counts()
sns.barplot(x=vitorias_equipe.index, y=vitorias_equipe.values)
plt.title('Vitórias por equipe em 2024', fontsize=16)
plt.xlabel('Equipe', fontsize=12)
plt.ylabel('Total de Vitórias', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('vitorias_por_equipe_2024.png')
plt.clf()

plt.figure(figsize=(12, 7))
vitorias_equipe = df_2025[df_2025['Position_numeric'] == 1]['Team'].value_counts()
sns.barplot(x=vitorias_equipe.index, y=vitorias_equipe.values)
plt.title('Vitórias por equipe em 2024', fontsize=16)
plt.xlabel('Equipe', fontsize=12)
plt.ylabel('Total de Vitórias', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('vitorias_por_equipe_2025.png')
plt.clf()

# Vitórias por piloto
plt.figure(figsize=(12, 7))
vitorias_piloto = df_2024[df_2024['Position_numeric'] == 1]['Driver'].value_counts()
sns.barplot(x=vitorias_piloto.index, y=vitorias_piloto.values)
plt.title('Vitórias em 2024', fontsize=16)
plt.xlabel('Piloto', fontsize=12)
plt.ylabel('Total de Vitórias', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('vitorias_por_piloto_2024.png')
plt.clf()

plt.figure(figsize=(12, 7))
vitorias_piloto = df_2025[df_2025['Position_numeric'] == 1]['Driver'].value_counts()
sns.barplot(x=vitorias_piloto.index, y=vitorias_piloto.values)
plt.title('Vitórias em 2025', fontsize=16)
plt.xlabel('Piloto', fontsize=12)
plt.ylabel('Total de Vitórias', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('vitorias_por_piloto_2025.png')
plt.clf()

# Relação entre posição de largada e posição final
df_2024_finalizado = df_2024.dropna(subset=['Position_numeric'])
grid_vs_pos_2024 = pd.crosstab(df_2024_finalizado['Starting Grid'], df_2024_finalizado['Position_numeric'])

plt.figure(figsize=(16, 10))
sns.heatmap(grid_vs_pos_2024, cmap="YlGnBu")
plt.title('Correlação entre Posição de Largada e Posição Final', fontsize=16)
plt.xlabel('Posição Final na Corrida', fontsize=12)
plt.ylabel('Posição no Grid de Largada', fontsize=12)
plt.tight_layout()
plt.savefig('grid_vs_posicao_final_2024.png')
plt.clf()

df_2025_finalizado = df_2025.dropna(subset=['Position_numeric'])
grid_vs_pos_2025 = pd.crosstab(df_2025_finalizado['Starting Grid'], df_2025_finalizado['Position_numeric'])

plt.figure(figsize=(16, 10))
sns.heatmap(grid_vs_pos_2025, cmap="YlGnBu")
plt.title('Correlação entre Posição de Largada e Posição Final', fontsize=16)
plt.xlabel('Posição Final na Corrida', fontsize=12)
plt.ylabel('Posição no Grid de Largada', fontsize=12)
plt.tight_layout()
plt.savefig('grid_vs_posicao_final_2025.png')
plt.clf()

# Boxplot de pontos por equipe
# Ordena as equipes pela mediana para uma melhor visualização
plt.figure(figsize=(12, 8))
ordem_equipes = df_2024.groupby('Team')['Points'].median().sort_values(ascending=False).index
sns.boxplot(data=df_2024, x='Points', y='Team', order=ordem_equipes)
plt.title('Distribuição de Pontos Conquistados por Equipe', fontsize=16)
plt.xlabel('Pontos em uma Única Corrida', fontsize=12)
plt.ylabel('Equipe', fontsize=12)
plt.tight_layout()
plt.savefig('distribuicao_pontos_equipe_2024.png')
plt.clf()

plt.figure(figsize=(12, 8))
ordem_equipes = df_2025.groupby('Team')['Points'].median().sort_values(ascending=False).index
sns.boxplot(data=df_2025, x='Points', y='Team', order=ordem_equipes)
plt.title('Distribuição de Pontos Conquistados por Corrida (por Equipe)', fontsize=16)
plt.xlabel('Pontos em uma Única Corrida', fontsize=12)
plt.ylabel('Equipe', fontsize=12)
plt.tight_layout()
plt.savefig('distribuicao_pontos_equipe_2025.png')
plt.clf()

# Heatmap de correlação entre valores numéricos
cols_num = ['Starting Grid', 'Laps', 'Points', 'Position_numeric', 'Total Points', 'Total Position']
matriz_corr = df_2024[cols_num].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor de Correlação entre Features Numéricas', fontsize=16)
plt.tight_layout()
plt.savefig('correlacao_features_2024.png')
plt.clf()

cols_num = ['Starting Grid', 'Laps', 'Points', 'Position_numeric', 'Total Points', 'Total Position']
matriz_corr = df_2025[cols_num].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor de Correlação entre Features Numéricas', fontsize=16)
plt.tight_layout()
plt.savefig('correlacao_features_2025.png')
plt.clf()

# Boxplot de posição na corrida por equipe
# Ordena as equipes pela mediana para uma melhor visualização
plt.figure(figsize=(14, 8))
ordem_equipes = df_2024.groupby('Team')['Position_numeric'].median().sort_values().index
sns.boxplot(data=df_2024, x='Position_numeric', y='Team', order=ordem_equipes)
df_2024.dropna(subset=['Position_numeric'], inplace=True)
df_2024['Position_numeric'] = df_2024['Position_numeric'].astype(int)
plt.title('Distribuição da Posição Final por Equipe', fontsize=18)
plt.xlabel('Posição Final na Corrida', fontsize=12)
plt.ylabel('Equipe', fontsize=12)
plt.xticks(range(1, df_2024['Position_numeric'].max() + 1, 1))
plt.tight_layout()
plt.savefig('distribuicao_posicao_equipe_2024.png')

plt.figure(figsize=(14, 8))
ordem_equipes = df_2025.groupby('Team')['Position_numeric'].median().sort_values().index
sns.boxplot(data=df_2025, x='Position_numeric', y='Team', order=ordem_equipes)
df_2025.dropna(subset=['Position_numeric'], inplace=True)
df_2025['Position_numeric'] = df_2025['Position_numeric'].astype(int)
plt.title('Distribuição da Posição Final por Equipe', fontsize=18)
plt.xlabel('Posição Final na Corrida', fontsize=12)
plt.ylabel('Equipe', fontsize=12)
plt.xticks(range(1, df_2025['Position_numeric'].max() + 1, 1)) 
plt.tight_layout()
plt.savefig('distribuicao_posicao_equipe_2025.png')