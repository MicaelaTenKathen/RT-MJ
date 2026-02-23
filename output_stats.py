"""
Production output statistics.
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('model_outputs')
df = pd.read_csv(OUTPUT_DIR / 'recommendations_production.csv')

n_clients = df['ACCOUNT_ID'].nunique()
k_per_client = df.groupby('ACCOUNT_ID')['k_recommended'].first()

output_path = OUTPUT_DIR / 'output_statistics.txt'

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("Estadisticas del Output de Produccion\n\n")

    f.write(f"{'Dimension':<40} {'Valor':<20}\n")
    f.write(f"{'-'*60}\n")
    f.write(f"{'Total de filas (recomendaciones)':<40} {len(df):,}\n")
    f.write(f"{'Clientes cubiertos':<40} {n_clients:,}\n")
    f.write(f"{'K minimo usado':<40} {k_per_client.min()}\n")
    f.write(f"{'K maximo usado':<40} {k_per_client.max()}\n")
    f.write(f"{'K promedio':<40} {k_per_client.mean():.1f}\n")
    f.write(f"{'K mediana':<40} {k_per_client.median():.0f}\n")
    f.write(f"{'Fecha de scoring':<40} {df['scoring_date'].iloc[0]}\n")
    f.write(f"{'Modelo':<40} {df['model'].iloc[0]}\n")

    f.write(f"\nDistribucion de K\n\n")
    f.write(f"{'K':<8} {'Clientes':<10}\n")
    f.write(f"{'-'*18}\n")
    for k, count in k_per_client.value_counts().sort_index().items():
        f.write(f"{k:<8} {count:,}\n")

print(f"Saved: {output_path}")
