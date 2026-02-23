"""
EDA report generation.
"""

import math
import pandas as pd
from datetime import datetime


def save_summary(trans, attrs, orders_by_sku, repurchase, monthly, diversity, by_channel, by_segment, exact_dups_removed, output_dir):
    """
    Export all EDA findings to a TXT.
    """

    n_orders = trans['ORDER_ID'].nunique()
    n_customers = trans['ACCOUNT_ID'].nunique()
    n_skus = trans['SKU_ID'].nunique()
    n_transactions = len(trans)
    n_interactions = trans.groupby(['ACCOUNT_ID', 'SKU_ID']).ngroups
    sparsity = 1 - (n_interactions / (n_customers * n_skus))
    repurchase_rate = (repurchase['n_orders'] >= 2).mean()

    total_orders = orders_by_sku.sum()
    cumsum = orders_by_sku.cumsum()
    pct_cumsum = cumsum / total_orders * 100
    n_skus_80pct = (pct_cumsum < 80).sum() + 1
    n_skus_95pct = (pct_cumsum < 95).sum() + 1

    exact_dups = exact_dups_removed
    logical_dups = trans.duplicated(subset=['ACCOUNT_ID', 'ORDER_ID', 'SKU_ID']).sum()

    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date_start = trans['INVOICE_DATE'].min().strftime('%Y-%m-%d')
    date_end = trans['INVOICE_DATE'].max().strftime('%Y-%m-%d')
    date_days = int((trans['INVOICE_DATE'].max() - trans['INVOICE_DATE'].min()).days)

    duplicate_rate = round(logical_dups / n_transactions * 100, 2)
    coverage_pct = round(trans['ACCOUNT_ID'].isin(attrs['ACCOUNT_ID']).sum() / len(trans) * 100, 2)
    missing_customers = trans[~trans['ACCOUNT_ID'].isin(attrs['ACCOUNT_ID'])]['ACCOUNT_ID'].unique()

    orders_per_cust = trans.groupby('ACCOUNT_ID')['ORDER_ID'].nunique()
    orders_per_cust_mean = round(orders_per_cust.mean(), 1)
    orders_per_cust_median = int(orders_per_cust.median())
    orders_per_cust_min = int(orders_per_cust.min())
    orders_per_cust_max = int(orders_per_cust.max())

    skus_per_order = trans.groupby('ORDER_ID')['SKU_ID'].nunique()
    skus_per_order_mean = round(skus_per_order.mean(), 1)
    skus_per_order_median = int(skus_per_order.median())
    skus_per_order_min = int(skus_per_order.min())
    skus_per_order_max = int(skus_per_order.max())

    top_10_skus_orders = orders_by_sku.head(10)
    customers_by_sku = trans.groupby('SKU_ID')['ACCOUNT_ID'].nunique()
    qty_by_sku = trans.groupby('SKU_ID')['ITEMS_PHYS_CASES'].sum().sort_values(ascending=False)
    pareto_80_catalog_pct = round(n_skus_80pct / len(orders_by_sku) * 100, 1)
    pareto_95_catalog_pct = round(n_skus_95pct / len(orders_by_sku) * 100, 1)

    div_mean = round(diversity['n_unique_skus'].mean(), 1)
    div_median = int(diversity['n_unique_skus'].median())
    div_min = int(diversity['n_unique_skus'].min())
    div_max = int(diversity['n_unique_skus'].max())
    avg_skus_per_order_mean = round(diversity['avg_skus_per_order'].mean(), 1)
    avg_skus_per_order_median = round(diversity['avg_skus_per_order'].median(), 1)

    correlation = None
    if 'SkuDistintosPromediosXOrden' in attrs.columns:
        merged_div = diversity.merge(attrs[['ACCOUNT_ID', 'SkuDistintosPromediosXOrden']],
                                     on='ACCOUNT_ID', how='left')
        correlation = round(merged_div[['avg_skus_per_order', 'SkuDistintosPromediosXOrden']].corr().iloc[0, 1], 3)

    range_1 = round((repurchase['n_orders'] == 1).mean() * 100, 1)
    range_2_3 = round(((repurchase['n_orders'] >= 2) & (repurchase['n_orders'] <= 3)).mean() * 100, 1)
    range_4_5 = round(((repurchase['n_orders'] >= 4) & (repurchase['n_orders'] <= 5)).mean() * 100, 1)
    range_6_plus = round((repurchase['n_orders'] >= 6).mean() * 100, 1)

    weekend_pct = round(trans['INVOICE_DATE'].dt.dayofweek.ge(5).mean() * 100, 1)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_names_es = {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miercoles',
                    'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sabado', 'Sunday': 'Domingo'}
    by_day = trans.groupby(trans['INVOICE_DATE'].dt.day_name())['ORDER_ID'].nunique().reindex(day_order)

    nulls_trans = trans.isnull().sum()
    nulls_trans = nulls_trans[nulls_trans > 0]

    nulls_attrs = attrs.isnull().sum()
    nulls_attrs = nulls_attrs[nulls_attrs > 0]

    neg_qty = (trans['ITEMS_PHYS_CASES'] <= 0).sum()

    attr_details = []
    for col in attrs.columns:
        if col == 'ACCOUNT_ID':
            continue
        dtype = attrs[col].dtype
        n_missing = attrs[col].isna().sum()
        pct_missing = round(n_missing / len(attrs) * 100, 1)
        n_unique = attrs[col].nunique()
        attr_details.append({
            'col': col, 'dtype': str(dtype), 'n_missing': n_missing,
            'pct_missing': pct_missing, 'n_unique': n_unique
        })

    txt_path = output_dir / 'eda_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DEL ANALISIS EXPLORATORIO DE DATOS (EDA)\n")
        f.write("Sistema de Recomendacion B2B - Quilmes\n\n")

        f.write(f"Generado: {generated_at}\n")
        f.write(f"Rango de fechas: {date_start} a {date_end}\n")
        f.write(f"Periodo: {date_days} dias (~{date_days/30:.1f} meses)\n\n")

        f.write("--- 1. VISION GENERAL DE LOS DATOS ---\n\n")
        f.write(f"  Transacciones:              {n_transactions:>10,}\n")
        f.write(f"  Ordenes:                    {n_orders:>10,}\n")
        f.write(f"  Clientes:                   {n_customers:>10,}\n")
        f.write(f"  SKUs (productos):           {n_skus:>10,}\n")
        f.write(f"  Pares cliente-SKU unicos:   {n_interactions:>10,}\n")
        f.write(f"  Pares posibles (CxS):       {n_customers * n_skus:>10,}\n\n")

        f.write("--- 2. CALIDAD DE DATOS ---\n\n")

        f.write("  2.1 Valores nulos en transacciones:\n")
        if len(nulls_trans) > 0:
            for col, count in nulls_trans.items():
                f.write(f"       - {col}: {count:,} ({count/n_transactions*100:.2f}%)\n")
        else:
            f.write("       Sin valores nulos\n")

        f.write("\n  2.2 Valores nulos en atributos:\n")
        if len(nulls_attrs) > 0:
            for col, count in nulls_attrs.items():
                f.write(f"       - {col}: {count:,} ({count/len(attrs)*100:.1f}%)\n")
        else:
            f.write("       Sin valores nulos\n")

        f.write(f"\n  2.3 Duplicados:\n")
        f.write(f"       - Duplicados exactos (todas las columnas):  {exact_dups:,}\n")
        f.write(f"       - Duplicados logicos (ACCOUNT+ORDER+SKU):   {logical_dups:,} ({duplicate_rate}%)\n")

        f.write(f"\n  2.4 Cobertura transacciones-atributos:\n")
        f.write(f"       - Transacciones con atributos:  {coverage_pct:.2f}%\n")
        f.write(f"       - Clientes sin atributos:       {len(missing_customers):,}\n")

        f.write(f"\n  2.5 Cantidades invalidas (<=0):     {neg_qty:,}\n\n")

        f.write("--- 3. ESTADISTICAS DESCRIPTIVAS ---\n\n")

        f.write("  3.1 Ordenes por cliente:\n")
        f.write(f"       Media:    {orders_per_cust_mean}\n")
        f.write(f"       Mediana:  {orders_per_cust_median}\n")
        f.write(f"       Minimo:   {orders_per_cust_min}\n")
        f.write(f"       Maximo:   {orders_per_cust_max}\n")

        f.write(f"\n  3.2 SKUs distintos por orden:\n")
        f.write(f"       Media:    {skus_per_order_mean}\n")
        f.write(f"       Mediana:  {skus_per_order_median}\n")
        f.write(f"       Minimo:   {skus_per_order_min}\n")
        f.write(f"       Maximo:   {skus_per_order_max}\n")

        f.write(f"\n  3.3 Esparcidad de la matriz:\n")
        f.write(f"       Esparcidad: {sparsity*100:.2f}%\n")
        f.write(f"       ({n_interactions:,} interacciones de {n_customers * n_skus:,} posibles)\n\n")

        f.write("--- 4. ANALISIS DE POPULARIDAD DE PRODUCTOS ---\n\n")

        f.write("  4.1 Top 10 SKUs por numero de ordenes:\n")
        f.write(f"       {'#':<4} {'SKU':<10} {'Ordenes':<10} {'Clientes':<10} {'Cantidad total':<15}\n")
        f.write(f"       {'-'*49}\n")
        for i, (sku, n_ord) in enumerate(top_10_skus_orders.items(), 1):
            n_cust = customers_by_sku.get(sku, 0)
            qty = qty_by_sku.get(sku, 0)
            f.write(f"       {i:<4} {sku:<10} {n_ord:<10,} {n_cust:<10,} {qty:<15,.0f}\n")

        f.write(f"\n  4.2 Concentracion (Analisis de Pareto):\n")
        f.write(f"       - 80% de las ordenes proviene de {n_skus_80pct} SKUs ({pareto_80_catalog_pct}% del catalogo)\n")
        f.write(f"       - 95% de las ordenes proviene de {n_skus_95pct} SKUs ({pareto_95_catalog_pct}% del catalogo)\n")
        f.write(f"       - Implicacion: Alta concentracion, se necesita fallback robusto\n")
        f.write(f"         para productos de baja frecuencia\n\n")

        f.write("--- 5. COMPORTAMIENTO DE RECOMPRA ---\n\n")

        f.write(f"  Tasa de recompra: {repurchase_rate*100:.1f}%\n")
        f.write(f"  ({repurchase_rate*100:.1f}% de los pares cliente-SKU tienen 2+ ordenes)\n")

        f.write(f"\n  5.1 Distribucion por rangos:\n")
        f.write(f"       1 orden:       {range_1:>6.1f}%\n")
        f.write(f"       2-3 ordenes:   {range_2_3:>6.1f}%\n")
        f.write(f"       4-5 ordenes:   {range_4_5:>6.1f}%\n")
        f.write(f"       6+ ordenes:    {range_6_plus:>6.1f}%\n")

        f.write(f"\n  5.2 Distribucion acumulada:\n")
        for threshold in [1, 2, 3, 5, 10]:
            pct = (repurchase['n_orders'] >= threshold).mean() * 100
            count = (repurchase['n_orders'] >= threshold).sum()
            f.write(f"       {threshold:2d}+ ordenes:   {pct:>6.1f}% ({count:>7,} pares)\n")

        f.write(f"\n  5.3 Implicacion para el modelo:\n")
        if repurchase_rate > 0.4:
            f.write(f"       Alta tasa de recompra ({repurchase_rate*100:.1f}%) indica que los patrones\n")
            f.write(f"       de frecuencia y recencia son senales fuertes para recomendacion.\n")
        else:
            f.write(f"       Tasa de recompra moderada ({repurchase_rate*100:.1f}%). Se requiere\n")
            f.write(f"       complementar con senales de popularidad.\n")
        f.write("\n")

        f.write("--- 6. DIVERSIDAD DEL CLIENTE ---\n\n")

        f.write("  6.1 SKUs unicos comprados por cliente:\n")
        f.write(f"       Media:    {div_mean}\n")
        f.write(f"       Mediana:  {div_median}\n")
        f.write(f"       Minimo:   {div_min}\n")
        f.write(f"       Maximo:   {div_max}\n")

        f.write(f"\n  6.2 SKUs promedio por orden (por cliente):\n")
        f.write(f"       Media:    {avg_skus_per_order_mean}\n")
        f.write(f"       Mediana:  {avg_skus_per_order_median}\n")

        if correlation is not None:
            corr_level = 'Alta' if abs(correlation) > 0.8 else 'Moderada' if abs(correlation) > 0.5 else 'Baja'
            f.write(f"\n  6.3 Correlacion con atributo SkuDistintosPromediosXOrden:\n")
            f.write(f"       Correlacion: {correlation:.3f} ({corr_level})\n")

        f.write(f"\n  6.4 Implicacion para K dinamico:\n")

        k_fijo = int(math.ceil(avg_skus_per_order_mean * 1.5))
        f.write(f"       Media de SKUs/orden = {avg_skus_per_order_mean} -> K fijo = ceil({avg_skus_per_order_mean} x 1.5) = {k_fijo}\n")
        f.write(f"       Pero la variabilidad es alta (min={div_min}, max={div_max}),\n")
        f.write(f"       lo que justifica usar K dinamico personalizado por cliente.\n\n")

        f.write("--- 7. ANALISIS POR SEGMENTOS Y CANALES ---\n\n")

        if by_channel is not None and len(by_channel) > 0:
            f.write("  7.1 Principales canales (por cantidad de ordenes):\n")
            f.write(f"       {'#':<4} {'Canal':<22} {'Ordenes':<10} {'Clientes':<10} {'SKUs/orden':<12}\n")
            f.write(f"       {'-'*58}\n")
            for idx, (channel, row) in enumerate(by_channel.iterrows(), 1):
                f.write(f"       {idx:<4} {str(channel):<22} {int(row['n_orders']):>8,}  {int(row['n_customers']):>8,}  {row['avg_order_size']:>8.1f}\n")

            missing_channel = trans.merge(attrs[['ACCOUNT_ID', 'canal']], on='ACCOUNT_ID', how='left')['canal'].isna().sum()
            if missing_channel > 0:
                f.write(f"\n       Transacciones sin canal: {missing_channel:,} ({missing_channel/n_transactions*100:.2f}%)\n")

        if by_segment is not None and len(by_segment) > 0:
            f.write(f"\n  7.2 Principales segmentos (por cantidad de ordenes):\n")
            f.write(f"       {'#':<4} {'Segmento':<22} {'Ordenes':<10} {'Clientes':<10} {'SKUs/orden':<12}\n")
            f.write(f"       {'-'*58}\n")
            for idx, (segment, row) in enumerate(by_segment.iterrows(), 1):
                f.write(f"       {idx:<4} {str(segment):<22} {int(row['n_orders']):>8,}  {int(row['n_customers']):>8,}  {row['avg_order_size']:>8.1f}\n")
        f.write("\n")

        f.write("--- 8. PATRONES TEMPORALES ---\n\n")

        f.write("  8.1 Actividad mensual:\n")
        f.write(f"       {'Mes':<12} {'Ordenes':<10} {'Clientes':<12} {'Transacciones':<15} {'Cantidad total':<15}\n")
        f.write(f"       {'-'*64}\n")
        for _, row in monthly.iterrows():
            f.write(f"       {row['year_month']:<12} {row['n_orders']:>8,}  {row['n_customers']:>10,}  {row['n_transactions']:>13,}  {row['total_qty']:>13,.0f}\n")

        f.write(f"\n  8.2 Ordenes por dia de la semana:\n")
        for day in day_order:
            count = by_day.get(day, 0)
            if pd.notna(count) and count > 0:
                bar_len = int(count / by_day.max() * 30) if by_day.max() > 0 else 0
                bar = '#' * bar_len
                f.write(f"       {day_names_es.get(day, day):12s} {count:>6,.0f}  {bar}\n")

        f.write(f"\n  8.3 Ordenes en fin de semana: {weekend_pct:.1f}%\n\n")

        f.write("--- 9. ATRIBUTOS DEL CLIENTE ---\n\n")

        f.write(f"  Total de clientes en archivo de atributos: {len(attrs):,}\n\n")
        f.write(f"  {'Columna':<35} {'Tipo':<12} {'Nulos':<12} {'% Nulos':<10} {'Unicos':<8}\n")
        f.write(f"  {'-'*77}\n")
        for ad in attr_details:
            f.write(f"  {ad['col']:<35} {ad['dtype']:<12} {ad['n_missing']:<12,} {ad['pct_missing']:<10.1f} {ad['n_unique']:<8,}\n")

        f.write("\n  Detalle por atributo:\n")
        for col in attrs.columns:
            if col == 'ACCOUNT_ID':
                continue
            dtype = attrs[col].dtype
            n_unique = attrs[col].nunique()
            if dtype in ['object', 'category'] or n_unique < 20:
                f.write(f"\n  - {col} (valores mas frecuentes):\n")
                top5 = attrs[col].value_counts().head(5)
                for val, count in top5.items():
                    f.write(f"      {val}: {count:,} ({count/len(attrs)*100:.1f}%)\n")
            else:
                f.write(f"\n  - {col} (distribucion numerica):\n")
                f.write(f"      Media:   {attrs[col].mean():.2f}\n")
                f.write(f"      Mediana: {attrs[col].median():.2f}\n")
                f.write(f"      Min:     {attrs[col].min():.2f}\n")
                f.write(f"      Max:     {attrs[col].max():.2f}\n")

        f.write("\n")

        f.write("--- 10. HALLAZGOS CLAVE E IMPLICACIONES PARA EL MODELO ---\n\n")

        f.write(f"  1. ALTA CONCENTRACION: {pareto_80_catalog_pct}% del catalogo ({n_skus_80pct} SKUs)\n")
        f.write(f"     genera el 80% de las ordenes. Se requiere fallback de popularidad\n")
        f.write(f"     para manejar productos de cola larga y clientes nuevos.\n\n")

        f.write(f"  2. RECOMPRA {'ALTA' if repurchase_rate > 0.4 else 'MODERADA'}: {repurchase_rate*100:.1f}% de pares cliente-SKU\n")
        f.write(f"     tienen 2+ ordenes. Las senales de frecuencia y recencia son\n")
        f.write(f"     predictivas del comportamiento futuro.\n\n")

        f.write(f"  3. ESPARCIDAD {sparsity*100:.1f}%: La matriz de interacciones es muy\n")
        f.write(f"     dispersa. Se necesitan modelos que manejen bien la esparcidad\n")
        f.write(f"     (NMF, EASE) y estrategias de fallback.\n\n")

        f.write(f"  4. DIVERSIDAD VARIABLE: Los clientes compran entre {div_min} y {div_max}\n")
        f.write(f"     SKUs unicos (media: {div_mean}). Esto justifica K dinamico\n")
        f.write(f"     en lugar de un K fijo para todos los clientes.\n\n")

        f.write(f"  5. CANALES DIFERENCIADOS: Los patrones de compra varian entre canales.\n")
        f.write(f"     El fallback debe ser especifico por canal, no solo global.\n\n")

        f.write(f"  6. PERIODO CORTO: {date_days} dias (~{date_days/30:.1f} meses) de datos.\n")
        f.write(f"     Insuficiente para capturar estacionalidad completa (se requeririan 12+ meses).\n")

        f.write("\n\n--- GRAFICOS GENERADOS ---\n\n")
        f.write("  - eda_outputs/product_popularity.png     (Curva de Pareto + Distribucion de popularidad)\n")
        f.write("  - eda_outputs/repurchase_distribution.png (Distribucion de recompra)\n")
        f.write("  - eda_outputs/temporal_patterns.png       (Tendencias mensuales + Ordenes por dia)\n")
        f.write("  - eda_outputs/segment_channel_heatmap.png (Heatmap segmento x canal)\n")
        f.write("  - eda_outputs/customer_diversity.png      (Diversidad de SKUs por cliente)\n")

        f.write("\n--- FIN DEL RESUMEN ---\n")

    print(f"  Saved: {txt_path}")
