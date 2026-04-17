import pandas as pd
import numpy as np

def load_and_process_data(file1_path='附件1.xlsx', file2_path='附件2.xlsx'):
    print(">>> 正在挂载数据引擎 (V2版: 包含时序与滞销上限)...\n")

    # 1. 物理与逻辑边界解析
    df_land = pd.read_excel(file1_path, sheet_name='乡村的现有耕地')
    df_land = df_land.dropna(subset=['地块名称'])
    land_name_to_id = {name: idx for idx, name in enumerate(df_land['地块名称'])}
    land_id_to_name = {idx: name for name, idx in land_name_to_id.items()}
    land_area_dict = {land_name_to_id[row['地块名称']]: row['地块面积/亩'] for _, row in df_land.iterrows()}
    land_type_dict = {land_name_to_id[row['地块名称']]: row['地块类型'] for _, row in df_land.iterrows()}

    df_crop = pd.read_excel(file1_path, sheet_name='乡村种植的农作物')
    df_crop = df_crop.dropna(subset=['作物编号'])
    crop_id_to_name = {int(row['作物编号']): row['作物名称'] for _, row in df_crop.iterrows()}
    bean_set = set(df_crop[df_crop['作物类型'].str.contains('豆')]['作物编号'].astype(int))

    # 2. 经济参数解析
    df_econ = pd.read_excel(file2_path, sheet_name='2023年统计的相关数据')
    def parse_price(price_str):
        if isinstance(price_str, (int, float)): return price_str
        parts = str(price_str).split('-')
        return (float(parts[0]) + float(parts[1])) / 2.0 if len(parts) == 2 else float(price_str)

    df_econ['平均售价'] = df_econ['销售单价/(元/斤)'].apply(parse_price)

    profit_dict = {}
    yield_dict = {} # 新增：亩产字典
    cost_dict = {}  # 新增：成本字典
    price_dict = {} # 新增：售价字典

    for _, row in df_econ.iterrows():
        c_id = int(row['作物编号'])
        l_type = row['地块类型'].strip() if isinstance(row['地块类型'], str) else row['地块类型']
        y = float(row['亩产量/斤'])
        c = float(row['种植成本/(元/亩)'])
        p = row['平均售价']

        if c_id not in profit_dict:
            profit_dict[c_id], yield_dict[c_id], cost_dict[c_id], price_dict[c_id] = {}, {}, {}, {}

        profit_dict[c_id][l_type] = (y * p) - c
        yield_dict[c_id][l_type] = y
        cost_dict[c_id][l_type] = c
        price_dict[c_id][l_type] = p

    # 3. 2023年历史数据解析 (核心升级)
    df_history = pd.read_excel(file2_path, sheet_name='2023年的农作物种植情况')
    # 处理合并单元格产生的NaN
    df_history['种植地块'] = df_history['种植地块'].ffill()

    # history_state[land_id][season] = [crop_ids]
    history_state = {lid: {'单季': [], '第一季': [], '第二季': []} for lid in land_name_to_id.values()}
    # 预期销售量上限 D_j (假设2024-2030持平于2023)
    sales_limit_2023 = {cid: 0.0 for cid in crop_id_to_name.keys()}

    for _, row in df_history.iterrows():
        l_name = row['种植地块']
        if l_name not in land_name_to_id: continue
        lid = land_name_to_id[l_name]
        cid = int(row['作物编号'])
        season = row['种植季次']
        area = float(row['种植面积/亩'])

        history_state[lid][season].append(cid)

        # 计算2023年总产量作为销售上限
        l_type = land_type_dict[lid]
        if cid in yield_dict and l_type in yield_dict[cid]:
            sales_limit_2023[cid] += area * yield_dict[cid][l_type]

    return {
        'land_name_to_id': land_name_to_id,
        'land_id_to_name': land_id_to_name,
        'land_area_dict': land_area_dict,
        'land_type_dict': land_type_dict,
        'crop_id_to_name': crop_id_to_name,
        'bean_set': bean_set,
        'profit_dict': profit_dict,
        'yield_dict': yield_dict,
        'cost_dict': cost_dict,
        'price_dict': price_dict,
        'history_state': history_state,
        'sales_limit': sales_limit_2023
    }

if __name__ == "__main__":
    assets = load_and_process_data()
    print("✅ 2023年小麦销售上限(基准产量):", assets['sales_limit'].get(6, 0), "斤")