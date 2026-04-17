import pandas as pd
import numpy as np

def load_and_process_data(file1_path='附件1.xlsx', file2_path='附件2.xlsx'):
    print("开始执行第一阶段数据清洗与特征工程...\n")

    # ==========================================
    # 1. 物理与逻辑边界解析 (附件1)
    # ==========================================
    # 读取地块数据
    df_land = pd.read_excel(file1_path, sheet_name='乡村的现有耕地')
    df_land = df_land.dropna(subset=['地块名称'])

    # 构建地块索引字典 (Name -> ID) 和 面积字典
    land_name_to_id = {name: idx for idx, name in enumerate(df_land['地块名称'])}
    land_id_to_name = {idx: name for name, idx in land_name_to_id.items()}
    land_area_dict = {land_name_to_id[row['地块名称']]: row['地块面积/亩'] for _, row in df_land.iterrows()}

    # 地块类型分类 (用于后续适用性约束)
    land_type_dict = {land_name_to_id[row['地块名称']]: row['地块类型'] for _, row in df_land.iterrows()}

    # 读取作物数据
    df_crop = pd.read_excel(file1_path, sheet_name='乡村种植的农作物')
    df_crop = df_crop.dropna(subset=['作物编号'])

    # 作物ID映射 (保持原编号1-41不变，方便与原表对齐)
    crop_id_to_name = {int(row['作物编号']): row['作物名称'] for _, row in df_crop.iterrows()}

    # 构建豆类集合 (约束：三年内必须种一次豆类)
    bean_set = set(df_crop[df_crop['作物类型'].str.contains('豆')]['作物编号'].astype(int))

    # ==========================================
    # 2. 状态基准与经济参数解析 (附件2)
    # ==========================================
    # 读取2023年种植历史 (初始状态 t=0)
    df_history = pd.read_excel(file2_path, sheet_name='2023年的农作物种植情况')
    df_history = df_history.dropna(subset=['种植地块'])

    # 构建 t=0 的状态记录: history_state[land_id][season] = [crop_ids]
    # 注意：水浇地和大棚有两季，其余单季
    history_state = {lid: {'单季': [], '第一季': [], '第二季': []} for lid in land_name_to_id.values()}

    for _, row in df_history.iterrows():
        l_name = row['种植地块']
        if pd.isna(l_name): continue  # 跳过合并单元格产生的空行

        # 处理合并单元格向下填充导致的缺失名称 (需要在使用前做ffill处理)

    # 读取经济参数
    df_econ = pd.read_excel(file2_path, sheet_name='2023年统计的相关数据')

    # 清洗销售单价 (将 'A-B' 格式转为均值作为基准参数)
    def parse_price(price_str):
        if isinstance(price_str, (int, float)): return price_str
        parts = str(price_str).split('-')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2.0
        return float(price_str)

    df_econ['平均售价'] = df_econ['销售单价/(元/斤)'].apply(parse_price)

    # 构建利润参数字典: profit_dict[crop_id][land_type] = 单位亩收益
    # 亩收益 = 亩产量 * 平均售价 - 种植成本
    profit_dict = {}
    for _, row in df_econ.iterrows():
        c_id = int(row['作物编号'])
        l_type = row['地块类型'].strip() if isinstance(row['地块类型'], str) else row['地块类型']
        yield_per_mu = float(row['亩产量/斤'])
        cost = float(row['种植成本/(元/亩)'])
        price = row['平均售价']

        if c_id not in profit_dict:
            profit_dict[c_id] = {}
        profit_dict[c_id][l_type] = (yield_per_mu * price) - cost

    # ==========================================
    # 3. 输出数据资产概况自检
    # ==========================================
    print(f"✅ 地块数量提取: {len(land_name_to_id)} 个 (预期34个+20个大棚=54个地块变量)")
    print(f"✅ 作物种类提取: {len(crop_id_to_name)} 种 (预期41种)")
    print(f"✅ 豆类作物集合: {len(bean_set)} 种 -> {bean_set}")
    print(f"✅ 经济数据解析: 成功解析 {len(df_econ)} 条作物-地块收益规则")

    # 抽查测试
    sample_crop_id = 1
    sample_land_type = '平旱地'
    if sample_crop_id in profit_dict and sample_land_type in profit_dict[sample_crop_id]:
        print(f"🔍 抽查: 作物[{crop_id_to_name[sample_crop_id]}] 在 [{sample_land_type}] 的基准亩利润为: {profit_dict[sample_crop_id][sample_land_type]:.2f} 元")

    return {
        'land_name_to_id': land_name_to_id,
        'land_area_dict': land_area_dict,
        'land_type_dict': land_type_dict,
        'crop_id_to_name': crop_id_to_name,
        'bean_set': bean_set,
        'profit_dict': profit_dict
    }

# 执行清洗函数
if __name__ == "__main__":
    # 注意：如果 df_history 有合并单元格导致的空值，需要在外部预处理，例如：
    # df_history['种植地块'] = df_history['种植地块'].fillna(method='ffill')
    data_assets = load_and_process_data()