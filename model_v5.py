import pulp
import pandas as pd
import numpy as np
from data_loader import load_and_process_data

def solve_v5_iteration(assets, current_prices, current_costs):
    """单次线性规划求解器"""
    YEARS = list(range(2024, 2031))
    L_ids = list(assets['land_id_to_name'].keys())
    C_ids = list(assets['crop_id_to_name'].keys())
    SEASONS = ['单季', '第一季', '第二季']

    model = pulp.LpProblem("Crop_Equilibrium_Iter", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("Area", ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS), lowBound=0, cat='Continuous')
    delta = pulp.LpVariable.dicts("IsPlanted", ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS), cat='Binary')
    S1 = pulp.LpVariable.dicts("S1", ((j, t) for j in C_ids for t in YEARS), lowBound=0)
    S2 = pulp.LpVariable.dicts("S2", ((j, t) for j in C_ids for t in YEARS), lowBound=0)

    # 目标函数：使用本轮迭代的价格和成本
    revenue = pulp.lpSum(current_prices[j, t] * S1[j, t] + 0.5 * current_prices[j, t] * S2[j, t]
                         for t in YEARS for j in C_ids if (j, t) in current_prices)
    costs = pulp.lpSum(current_costs[i, j, t] * x[i, j, t, s]
                       for t in YEARS for i in L_ids for j in C_ids for s in SEASONS if (i, j, t) in current_costs)
    model += revenue - costs

    # 约束条件 (保持 V3 的物理防线与时序约束)
    for t in YEARS:
        for j in C_ids:
            total_output = pulp.lpSum(x[i, j, t, s] * assets['yield_dict'][j][assets['land_type_dict'][i]]
                                      for i in L_ids for s in SEASONS if j in assets['yield_dict'] and assets['land_type_dict'][i] in assets['yield_dict'][j])
            model += S1[j, t] + S2[j, t] == total_output
            limit = assets['sales_limit'].get(j, 1e9) if assets['sales_limit'].get(j, 0) > 0 else 1e9
            model += S1[j, t] <= limit

        for i in L_ids:
            l_type = assets['land_type_dict'][i]
            model += pulp.lpSum(x[i, j, t, '单季'] + x[i, j, t, '第一季'] for j in C_ids) <= assets['land_area_dict'][i]
            model += pulp.lpSum(x[i, j, t, '第二季'] for j in C_ids) <= assets['land_area_dict'][i]
            if l_type in ['平旱地', '梯田', '山坡地']:
                for j in C_ids: model += x[i, j, t, '第一季'] == 0; model += x[i, j, t, '第二季'] == 0
            elif '大棚' in l_type:
                for j in C_ids: model += x[i, j, t, '单季'] == 0

            for j in C_ids:
                if j not in assets['profit_dict'] or l_type not in assets['profit_dict'][j]:
                    for s in SEASONS: model += x[i, j, t, s] == 0
                else:
                    for s in SEASONS: model += x[i, j, t, s] <= assets['land_area_dict'][i] * delta[i, j, t, s]
                if t == 2024:
                    hist = assets['history_state'][i]['单季'] + assets['history_state'][i]['第一季'] + assets['history_state'][i]['第二季']
                    if j in hist: model += delta[i, j, t, '单季'] + delta[i, j, t, '第一季'] == 0
                else:
                    model += pulp.lpSum(delta[i, j, t, s] for s in SEASONS) + pulp.lpSum(delta[i, j, t-1, s] for s in SEASONS) <= 1

    # 三年一豆
    for i in L_ids:
        for start_year in range(2023, 2029):
            legume_count = []
            for ty in range(start_year, start_year + 3):
                if ty == 2023:
                    is_legume_2023 = any(c in assets['bean_set'] for c in (assets['history_state'][i]['单季'] + assets['history_state'][i]['第一季']))
                    legume_count.append(1 if is_legume_2023 else 0)
                else: legume_count.append(pulp.lpSum(delta[i, j, ty, s] for j in assets['bean_set'] for s in SEASONS))
            model += pulp.lpSum(legume_count) >= 1

    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=200))
    return x, model

def main_v5_final():
    print(">>> 正在启动 V5 最终迭代求解器 (SLP序列线性规划实现)...")
    assets = load_and_process_data()
    YEARS = list(range(2024, 2031))
    L_ids = list(assets['land_id_to_name'].keys())
    C_ids = list(assets['crop_id_to_name'].keys())

    # 初始参数设定
    curr_prices = {(j, t): list(assets['price_dict'][j].values())[0] for j in C_ids for t in YEARS if j in assets['price_dict']}
    curr_costs = {(i, j, t): assets['cost_dict'][j][assets['land_type_dict'][i]]
                  for i in L_ids for j in C_ids for t in YEARS if j in assets['cost_dict'] and assets['land_type_dict'][i] in assets['cost_dict'][j]}

    # 定义作物组
    groups = {
        'Grain': [c for c in C_ids if c <= 15],
        'Veg': [c for c in C_ids if 16 < c <= 34],
        'Fungi': [c for c in C_ids if c >= 37]
    }
    # 计算各组历史总产量上限
    group_limits = {gn: sum(assets['sales_limit'].get(c, 0) for c in clist) for gn, clist in groups.items()}

    for iter_idx in range(1, 4): # 执行3轮迭代
        print(f"\n>>> 正在执行第 {iter_idx} 轮市场均衡迭代...")
        x_vars, model = solve_v5_iteration(assets, curr_prices, curr_costs)

        if pulp.LpStatus[model.status] != 'Optimal': break

        print(f"    - 本轮利润: {pulp.value(model.objective):,.2f} 元")

        # 统计本轮结果用于下一轮参数更新
        yearly_crop_area = {t: {j: 0.0 for j in C_ids} for t in YEARS}
        yearly_group_yield = {t: {gn: 0.0 for gn in groups} for t in YEARS}

        for t in YEARS:
            for i in L_ids:
                l_type = assets['land_type_dict'][i]
                for j in C_ids:
                    area_val = sum(x_vars[i, j, t, s].varValue or 0 for s in ['单季','第一季','第二季'])
                    if area_val > 1e-4:
                        yearly_crop_area[t][j] += area_val
                        # 累计组产量
                        for gn, clist in groups.items():
                            if j in clist and j in assets['yield_dict'] and l_type in assets['yield_dict'][j]:
                                yearly_group_yield[t][gn] += area_val * assets['yield_dict'][j][l_type]

        # 更新下一轮参数
        for t in YEARS:
            # 1. 价格更新 (基于组产量)
            for gn, clist in groups.items():
                if yearly_group_yield[t][gn] > group_limits[gn] * 1.05: # 显著溢出
                    for j in clist:
                        if (j, t) in curr_prices: curr_prices[j, t] *= 0.95 # 降价5%
            # 2. 成本更新 (基于单项面积规模)
            for j in C_ids:
                if yearly_crop_area[t][j] > 200: # 规模效应
                    for i in L_ids:
                        if (i, j, t) in curr_costs: curr_costs[i, j, t] *= 0.95 # 减产5%成本

    print("\n✅ 第三问关联性均衡求解完成！")
    print(f"💰 最终均衡状态下总利润: {pulp.value(model.objective):,.2f} 元")
    # 此处可添加导出 result3.xlsx 的逻辑，与 V3 类似

if __name__ == "__main__":
    main_v5_final()