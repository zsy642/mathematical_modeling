import pulp
import pandas as pd
from data_loader import load_and_process_data

def solve_v3_logic():
    print(">>> 正在启动 V3 模型 (情景 1.2: 弹性降价 50%)...")
    assets = load_and_process_data()

    YEARS = list(range(2024, 2031))
    L_ids = list(assets['land_id_to_name'].keys())
    C_ids = list(assets['crop_id_to_name'].keys())
    SEASONS = ['单季', '第一季', '第二季']

    model = pulp.LpProblem("Crop_Optimization_V3", pulp.LpMaximize)

    # --- 变量定义 ---
    x = pulp.LpVariable.dicts("Area", ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS), lowBound=0, cat='Continuous')
    delta = pulp.LpVariable.dicts("IsPlanted", ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS), cat='Binary')

    # 【V3 核心修改】：将销量拆分为 全价(S1) 和 半价(S2)
    S1 = pulp.LpVariable.dicts("Sales_FullPrice", ((j, t) for j in C_ids for t in YEARS), lowBound=0, cat='Continuous')
    S2 = pulp.LpVariable.dicts("Sales_HalfPrice", ((j, t) for j in C_ids for t in YEARS), lowBound=0, cat='Continuous')

    revenue_terms = []
    cost_terms = []

    # 【V3 核心修改】：目标函数接入分段收益
    for t in YEARS:
        for j in C_ids:
            if j in assets['price_dict']:
                price = list(assets['price_dict'][j].values())[0]
                # 全价收入 + 50%降价收入
                revenue_terms.append(price * S1[j, t] + 0.5 * price * S2[j, t])

    for t in YEARS:
        for i in L_ids:
            l_type = assets['land_type_dict'][i]
            for j in C_ids:
                if j in assets['cost_dict'] and l_type in assets['cost_dict'][j]:
                    cost = assets['cost_dict'][j][l_type]
                    for s in SEASONS:
                        cost_terms.append(cost * x[i, j, t, s])

    model += pulp.lpSum(revenue_terms) - pulp.lpSum(cost_terms), "Total_Profit"

    # --- 约束条件 ---
    for t in YEARS:
        for j in C_ids:
            total_output = pulp.lpSum(
                x[i, j, t, s] * assets['yield_dict'][j][assets['land_type_dict'][i]]
                for i in L_ids for s in SEASONS
                if j in assets['yield_dict'] and assets['land_type_dict'][i] in assets['yield_dict'][j]
            )

            # 【V3 核心修改】：全价销量 + 半价销量 = 总产量
            model += S1[j, t] + S2[j, t] == total_output, f"Total_Output_Match_{j}_{t}"

            # 全价销量 S1 严格受限于市场预期 D_j
            limit = assets['sales_limit'].get(j, 0)
            if limit <= 0: limit = 1e9
            model += S1[j, t] <= limit, f"Sales1_Limit_{j}_{t}"
            # S2 没有上限，由模型自由发挥寻找半价后的利润点

        # ---------------- 以下为物理与时序约束 (与V2完全相同) ----------------
        for i in L_ids:
            l_type = assets['land_type_dict'][i]
            model += pulp.lpSum(x[i, j, t, '单季'] + x[i, j, t, '第一季'] for j in C_ids) <= assets['land_area_dict'][i]
            model += pulp.lpSum(x[i, j, t, '第二季'] for j in C_ids) <= assets['land_area_dict'][i]

            # 物理季次隔离防线
            if l_type in ['平旱地', '梯田', '山坡地']:
                for j in C_ids:
                    model += x[i, j, t, '第一季'] == 0
                    model += x[i, j, t, '第二季'] == 0
            elif '大棚' in l_type:
                for j in C_ids:
                    model += x[i, j, t, '单季'] == 0

            for j in C_ids:
                if j not in assets['profit_dict'] or l_type not in assets['profit_dict'][j]:
                    model += x[i, j, t, '单季'] == 0
                    model += x[i, j, t, '第一季'] == 0
                    model += x[i, j, t, '第二季'] == 0
                else:
                    for s in SEASONS:
                        model += x[i, j, t, s] <= assets['land_area_dict'][i] * delta[i, j, t, s]

                if t == 2024:
                    hist_crops = assets['history_state'][i]['单季'] + assets['history_state'][i]['第一季'] + assets['history_state'][i]['第二季']
                    if j in hist_crops:
                        model += delta[i, j, t, '单季'] + delta[i, j, t, '第一季'] == 0
                else:
                    model += pulp.lpSum(delta[i, j, t, s] for s in SEASONS) + \
                             pulp.lpSum(delta[i, j, t-1, s] for s in SEASONS) <= 1

    for i in L_ids:
        for start_year in range(2023, 2029):
            window = range(start_year, start_year + 3)
            legume_count = []
            for ty in window:
                if ty == 2023:
                    is_legume_2023 = any(c in assets['bean_set'] for c in (assets['history_state'][i]['单季'] + assets['history_state'][i]['第一季']))
                    legume_count.append(1 if is_legume_2023 else 0)
                else:
                    legume_count.append(pulp.lpSum(delta[i, j, ty, s] for j in assets['bean_set'] for s in SEASONS))
            model += pulp.lpSum(legume_count) >= 1

    print(">>> 模型编译完成，启动求解引擎 (TimeLimit: 300s)...")
    model.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=300))

    if pulp.LpStatus[model.status] == 'Optimal':
        print(f"\n✅ V3 求解成功!")
        print(f"💰 V3 7年总利润 (1.2 问 - 降价倾销): {pulp.value(model.objective):.2f} 元")

        # 导出结果2矩阵
        records = []
        for t in YEARS:
            for i in L_ids:
                for s in SEASONS:
                    for j in C_ids:
                        val = x[i, j, t, s].varValue
                        if val and val > 1e-4:
                            records.append({'年份': t, '季次': s, '地块名': assets['land_id_to_name'][i], '作物名': assets['crop_id_to_name'][j], '面积': round(val, 2)})
        df_all = pd.DataFrame(records)
        output_file = 'result1_2_降价输出.xlsx'
        with pd.ExcelWriter(output_file) as writer:
            for year in YEARS:
                df_year = df_all[df_all['年份'] == year]
                if not df_year.empty:
                    pivot_df = pd.pivot_table(df_year, values='面积', index=['季次', '地块名'], columns='作物名', aggfunc='sum', fill_value=0)
                    pivot_df.to_excel(writer, sheet_name=str(year))
        print(f"✅ 情景1.2数据已成功导出: {output_file}")
    else:
        print("❌ 求解异常")

if __name__ == "__main__":
    solve_v3_logic()