import pulp
from data_loader import load_and_process_data

def solve_v2_logic():
    print(">>> 正在启动 V2 全量约束求解器 (2024-2030)...")
    assets = load_and_process_data()

    YEARS = list(range(2024, 2031))
    L_ids = list(assets['land_id_to_name'].keys())
    C_ids = list(assets['crop_id_to_name'].keys())
    SEASONS = ['单季', '第一季', '第二季']

    # 建立最大化模型
    model = pulp.LpProblem("Crop_Optimization_V2", pulp.LpMaximize)

    # ==========================================
    # 1. 修复：使用推导式生成器构建多维变量字典
    # ==========================================
    print(">>> 正在挂载高维决策变量空间...")
    # x[i, j, t, s] = 面积
    x = pulp.LpVariable.dicts("Area",
                              ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS),
                              lowBound=0, cat='Continuous')

    # delta[i, j, t, s] = 是否种植(0-1)
    delta = pulp.LpVariable.dicts("IsPlanted",
                                  ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS),
                                  cat='Binary')

    # S[j, t] = 实际有效销量 (受市场预期上限约束)
    S = pulp.LpVariable.dicts("ValidSales",
                              ((j, t) for j in C_ids for t in YEARS),
                              lowBound=0, cat='Continuous')

    # ==========================================
    # 2. 修复：总收入与总成本解耦
    # ==========================================
    print(">>> 正在构建解耦目标函数...")
    revenue_terms = []
    cost_terms = []

    # 计算总收入：价格 P_j * 销量 S_j,t
    for t in YEARS:
        for j in C_ids:
            if j in assets['price_dict']:
                # 获取该作物的统一市场售价
                price = list(assets['price_dict'][j].values())[0]
                revenue_terms.append(price * S[j, t])

    # 计算总成本：地块面积 x_i,j,t,s * 对应地块类型的亩成本 C_i,j
    for t in YEARS:
        for i in L_ids:
            l_type = assets['land_type_dict'][i]
            for j in C_ids:
                if j in assets['cost_dict'] and l_type in assets['cost_dict'][j]:
                    cost = assets['cost_dict'][j][l_type]
                    for s in SEASONS:
                        cost_terms.append(cost * x[i, j, t, s])

    # 最大化总利润 = 收入 - 成本
    model += pulp.lpSum(revenue_terms) - pulp.lpSum(cost_terms), "Total_Profit"

    # ==========================================
    # 3. 核心约束条件灌入
    # ==========================================
    print(">>> 正在灌入时序约束与业务法则...")
    for t in YEARS:
        # (A) 产销平衡与滞销上限约束
        for j in C_ids:
            # 当年该作物总产出 Q_j,t
            total_output = pulp.lpSum(
                x[i, j, t, s] * assets['yield_dict'][j][assets['land_type_dict'][i]]
                for i in L_ids for s in SEASONS
                if j in assets['yield_dict'] and assets['land_type_dict'][i] in assets['yield_dict'][j]
            )
            # 有效销量 S 必须 <= 总产出
            model += S[j, t] <= total_output, f"Sales_Limit_Output_{j}_{t}"

            # 有效销量 S 必须 <= 2023年基准上限 (如果2023没种过，这里暂定赋予极大值不设限)
            limit = assets['sales_limit'].get(j, 0)
            if limit <= 0: limit = 1e9
            model += S[j, t] <= limit, f"Sales_Limit_Market_{j}_{t}"

        for i in L_ids:
            # (新增物理防线) 地块类型与季次的严格匹配
            if l_type in ['平旱地', '梯田', '山坡地']:
                # 旱地类只能种【单季】，封死第一季和第二季
                for j in C_ids:
                    model += x[i, j, t, '第一季'] == 0
                    model += x[i, j, t, '第二季'] == 0
            elif l_type == '水浇地':
                # 水浇地不能出现【单季】与【第一季/第二季】并存的混沌状态
                # 这里可以利用0-1变量控制，或者简化为：只能是单季水稻，或两季蔬菜
                pass # 具体需要在适用性字典里把水浇地单季作物的ID和其他区分开
            elif '大棚' in l_type:
                # 大棚绝对不能有【单季】
                for j in C_ids:
                    model += x[i, j, t, '单季'] == 0
            l_type = assets['land_type_dict'][i]
            # (B) 地块面积容积率约束
            model += pulp.lpSum(x[i, j, t, '单季'] + x[i, j, t, '第一季'] for j in C_ids) <= assets['land_area_dict'][i]
            model += pulp.lpSum(x[i, j, t, '第二季'] for j in C_ids) <= assets['land_area_dict'][i]

            for j in C_ids:
                # (C) 大M法绑定变量与地块适用性硬约束

                if j not in assets['profit_dict'] or l_type not in assets['profit_dict'][j]:
                    model += x[i, j, t, '单季'] == 0
                    model += x[i, j, t, '第一季'] == 0
                    model += x[i, j, t, '第二季'] == 0
                else:
                    for s in SEASONS:
                        model += x[i, j, t, s] <= assets['land_area_dict'][i] * delta[i, j, t, s]

                # (D) 禁止重茬 (跨年时间序列)
                if t == 2024:
                    # 追溯2023年历史数据
                    hist_crops = assets['history_state'][i]['单季'] + assets['history_state'][i]['第一季'] + assets['history_state'][i]['第二季']
                    if j in hist_crops:
                        model += delta[i, j, t, '单季'] + delta[i, j, t, '第一季'] == 0
                else:
                    # 相邻两年不能种同一种作物
                    model += pulp.lpSum(delta[i, j, t, s] for s in SEASONS) + \
                             pulp.lpSum(delta[i, j, t-1, s] for s in SEASONS) <= 1

    # (E) 三年豆类轮作滑动窗口 (2023-2025 到 2028-2030)
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
            # 窗口内豆类种植次数 >= 1
            model += pulp.lpSum(legume_count) >= 1

    print(">>> 模型编译完成，启动求解引擎 (由于添加了三年窗口跨期约束，这可能需要1-3分钟)...")
    # 启用求解进度回显，并设置 300 秒强制阻断防止卡死
    model.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=300))

    if pulp.LpStatus[model.status] == 'Optimal':
        print(f"\n✅ 求解成功!")
        print(f"💰 V2 7年总利润 (1.1 问): {pulp.value(model.objective):.2f} 元")

        # 溯源验证: A1地块2023年种了小麦，看2024年模型选了什么
        a1_id = assets['land_name_to_id']['A1']
        print("\n🔍 溯源检验 A1地块 (平旱地, 80亩, 2023年种植了[小麦]):")
        for t in [2024, 2025]:
            for j in C_ids:
                for s in SEASONS:
                    if x[a1_id, j, t, s].varValue and x[a1_id, j, t, s].varValue > 0.01:
                        print(f"  - {t}年 种植: {assets['crop_id_to_name'][j]}, 面积: {x[a1_id, j, t, s].varValue} 亩")
                # ==========================================
                # 自动化导出模块：生成 result1_1 格式矩阵
                # ==========================================
                import pandas as pd
                print("\n>>> 正在执行矩阵序列化与 Excel 导出...")

                # 1. 扁平化提取非零决策变量
                records = []
                for t in YEARS:
                    for i in L_ids:
                        for s in SEASONS:
                            for j in C_ids:
                                val = x[i, j, t, s].varValue
                                if val and val > 1e-4:  # 消除浮点误差
                                    records.append({
                                        '年份': t,
                                        '季次': s,
                                        '地块名': assets['land_id_to_name'][i],
                                        '作物名': assets['crop_id_to_name'][j],
                                        '面积': round(val, 2)
                                    })

                df_all = pd.DataFrame(records)

                # 2. 按年份生成透视表并写入同一个 Excel 的不同 Sheet
                output_file = 'result1_1_求解输出.xlsx'
                with pd.ExcelWriter(output_file) as writer:
                    for year in YEARS:
                        df_year = df_all[df_all['年份'] == year]
                        if df_year.empty:
                            continue
                        # 构建透视表：行索引为[季次, 地块名]，列索引为作物名，值为面积
                        pivot_df = pd.pivot_table(
                            df_year,
                            values='面积',
                            index=['季次', '地块名'],
                            columns='作物名',
                            aggfunc='sum',
                            fill_value=0
                        )
                        pivot_df.to_excel(writer, sheet_name=str(year))

                print(f"✅ 数据已成功导出至当前目录: {output_file}")
    else:
        print(f"❌ 求解异常: 状态 {pulp.LpStatus[model.status]}")

if __name__ == "__main__":
    solve_v2_logic()