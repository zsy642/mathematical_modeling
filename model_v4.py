import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from data_loader import load_and_process_data

def solve_and_simulate_v4():
    print(">>> 正在启动 V4 风险约束模型 (马科维茨惩罚 + 蒙特卡洛模拟)...")
    assets = load_and_process_data()

    YEARS = list(range(2024, 2031))
    L_ids = list(assets['land_id_to_name'].keys())
    C_ids = list(assets['crop_id_to_name'].keys())
    SEASONS = ['单季', '第一季', '第二季']

    # ==========================================
    # [V4 核心修改 1]：定义风险厌恶系数与波动率
    # ==========================================
    LAMBDA_RISK = 0.5  # 风险厌恶系数 (0为极度贪婪，1为极度保守)

    # 假设：粮食类价格波动极小(5%)，蔬菜类波动极大(20%)，豆类居中(10%)
    volatility_dict = {}
    for j, name in assets['crop_id_to_name'].items():
        if '蔬菜' in name or j > 16: # 简化分类：ID大于16的多数为蔬菜或菌类
            volatility_dict[j] = 0.20
        elif j in assets['bean_set']:
            volatility_dict[j] = 0.10
        else:
            volatility_dict[j] = 0.05

    # ==========================================
    # 构建 MILP 求解器 (与V3类似，但使用惩罚价格)
    # ==========================================
    model = pulp.LpProblem("Crop_Optimization_V4_Risk", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("Area", ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS), lowBound=0, cat='Continuous')
    delta = pulp.LpVariable.dicts("IsPlanted", ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS), cat='Binary')
    S1 = pulp.LpVariable.dicts("Sales_FullPrice", ((j, t) for j in C_ids for t in YEARS), lowBound=0, cat='Continuous')
    S2 = pulp.LpVariable.dicts("Sales_HalfPrice", ((j, t) for j in C_ids for t in YEARS), lowBound=0, cat='Continuous')

    revenue_terms = []
    cost_terms = []

    for t in YEARS:
        for j in C_ids:
            if j in assets['price_dict']:
                expected_price = list(assets['price_dict'][j].values())[0]
                # [V4 核心修改 2]：风险惩罚目标函数 P_adj = E(P) - Lambda * StdDev(P)
                std_dev_price = expected_price * volatility_dict[j]
                risk_adjusted_price = expected_price - LAMBDA_RISK * std_dev_price

                # 使用调整后的安全价格计算预期理论收益
                revenue_terms.append(risk_adjusted_price * S1[j, t] + 0.5 * risk_adjusted_price * S2[j, t])

    for t in YEARS:
        for i in L_ids:
            l_type = assets['land_type_dict'][i]
            for j in C_ids:
                if j in assets['cost_dict'] and l_type in assets['cost_dict'][j]:
                    cost = assets['cost_dict'][j][l_type]
                    for s in SEASONS:
                        cost_terms.append(cost * x[i, j, t, s])

    model += pulp.lpSum(revenue_terms) - pulp.lpSum(cost_terms), "Total_Risk_Adjusted_Profit"

    # --- 灌入刚性约束 (与V3完全一致，保持物理隔离与轮作) ---
    print(">>> 正在灌入物理与生态轮作防线...")
    for t in YEARS:
        for j in C_ids:
            total_output = pulp.lpSum(
                x[i, j, t, s] * assets['yield_dict'][j][assets['land_type_dict'][i]]
                for i in L_ids for s in SEASONS if j in assets['yield_dict'] and assets['land_type_dict'][i] in assets['yield_dict'][j]
            )
            model += S1[j, t] + S2[j, t] == total_output
            limit = assets['sales_limit'].get(j, 0)
            if limit <= 0: limit = 1e9
            model += S1[j, t] <= limit

        for i in L_ids:
            l_type = assets['land_type_dict'][i]
            model += pulp.lpSum(x[i, j, t, '单季'] + x[i, j, t, '第一季'] for j in C_ids) <= assets['land_area_dict'][i]
            model += pulp.lpSum(x[i, j, t, '第二季'] for j in C_ids) <= assets['land_area_dict'][i]

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
                    model += pulp.lpSum(delta[i, j, t, s] for s in SEASONS) + pulp.lpSum(delta[i, j, t-1, s] for s in SEASONS) <= 1

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

    print(">>> 启动风险调整求解器 (TimeLimit: 300s)...")
    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=300))

    if pulp.LpStatus[model.status] == 'Optimal':
        print(f"\n✅ 稳健策略求解成功！(理论安全利润: {pulp.value(model.objective):.2f} 元)")

        # ==========================================
        # [V4 核心修改 3]：对求出的方案进行 10000 次蒙特卡洛压力测试
        # ==========================================
        print("\n>>> 正在提取稳健策略，启动 10,000 次蒙特卡洛环境震荡模拟...")
        N_SIMS = 10000
        simulated_profits = np.zeros(N_SIMS)

        # 预先提取各个作物每年规划的“总面积”和“基准总成本”，加速模拟
        plan_area = {t: {j: 0.0 for j in C_ids} for t in YEARS}
        plan_cost = {t: {j: 0.0 for j in C_ids} for t in YEARS}

        for t in YEARS:
            for i in L_ids:
                l_type = assets['land_type_dict'][i]
                for j in C_ids:
                    for s in SEASONS:
                        val = x[i, j, t, s].varValue
                        if val and val > 1e-4:
                            plan_area[t][j] += val
                            plan_cost[t][j] += val * assets['cost_dict'][j][l_type]

        # 开始模拟暴雨、干旱与市场暴涨暴跌
        for sim in range(N_SIMS):
            total_profit_sim = 0
            for t in YEARS:
                for j in C_ids:
                    if plan_area[t][j] > 0:
                        expected_price = list(assets['price_dict'][j].values())[0]
                        # 假设产量方差与价格方差同步关联（实际中可能更复杂，此处做独立正态分布简化）
                        actual_price = np.random.normal(expected_price, expected_price * volatility_dict[j])
                        actual_price = max(0.1, actual_price) # 价格不可能为负

                        # 简化的总体产量扰动
                        avg_yield = np.mean(list(assets['yield_dict'][j].values()))
                        actual_yield_per_mu = np.random.normal(avg_yield, avg_yield * volatility_dict[j])
                        actual_yield_per_mu = max(0.1, actual_yield_per_mu)

                        total_output = plan_area[t][j] * actual_yield_per_mu
                        limit = assets['sales_limit'].get(j, 1e9) if assets['sales_limit'].get(j, 0) > 0 else 1e9

                        # 弹性降价结算逻辑
                        if total_output <= limit:
                            revenue = total_output * actual_price
                        else:
                            revenue = limit * actual_price + (total_output - limit) * 0.5 * actual_price

                        total_profit_sim += (revenue - plan_cost[t][j])

            simulated_profits[sim] = total_profit_sim

        # ==========================================
        # 绘制极具论文杀伤力的风险分布图
        # ==========================================
        mean_profit = np.mean(simulated_profits)
        var_95 = np.percentile(simulated_profits, 5) # 95%置信度下的最坏情况 (VaR)

        print(f"📊 蒙特卡洛模拟完成！")
        print(f"   - 预期平均总利润: {mean_profit:,.2f} 元")
        print(f"   - 95% 极端恶劣天气下保底利润 (VaR): {var_95:,.2f} 元")

        plt.figure(figsize=(10, 6))
        plt.hist(simulated_profits, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(mean_profit, color='red', linestyle='dashed', linewidth=2, label=f'预期均值: {mean_profit/10000:.0f}万')
        plt.axvline(var_95, color='orange', linestyle='solid', linewidth=2, label=f'95% VaR底线: {var_95/10000:.0f}万')
        plt.title('2024-2030 稳健型种植策略的利润概率分布 (10000次蒙特卡洛模拟)', fontsize=14)
        plt.xlabel('7年总利润 (元)', fontsize=12)
        plt.ylabel('出现频次 (概率密度)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig('V4_风险利润分布图.png', dpi=300, bbox_inches='tight')
        print("\n✅ 绝杀图表已生成: V4_风险利润分布图.png (请直接插入论文!)")

    else:
        print("❌ 求解异常")

if __name__ == "__main__":
    solve_and_simulate_v4()