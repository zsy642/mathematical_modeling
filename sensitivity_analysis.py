import pulp
import numpy as np
import matplotlib.pyplot as plt
import copy
from data_loader import load_and_process_data

# 解决图表中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def run_single_year_model(assets_modified):
    """
    极速单年评估核心（基于 V3 降价机制）
    专门用于灵敏度分析，通过剥离时间序列约束，实现 0.1 秒内极速求解
    """
    YEAR = 2024
    L_ids = list(assets_modified['land_id_to_name'].keys())
    C_ids = list(assets_modified['crop_id_to_name'].keys())
    SEASONS = ['单季', '第一季', '第二季']

    model = pulp.LpProblem("Sensitivity_Test", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("Area", ((i, j, s) for i in L_ids for j in C_ids for s in SEASONS), lowBound=0, cat='Continuous')
    delta = pulp.LpVariable.dicts("IsPlanted", ((i, j, s) for i in L_ids for j in C_ids for s in SEASONS), cat='Binary')
    S1 = pulp.LpVariable.dicts("S1", (j for j in C_ids), lowBound=0)
    S2 = pulp.LpVariable.dicts("S2", (j for j in C_ids), lowBound=0)

    # 目标函数
    rev = pulp.lpSum(list(assets_modified['price_dict'][j].values())[0] * S1[j] +
                     0.5 * list(assets_modified['price_dict'][j].values())[0] * S2[j]
                     for j in C_ids if j in assets_modified['price_dict'])
    cost = pulp.lpSum(assets_modified['cost_dict'][j][assets_modified['land_type_dict'][i]] * x[i, j, s]
                      for i in L_ids for j in C_ids for s in SEASONS
                      if j in assets_modified['cost_dict'] and assets_modified['land_type_dict'][i] in assets_modified['cost_dict'][j])
    model += rev - cost

    # 约束条件
    for j in C_ids:
        total_out = pulp.lpSum(x[i, j, s] * assets_modified['yield_dict'][j][assets_modified['land_type_dict'][i]]
                               for i in L_ids for s in SEASONS if j in assets_modified['yield_dict'] and assets_modified['land_type_dict'][i] in assets_modified['yield_dict'][j])
        model += S1[j] + S2[j] == total_out
        limit = assets_modified['sales_limit'].get(j, 1e9) if assets_modified['sales_limit'].get(j, 0) > 0 else 1e9
        model += S1[j] <= limit

    for i in L_ids:
        l_type = assets_modified['land_type_dict'][i]
        model += pulp.lpSum(x[i, j, '单季'] + x[i, j, '第一季'] for j in C_ids) <= assets_modified['land_area_dict'][i]
        model += pulp.lpSum(x[i, j, '第二季'] for j in C_ids) <= assets_modified['land_area_dict'][i]

        if l_type in ['平旱地', '梯田', '山坡地']:
            for j in C_ids: model += x[i, j, '第一季'] == 0; model += x[i, j, '第二季'] == 0
        elif '大棚' in l_type:
            for j in C_ids: model += x[i, j, '单季'] == 0

        for j in C_ids:
            if j not in assets_modified['profit_dict'] or l_type not in assets_modified['profit_dict'][j]:
                for s in SEASONS: model += x[i, j, s] == 0
            else:
                for s in SEASONS: model += x[i, j, s] <= assets_modified['land_area_dict'][i] * delta[i, j, s]

            # 重茬约束（对比2023）
            hist = assets_modified['history_state'][i]['单季'] + assets_modified['history_state'][i]['第一季'] + assets_modified['history_state'][i]['第二季']
            if j in hist:
                model += delta[i, j, '单季'] + delta[i, j, '第一季'] == 0

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.value(model.objective) if pulp.LpStatus[model.status] == 'Optimal' else None

def perform_sensitivity_analysis():
    print(">>> 正在加载原始数据引擎...")
    base_assets = load_and_process_data()

    # 扰动比例：-20%, -10%, 0%, 10%, 20%
    perturbations = [-0.20, -0.10, 0.0, 0.10, 0.20]

    results = {
        'Price': [],
        'Cost': [],
        'Limit': []
    }

    print("\n>>> 开始执行控制变量法灵敏度推演 (共 15 次独立优化求解)...")

    for p in perturbations:
        ratio = 1 + p

        # 1. 仅扰动销售价格
        print(f"  -> 测试 [销售价格] 波动 {p*100:+.0f}% ...")
        assets_p = copy.deepcopy(base_assets)
        for j in assets_p['price_dict']:
            for l in assets_p['price_dict'][j]:
                assets_p['price_dict'][j][l] *= ratio
        results['Price'].append(run_single_year_model(assets_p))

        # 2. 仅扰动种植成本
        print(f"  -> 测试 [种植成本] 波动 {p*100:+.0f}% ...")
        assets_c = copy.deepcopy(base_assets)
        for j in assets_c['cost_dict']:
            for l in assets_c['cost_dict'][j]:
                assets_c['cost_dict'][j][l] *= ratio
        results['Cost'].append(run_single_year_model(assets_c))

        # 3. 仅扰动销售上限(需求量)
        print(f"  -> 测试 [需求上限] 波动 {p*100:+.0f}% ...")
        assets_l = copy.deepcopy(base_assets)
        for j in assets_l['sales_limit']:
            assets_l['sales_limit'][j] *= ratio
        results['Limit'].append(run_single_year_model(assets_l))

    print("\n>>> 数据计算完成，正在渲染多维对比曲线图...")

    # 开始绘图
    x_labels = [f"{p*100:+.0f}%" for p in perturbations]

    plt.figure(figsize=(10, 6))
    # 将结果转换为万元，方便观察
    plt.plot(x_labels, [r / 10000 for r in results['Price']], marker='o', linewidth=2.5, color='#d62728', label='销售单价波动 (Price)')
    plt.plot(x_labels, [r / 10000 for r in results['Cost']], marker='s', linewidth=2.5, color='#1f77b4', label='种植成本波动 (Cost)')
    plt.plot(x_labels, [r / 10000 for r in results['Limit']], marker='^', linewidth=2.5, color='#2ca02c', label='预期需求上限波动 (Limit)')

    plt.title('农作物种植优化模型灵敏度分析 (以2024年单年切片为例)', fontsize=15, fontweight='bold')
    plt.xlabel('参数扰动幅度', fontsize=12)
    plt.ylabel('系统总利润 (万元)', fontsize=12)
    plt.axvline(x=2, color='gray', linestyle='--', alpha=0.6) # 标出 0% 的基准线
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)

    # 标注极值点
    base_profit = results['Price'][2] / 10000
    plt.annotate(f"基准利润\n{base_profit:.0f}万", xy=(2, base_profit), xytext=(1.8, base_profit*0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))

    plt.savefig('Sensitivity_Analysis_Chart.png', dpi=300, bbox_inches='tight')
    print("✅ 灵敏度分析图表已成功保存为: Sensitivity_Analysis_Chart.png")

if __name__ == "__main__":
    perform_sensitivity_analysis()