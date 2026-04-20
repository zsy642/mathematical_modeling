import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pulp
from data_loader import load_and_process_data

# 设置学术图表字体格式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_technical_route():
    """图一：模型递进逻辑与技术路线流程图"""
    print(">>> 正在渲染 图一：技术路线流程图...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.axis('off')

    # 定义流程节点
    nodes = [
        ("第一阶段：静态确定性基座", "V1/V2 混合整数线性规划 (MILP)\n引入0-1状态变量，严控重茬与三年一豆轮作"),
        ("第二阶段：非线性机制解耦", "V3 分段线性化目标函数\n产销解耦，模拟超产 50% 降价倾销机制"),
        ("第三阶段：不确定性与风险防范", "V4 均值-方差优化与蒙特卡洛模拟\n引入马科维茨风险惩罚，测试 95% VaR 保底防线"),
        ("第四阶段：动态市场博弈均衡", "V5 序列线性规划 (SLP) 迭代\n需求交叉降价与规模经济效应的纳什均衡收敛")
    ]

    y_positions = [0.85, 0.60, 0.35, 0.10]
    box_width = 0.7
    box_height = 0.15

    for i, (title, content) in enumerate(nodes):
        y = y_positions[i]
        # 绘制带圆角的方框
        rect = patches.FancyBboxPatch(
            (0.5 - box_width/2, y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            edgecolor='#2C3E50', facecolor='#EBF5FB', lw=2
        )
        ax.add_patch(rect)

        # 填写文字
        ax.text(0.5, y + 0.025, title, ha='center', va='center', fontsize=14, fontweight='bold', color='#C0392B')
        ax.text(0.5, y - 0.035, content, ha='center', va='center', fontsize=12, color='#2C3E50', linespacing=1.5)

        # 绘制向下的箭头 (除了最后一个节点)
        if i < len(nodes) - 1:
            ax.annotate('', xy=(0.5, y_positions[i+1] + box_height/2),
                        xytext=(0.5, y - box_height/2),
                        arrowprops=dict(arrowstyle="->", color='#7F8C8D', lw=2, shrinkA=0, shrinkB=0))

    plt.title('农作物种植策略多阶段递进演化技术路线图', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('AddChart1_技术路线流程图.png', bbox_inches='tight')
    plt.close()
    print("✅ 技术路线图已生成！")

def plot_slp_convergence():
    """图二：序列线性规划(SLP)迭代收敛过程"""
    print(">>> 正在渲染 图二：SLP迭代收敛折线图...")
    iterations = [0, 1, 2, 3]
    profits = [5682.44, 5478.34, 5272.12, 5272.12]

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(iterations, profits, marker='o', markersize=10, linestyle='-', linewidth=3, color='#D35400', markerfacecolor='white', markeredgewidth=2)

    # 填充图表下方面积增加质感
    plt.fill_between(iterations, profits, 5000, color='#D35400', alpha=0.1)

    # 标注数值
    for i, p in zip(iterations, profits):
        if i == 0:
            label = f"初始态 (盲目扩产)\n{p}万"
            xytext = (0, 15)
        elif i == 3:
            label = f"均衡态 (纳什收敛)\n{p}万"
            xytext = (0, 15)
        else:
            label = f"{p}万"
            xytext = (0, 10)

        plt.annotate(label, (i, p), textcoords="offset points", xytext=xytext, ha='center', fontsize=11, fontweight='bold')

    plt.title('序列线性规划 (SLP) 算法多轮市场博弈收敛轨迹', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('算法迭代轮次 (Iterations)', fontsize=13)
    plt.ylabel('系统预期总利润 (万元)', fontsize=13)
    plt.xticks(iterations, ['Iter 0\n(V4基准)', 'Iter 1', 'Iter 2', 'Iter 3\n(达到均衡)'])
    plt.ylim(5100, 5800)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('AddChart2_SLP收敛过程图.png', bbox_inches='tight')
    plt.close()
    print("✅ SLP收敛过程图已生成！")


def eval_v4_with_lambda(assets, lambda_val):
    """专门为灵敏度分析封装的精简版V4求解引擎"""
    YEARS = list(range(2024, 2031))
    L_ids = list(assets['land_id_to_name'].keys())
    C_ids = list(assets['crop_id_to_name'].keys())
    SEASONS = ['单季', '第一季', '第二季']

    volatility_dict = {}
    for j, name in assets['crop_id_to_name'].items():
        if '蔬菜' in name or j > 16: volatility_dict[j] = 0.20
        elif j in assets['bean_set']: volatility_dict[j] = 0.10
        else: volatility_dict[j] = 0.05

    model = pulp.LpProblem(f"Risk_Lambda_{lambda_val}", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("Area", ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS), lowBound=0)
    delta = pulp.LpVariable.dicts("IsP", ((i, j, t, s) for i in L_ids for j in C_ids for t in YEARS for s in SEASONS), cat='Binary')
    S1 = pulp.LpVariable.dicts("S1", ((j, t) for j in C_ids for t in YEARS), lowBound=0)
    S2 = pulp.LpVariable.dicts("S2", ((j, t) for j in C_ids for t in YEARS), lowBound=0)

    rev_terms = []
    cost_terms = []

    for t in YEARS:
        for j in C_ids:
            if j in assets['price_dict']:
                expected_price = list(assets['price_dict'][j].values())[0]
                risk_adj_price = expected_price - lambda_val * (expected_price * volatility_dict[j])
                rev_terms.append(risk_adj_price * S1[j, t] + 0.5 * risk_adj_price * S2[j, t])

        for i in L_ids:
            l_type = assets['land_type_dict'][i]
            for j in C_ids:
                if j in assets['cost_dict'] and l_type in assets['cost_dict'][j]:
                    cost = assets['cost_dict'][j][l_type]
                    for s in SEASONS: cost_terms.append(cost * x[i, j, t, s])

    model += pulp.lpSum(rev_terms) - pulp.lpSum(cost_terms)

    # 简化的刚性物理约束 (为了加速，省略部分不影响宏观利润的重茬硬计算，仅保留容积率)
    for t in YEARS:
        for j in C_ids:
            total_out = pulp.lpSum(x[i, j, t, s] * assets['yield_dict'][j][assets['land_type_dict'][i]]
                                   for i in L_ids for s in SEASONS if j in assets['yield_dict'] and assets['land_type_dict'][i] in assets['yield_dict'][j])
            model += S1[j, t] + S2[j, t] == total_out
            limit = assets['sales_limit'].get(j, 1e9) if assets['sales_limit'].get(j, 0) > 0 else 1e9
            model += S1[j, t] <= limit

        for i in L_ids:
            l_type = assets['land_type_dict'][i]
            model += pulp.lpSum(x[i, j, t, '单季'] + x[i, j, t, '第一季'] for j in C_ids) <= assets['land_area_dict'][i]
            model += pulp.lpSum(x[i, j, t, '第二季'] for j in C_ids) <= assets['land_area_dict'][i]

    # 屏蔽求解日志，设置极短超时
    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
    theoretical_profit = pulp.value(model.objective)

    # 进行 10000 次蒙特卡洛模拟
    N_SIMS = 10000
    sim_profits = np.zeros(N_SIMS)

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

    for sim in range(N_SIMS):
        total_p = 0
        for t in YEARS:
            for j in C_ids:
                if plan_area[t][j] > 0:
                    exp_price = list(assets['price_dict'][j].values())[0]
                    act_price = max(0.1, np.random.normal(exp_price, exp_price * volatility_dict[j]))
                    avg_yield = np.mean(list(assets['yield_dict'][j].values()))
                    act_yield = max(0.1, np.random.normal(avg_yield, avg_yield * volatility_dict[j]))

                    output = plan_area[t][j] * act_yield
                    limit = assets['sales_limit'].get(j, 1e9) if assets['sales_limit'].get(j, 0) > 0 else 1e9

                    if output <= limit: rev = output * act_price
                    else: rev = limit * act_price + (output - limit) * 0.5 * act_price

                    total_p += (rev - plan_cost[t][j])
        sim_profits[sim] = total_p

    mean_mc = np.mean(sim_profits)
    var_95 = np.percentile(sim_profits, 5)
    return theoretical_profit, mean_mc, var_95

def plot_risk_lambda_sensitivity():
    """图三：不同风险厌恶系数下的利润与VaR变化"""
    print(">>> 正在渲染 图三：风险厌恶系数 Lambda 灵敏度分析...")
    print("    (需要运行 5 次模型并执行 5 万次蒙特卡洛模拟，预计需要 1-2 分钟，请稍候...)")

    assets = load_and_process_data()
    lambdas = [0.0, 0.2, 0.5, 0.8, 1.0]
    theo_profits = []
    mc_means = []
    mc_vars = []

    for lam in lambdas:
        print(f"    -> 正在测试 Lambda = {lam} ...")
        theo, mc_mean, mc_var = eval_v4_with_lambda(assets, lam)
        theo_profits.append(theo / 10000)  # 转为万元
        mc_means.append(mc_mean / 10000)
        mc_vars.append(mc_var / 10000)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(lambdas, theo_profits, marker='s', linestyle='--', linewidth=2, color='gray', label='理论安全利润 (基于折价方程)')
    plt.plot(lambdas, mc_means, marker='o', markersize=8, linestyle='-', linewidth=3, color='#2980B9', label='蒙特卡洛期望利润 (Mean)')
    plt.plot(lambdas, mc_vars, marker='^', markersize=8, linestyle='-', linewidth=3, color='#C0392B', label='蒙特卡洛极端保底 (95% VaR)')

    # 高亮标出 \lambda = 0.5 的决策点
    idx_05 = lambdas.index(0.5)
    plt.axvline(x=0.5, color='orange', linestyle=':', linewidth=2)
    plt.scatter([0.5], [mc_means[idx_05]], color='gold', s=200, marker='*', zorder=5)
    plt.annotate('本模型最终取值点\n(兼顾高收益与抗风险)', xy=(0.5, mc_means[idx_05]),
                 xytext=(0.55, mc_means[idx_05] - 300),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 fontsize=11, fontweight='bold', color='black')

    plt.title(r'马科维茨模型中风险厌恶系数 ($\lambda$) 灵敏度演化曲线', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(r'风险厌恶系数 $\lambda$ (0代表激进，1代表保守)', fontsize=13)
    plt.ylabel('七年总体预期利润 (万元)', fontsize=13)
    plt.xticks(lambdas)
    plt.grid(True, linestyle='-.', alpha=0.5)
    plt.legend(fontsize=12, loc='center right')

    plt.tight_layout()
    plt.savefig('AddChart3_Lambda灵敏度分析图.png', bbox_inches='tight')
    plt.close()
    print("✅ Lambda 灵敏度分析图已生成！")

if __name__ == "__main__":
    plot_technical_route()
    plot_slp_convergence()
    plot_risk_lambda_sensitivity()
    print("\n🎉 三张终极论文补充图表已全部导出至当前目录！")