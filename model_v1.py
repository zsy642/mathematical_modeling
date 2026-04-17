import pulp
from data_loader import load_and_process_data

def build_and_solve_v1():
    print(">>> 正在加载基础数据资产...")
    assets = load_and_process_data()

    land_id_to_name = {v: k for k, v in assets['land_name_to_id'].items()}
    crop_id_to_name = assets['crop_id_to_name']
    land_area = assets['land_area_dict']
    land_type = assets['land_type_dict']
    profit_dict = assets['profit_dict']

    # 提取地块和作物的ID列表
    L_ids = list(land_id_to_name.keys())
    C_ids = list(crop_id_to_name.keys())

    print(f"\n>>> 开始构建 V1 单年静态优化模型 (2024年)...")
    # 1. 初始化最大化模型
    model = pulp.LpProblem("Maximize_Profit_2024", pulp.LpMaximize)

    # 2. 定义决策变量
    # x[i, j] 表示在地块 i 种植作物 j 的面积
    x = pulp.LpVariable.dicts("Area",
                              ((i, j) for i in L_ids for j in C_ids),
                              lowBound=0,
                              cat='Continuous')

    # delta[i, j] 表示是否在地块 i 种植作物 j (0-1变量)
    delta = pulp.LpVariable.dicts("Is_Planted",
                                  ((i, j) for i in L_ids for j in C_ids),
                                  cat='Binary')

    # 3. 构建目标函数：总利润最大化 (V1 暂不考虑滞销量上限，验证纯粹的算力)
    objective_terms = []
    for i in L_ids:
        l_type = land_type[i]
        for j in C_ids:
            # 只有当该作物能在这个地块类型上种植时，才有利润系数
            if j in profit_dict and l_type in profit_dict[j]:
                unit_profit = profit_dict[j][l_type]
                objective_terms.append(unit_profit * x[i, j])
            else:
                # 适用性硬约束：如果不能种，强制面积为0
                model += x[i, j] == 0, f"Suitability_Constraint_{i}_{j}"

    model += pulp.lpSum(objective_terms), "Total_Profit"

    # 4. 添加约束条件
    for i in L_ids:
        # 约束 A: 地块面积不超标 (这里暂时简化为单季总面积，V2再拆分季次)
        model += pulp.lpSum(x[i, j] for j in C_ids) <= land_area[i], f"Max_Area_Land_{i}"

        # 约束 B: Big-M 逻辑约束，面积和0-1变量绑定 (M为该地块最大面积)
        for j in C_ids:
            model += x[i, j] <= land_area[i] * delta[i, j], f"BigM_Link_{i}_{j}"

    print(">>> 模型构建完毕，启动 PuLP (CBC) 求解器...")
    # 5. 求解模型
    model.solve(pulp.PULP_CBC_CMD(msg=True)) # msg=True 会在控制台打印求解过程

    # 6. 解析结果
    status = pulp.LpStatus[model.status]
    print(f"\n✅ 求解状态: {status}")
    print(f"💰 V1理论最大总利润: {pulp.value(model.objective):.2f} 元")

    # 打印非零的种植决策
    print("\n--- 部分种植决策方案截取 ---")
    count = 0
    for i in L_ids:
        for j in C_ids:
            if x[i, j].varValue and x[i, j].varValue > 0.01: # 过滤掉极小的浮点误差
                print(f"地块: {land_id_to_name[i]:<4} | 类型: {land_type[i]:<4} | 作物: {crop_id_to_name[j]:<6} | 面积: {x[i, j].varValue:>5.1f} 亩")
                count += 1
                if count >= 10: # 只展示前10条防刷屏
                    print("... (其余省略)")
                    return

if __name__ == "__main__":
    build_and_solve_v1()