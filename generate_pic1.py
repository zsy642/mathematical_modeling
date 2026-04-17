import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 配置顶级学术图表的字体与清晰度
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] # 支持中文
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font=['SimHei', 'Microsoft YaHei'])

def load_result_data(file_path='result1_2_降价输出.xlsx'):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件 {file_path}，请确保文件名正确。")
        return None

    print(f">>> 正在解析优化结果文件: {file_path}")
    all_data = []
    # 遍历 2024 到 2030 的所有 Sheet
    for year in range(2024, 2031):
        df = pd.read_excel(file_path, sheet_name=str(year))
        # 宽表转长表 (融合所有的作物列)
        df_melt = df.melt(id_vars=['季次', '地块名'], var_name='作物名', value_name='种植面积')
        df_melt['年份'] = year
        df_melt = df_melt[df_melt['种植面积'] > 0] # 只保留实际种植的记录
        all_data.append(df_melt)

    return pd.concat(all_data, ignore_index=True)

def categorize_crop(crop_name):
    """为作物打上学术标签，用于图表分组"""
    beans = ['黄豆', '黑豆', '红豆', '绿豆', '爬豆', '豇豆', '刀豆', '芸豆']
    grains = ['小麦', '玉米', '谷子', '高粱', '黍子', '荞麦', '南瓜', '红薯', '莜麦', '大麦', '水稻']
    if crop_name in beans:
        return '豆类 (固氮轮作)'
    elif crop_name in grains:
        return '基础粮食 (主粮)'
    elif '菇' in crop_name or '菌' in crop_name:
        return '食用菌 (大棚高附加值)'
    else:
        return '经济蔬菜 (高风险/高收益)'

def plot_macro_structure(df_all):
    print(">>> 正在渲染 图一：宏观种植结构演变图...")
    df_all['作物大类'] = df_all['作物名'].apply(categorize_crop)

    # 聚合每年各大类的总面积
    yearly_structure = df_all.groupby(['年份', '作物大类'])['种植面积'].sum().unstack().fillna(0)

    # 使用学术级配色方案
    colors = ['#E6B0AA', '#1F77B4', '#2CA02C', '#FF7F0E']

    ax = yearly_structure.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors, alpha=0.85)
    plt.title('2024-2030年 乡村总体种植结构演化轨迹 (资源配置动态平衡)', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('规划年份', fontsize=12)
    plt.ylabel('投入种植面积 (亩)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='作物战略分组', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig('Chart1_宏观种植结构.png', dpi=300)
    plt.close()

def plot_micro_rotation_heatmap(df_all):
    print(">>> 正在渲染 图二：典型地块微观轮作合规热力图...")
    # 截取 A1 到 A6 的平旱地进行特写展示
    target_lands = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    df_sub = df_all[(df_all['地块名'].isin(target_lands)) & (df_all['季次'] == '单季')].copy()

    # 选出面积最大的主导作物代表该地块当年的种植属性
    df_dominant = df_sub.loc[df_sub.groupby(['年份', '地块名'])['种植面积'].idxmax()]
    df_dominant['是否豆类'] = df_dominant['作物名'].apply(lambda x: 1 if x in ['黄豆', '黑豆', '红豆', '绿豆', '爬豆'] else 0)

    pivot_bean = df_dominant.pivot(index='地块名', columns='年份', values='是否豆类').fillna(0)
    pivot_name = df_dominant.pivot(index='地块名', columns='年份', values='作物名').fillna('未种植')

    plt.figure(figsize=(10, 4))
    # 自定义离散色卡：0为非豆类(淡灰色)，1为豆类(护眼绿色)
    cmap = sns.color_palette(["#EAECEE", "#2ECC71"])

    ax = sns.heatmap(pivot_bean, annot=pivot_name, fmt="", cmap=cmap, cbar=False,
                     linewidths=1, linecolor='white', annot_kws={"size": 11, "weight": "bold"})

    plt.title('A区典型旱地轮作状态实录 (绿色高亮代表完成“三年一豆”固氮休耕)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('规划年份', fontsize=12)
    plt.ylabel('典型地块编号', fontsize=12)
    plt.tight_layout()

    plt.savefig('Chart2_微观轮作合规矩阵.png', dpi=300)
    plt.close()

def plot_economic_top_crops(df_all):
    print(">>> 正在渲染 图三：核心经济作物规模排序图...")
    # 统计7年来总投入面积排名前10的作物
    top_crops = df_all.groupby('作物名')['种植面积'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(11, 5))
    # 绘制横向条形图，极具视觉冲击力
    sns.barplot(x=top_crops.values, y=top_crops.index, palette="viridis")

    plt.title('2024-2030年 累计投入土地规模 Top10 核心经济作物', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('七年累计总种植面积 (亩)', fontsize=12)
    plt.ylabel('作物名称', fontsize=12)

    # 给柱子加上具体数值标签
    for i, v in enumerate(top_crops.values):
        plt.text(v + 10, i, f"{v:.0f} 亩", va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.savefig('Chart3_核心作物规模分布.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    df_result = load_result_data('result1_2_降价输出.xlsx')
    if df_result is not None:
        plot_macro_structure(df_result)
        plot_micro_rotation_heatmap(df_result)
        plot_economic_top_crops(df_result)
        print("\n✅ 所有高质量国赛图表已生成完毕，请查收当前目录的 PNG 文件！")