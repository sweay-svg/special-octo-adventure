# 二手机动车交易价格影响因素及定价规律探析 —— 分析说明

## 1. 数据与任务
- `train.csv`：训练集 150,000 条（含真实价格 `price`），用于训练回归模型 + 5 折交叉验证
- `test.csv`：测试集 50,000 条（不含价格），用于模型预测
- `result.csv`：验证集 50,000 条（测试集对应的真实价格），用于与预测结果对比、计算各项误差指标
- 数据格式：`train.csv`/`test.csv` 为空格分隔，`result.csv` 为逗号分隔
- 字段：SaleID、name、regDate(注册日期)、model、brand、bodyType、fuelType、gearbox、
  power(功率)、kilometer(公里数)、notRepairedDamage、regionCode、seller、offerType、
  creatDate(挂牌日期)、price(目标)、v_0~v_14(15 个匿名特征)

## 2. 数据质量问题及处理
1. **分隔符**：训练/测试集为空白分隔，验证集为逗号分隔；
2. **负价格脏数据**：训练集中 1,571 条 (1.05%) 价格为负值，予以剔除；
3. **缺失值**：`power`/`kilometer`/`gearbox` 含 `-`，`notRepairedDamage` 含 `-`(约 11.8%)，
   `v_12~v_14` 含 NaN → 统一转为 NaN 后用训练集中位数填充，并保留"是否缺失"标记特征；
4. **字段语义变化**：本数据集中 `fuelType`/`gearbox`/`seller`/`offerType` 已变换为数值型
   （如 offerType 有 1,898 种连续取值、gearbox 296 种取值且含缺失），故按数值特征处理；
   `regionCode` 为 7,886 类高基数 ID，按类别特征处理。

## 3. 特征工程（34 个建模特征）
- 日期解析：注册日期、挂牌日期 → 车龄 `used_days`/`age_years`、注册年份/月份、挂牌月份；
- 名称计数特征 `name_count`；
- 交互特征 `power_per_model`（单位型号的功率）；
- 缺失标记特征 `nrd_na`、`gearbox_na`；
- 类别编码：`model/brand/bodyType/regionCode` 统一整数编码（未见类别=-1）。

## 4. 建模方案
- 目标变换：`log1p(price)` 训练、预测后 `expm1` 还原；
- 5 折交叉验证评估泛化能力，全量训练后预测测试集；
- 模型：Ridge 线性回归、随机森林(RF)、HistGradientBoosting(HistGB)、
  LightGBM、XGBoost；
- 集成：按 1/CV-MAE 加权平均各模型预测。

## 5. 评价指标（测试集预测 vs 验证集真实价格）
MAE（平均绝对误差）、MSE、RMSE（均方根误差）、MAPE（平均绝对百分比误差）、
SMAPE（对称平均绝对百分比误差）、R²（决定系数）、中位绝对误差、P90 绝对误差、
±5%/±10%/±20% 内准确率、平均偏差（系统性高估/低估）。

## 6. 实验结果

### 6.1 模型对比（测试集预测 vs 验证集 result.csv）

| 模型 | 5折CV MAE(元) | 验证集MAE(元) | RMSE(元) | MAPE(%) | R² | ±10%内准确率 |
|---|---|---|---|---|---|---|
| Ridge(线性) | 906.28 | **649.42** | **1806.84** | **15.79** | **0.9429** | **68.42%** |
| RF(随机森林) | 581.60 | 928.99 | 2130.77 | 21.25 | 0.9205 | 48.24% |
| HistGB | 584.19 | 907.13 | 2134.37 | 20.35 | 0.9203 | 49.71% |
| LGB(LightGBM) | **555.20** | 966.81 | 2252.53 | 21.79 | 0.9112 | 47.36% |
| XGB(XGBoost) | 633.68 | 893.65 | 2136.26 | 19.82 | 0.9201 | 50.66% |
| 加权集成 | 565.13 | 829.86 | 2021.09 | 18.82 | 0.9285 | 55.57% |

### 6.2 主要发现
1. **价格影响因素排序**（Spearman 相关）：匿名特征 v_0 (+0.917)、v_12 (+0.830)、
   v_3 (-0.800)、v_8 (+0.664) 与价格相关性最强；可解释特征中 **功率 power (+0.606)**
   居首，其后为 offerType (-0.481)、fuelType (+0.270)、bodyType (+0.208)、
   seller (-0.200)、公里数 kilometer (-0.194)；特征重要性上 regionCode(地区)、
   used_days(车龄)、name_count(车型热度)、model、power 等排在前列；
2. **训练集与测试集存在分布漂移**：树模型在交叉验证中 MAE 仅 555~634 元，
   但测试集上 MAE 升到 894~967 元；线性模型 Ridge 相反，在测试集上表现最好
   （MAE 649 元、R² 0.943），说明线性定价规律跨分布更稳健；
3. 所有模型预测整体**偏低 315~360 元**（平均偏差为负），存在轻微系统性低估；
4. 数据中有 1.05% 负价格脏样本、约 11.8% 的 notRepairedDamage 缺失('-')，
   已做清洗与缺失值处理。

## 7. 运行方式
```
python used_car_analysis.py
```
依赖：pandas、numpy、scikit-learn、scipy、matplotlib、lightgbm、xgboost
（本机安装在 `site-packages/`，运行时需将 `E:\Code\数据挖掘\site-packages` 加入 PYTHONPATH）。

## 8. 输出文件（output/ 目录）
- `predictions_test.csv` —— 测试集逐条预测（各模型 + 加权集成 + 验证集真实价格）
- `metrics_report.csv` —— 各模型与集成模型的全套误差指标
- `feature_importance.csv` —— 定价影响因素重要性排序
- `factor_analysis.csv` —— 各特征与价格的 Spearman 相关系数及显著性
- `group_means.csv` —— 品牌/车身/燃油类型分组均价
- `plots/` —— 价格分布、相关性热图、散点图、分组均价图、特征重要性图、预测vs真实、残差分布图
