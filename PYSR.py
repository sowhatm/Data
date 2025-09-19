import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sympy
from sympy import symbols, latex
from sympy.printing.mathml import mathml
import os

# ==================== 自定义变量符号 ==================== #
F1, F2, F3, F4, F5, F6, target = symbols("F1 F2 F3 F4 F5 F6 target")

# ==================== 构造 F2-F5 的组合特征 ==================== #
def transform_f2f5(df):
    new = pd.DataFrame()
    F2, F3, F4, F5 = df["F2"], df["F3"], df["F4"], df["F5"]

    # 基础运算
    new["F2_plus_F3"] = F2 + F3
    new["F2_plus_F4"] = F2 + F4
    new["F2_plus_F5"] = F2 + F5
    new["F3_plus_F4"] = F3 + F4
    new["F3_plus_F5"] = F3 + F5
    new["F4_plus_F5"] = F4 + F5

    new["F2_minus_F3"] = F2 - F3
    new["F2_minus_F4"] = F2 - F4
    new["F2_minus_F5"] = F2 - F5
    new["F3_minus_F4"] = F3 - F4
    new["F3_minus_F5"] = F3 - F5
    new["F4_minus_F5"] = F4 - F5

    new["F2_mul_F3"] = F2 * F3
    new["F2_mul_F4"] = F2 * F4
    new["F2_mul_F5"] = F2 * F5
    new["F3_mul_F4"] = F3 * F4
    new["F3_mul_F5"] = F3 * F5
    new["F4_mul_F5"] = F4 * F5

    new["F2_div_F3"] = F2 / F3.replace(0, 1e-6)
    new["F2_div_F4"] = F2 / F4.replace(0, 1e-6)
    new["F2_div_F5"] = F2 / F5.replace(0, 1e-6)
    new["F3_div_F4"] = F3 / F4.replace(0, 1e-6)
    new["F3_div_F5"] = F3 / F5.replace(0, 1e-6)
    new["F4_div_F5"] = F4 / F5.replace(0, 1e-6)

    # 幂次与开方
    for col in ["F2", "F3", "F4", "F5"]:
        new[f"{col}_squared"] = df[col] ** 2
        new[f"{col}_sqrt"] = np.sqrt(np.abs(df[col]) + 1e-6)

    return new

# ==================== 按 sheet 进行交叉验证 ==================== #
def cross_validate_excel(filepath):
    xls = pd.ExcelFile(filepath)
    metrics_train, metrics_test = [], []

    if not os.path.exists("cv_results"):
        os.makedirs("cv_results")

    for fold, sheet_name in enumerate(xls.sheet_names, start=1):
        print(f"\n===== Fold {fold} ({sheet_name}) =====")
        df = pd.read_excel(filepath, sheet_name=sheet_name).dropna()

        # 按 80/20 划分
        split_idx = int(len(df) * 0.8)
        df_train, df_test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

        # 构造 F2-F5 组合特征
        X_lift_train = transform_f2f5(df_train)
        X_lift_test = transform_f2f5(df_test)

        # PySR 拟合 f(F2,F3,F4,F5)
        model_f = PySRRegressor(
            niterations=300,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "square", "abs", "cube", "inv"],
            extra_sympy_mappings={
                "cube": lambda x: x**3,
                "inv": lambda x: 1 / x,
            },
            model_selection="best",
            maxsize=25,
            verbosity=0,
            procs=0,
            random_state=42,
        )
        model_f.fit(X_lift_train, df_train["target"])

        f_train = model_f.predict(X_lift_train)
        f_test = model_f.predict(X_lift_test)

        # 把 PySR 结果添加到 df
        df_train["f_F2F5"] = f_train
        df_test["f_F2F5"] = f_test

        # === 保存训练集和测试集到 Excel ===
        save_path = f"cv_results2/fold{fold}_{sheet_name}.xlsx"
        with pd.ExcelWriter(save_path) as writer:
            df_train.to_excel(writer, sheet_name="train", index=False)
            df_test.to_excel(writer, sheet_name="test", index=False)
        print(f"[数据保存] 训练集和测试集已保存到 {save_path}")

        # 最终回归输入：F1, f(F2,F3,F4,F5), F6
        X_final_train = df_train[["F1", "f_F2F5", "F6"]]
        X_final_test = df_test[["F1", "f_F2F5", "F6"]]

        # 线性回归
        lr = LinearRegression()
        lr.fit(X_final_train, df_train["target"])

        a, b, c = lr.coef_
        d = lr.intercept_

        # 表达式保存
        best_eq = model_f.get_best()
        expr = best_eq["sympy_format"]

        # 保存 LaTeX
        latex_str = latex(expr)
        with open(f"cv_results/fold{fold}_f_F2F5.tex", "w", encoding="utf-8") as f:
            f.write(f"\\[\n f(F_2,F_3,F_4,F_5) = {latex_str} \\]\n")

        # 保存 MathML
        full_expr = a * F1 + b * expr + c * F6 + d
        target_eq = sympy.Eq(target, full_expr)
        mathml_str = mathml(target_eq, printer="presentation")
        with open(f"cv_results/fold{fold}_model.xml", "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<math xmlns="http://www.w3.org/1998/Math/MathML">\n')
            f.write(mathml_str)
            f.write('\n</math>')

        print("[PySR] 最佳 f(F2,F3,F4,F5) 表达式：", expr)
        print(f"[最终线性模型] target = {a:.4f}*F1 + {b:.4f}*f(F2,F3,F4,F5) + {c:.4f}*F6 + {d:.4f}")

        # 评估
        y_train_pred = lr.predict(X_final_train)
        y_test_pred = lr.predict(X_final_test)

        train_rmse = np.sqrt(mean_squared_error(df_train["target"], y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(df_test["target"], y_test_pred))
        train_r2 = r2_score(df_train["target"], y_train_pred)
        test_r2 = r2_score(df_test["target"], y_test_pred)
        train_mae = mean_absolute_error(df_train["target"], y_train_pred)
        test_mae = mean_absolute_error(df_test["target"], y_test_pred)

        print(f"[训练集] R² = {train_r2:.4f}, RMSE = {train_rmse:.4f}, MAE = {train_mae:.4f}")
        print(f"[验证集] R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f}")

        metrics_train.append((train_r2, train_rmse, train_mae))
        metrics_test.append((test_r2, test_rmse, test_mae))

    # 平均结果
    avg_train = np.mean(metrics_train, axis=0)
    avg_test = np.mean(metrics_test, axis=0)
    print("\n===== 五折交叉验证平均结果 =====")
    print(f"训练集: R²={avg_train[0]:.4f}, RMSE={avg_train[1]:.4f}, MAE={avg_train[2]:.4f}")
    print(f"验证集: R²={avg_test[0]:.4f}, RMSE={avg_test[1]:.4f}, MAE={avg_test[2]:.4f}")


# ==================== 运行 ==================== #
cross_validate_excel("data.xlsx")
