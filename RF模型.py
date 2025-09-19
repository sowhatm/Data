import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path  # 用于文件夹管理

# 设置全局随机种子（可选）
np.random.seed(None)

def load_and_validate_data(file_path):
    """加载并验证数据"""
    try:
        data = pd.read_excel(file_path)
        if data.empty:
            raise ValueError("数据文件为空")
        if data.shape[1] < 3:
            raise ValueError("数据至少需要3列：CIF文件名、特征和目标值")
        return data
    except Exception as e:
        raise Exception(f"加载数据时出错: {str(e)}")

def main():
    try:
        data = load_and_validate_data('石榴石数据集.xlsx')
    except Exception as e:
        print(e)
        return

    # 数据准备
    cif_names = data.iloc[:, 0]
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]

    if X.isnull().any().any() or y.isnull().any():
        print("警告：数据中存在缺失值，将被移除")
        mask = ~X.isnull().any(axis=1) & ~y.isnull()
        X, y, cif_names = X[mask], y[mask], cif_names[mask]

    times = 10
    best_metrics = {
        'mae_test': float('inf'),
        'r2_test': -float('inf'),
        'mae_train': float('inf'),
        'r2_train': -float('inf'),
        'model': None,
        'split': None
    }

    for i in range(times):
        X_train, X_test, y_train, y_test, cif_train, cif_test = train_test_split(
            X, y, cif_names, test_size=0.2
        )
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        rf.fit(X_train, y_train)

        metrics = {
            'mae_train': mean_absolute_error(y_train, rf.predict(X_train)),
            'r2_train': r2_score(y_train, rf.predict(X_train)),
            'mae_test': mean_absolute_error(y_test, rf.predict(X_test)),
            'r2_test': r2_score(y_test, rf.predict(X_test))
        }

        print(f"Iteration {i+1}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        if metrics['r2_test'] > best_metrics['r2_test']:
            best_metrics.update(metrics)
            best_metrics['model'] = rf
            best_metrics['split'] = (X_train, X_test, y_train, y_test, cif_train, cif_test)

    # === 新增：创建保存文件夹 === #
    result_folder = Path(f"results_r2_{best_metrics['r2_test']:.4f}_mae_{best_metrics['mae_test']:.4f}")  # NEW
    result_folder.mkdir(exist_ok=True)  # NEW

    # 保存模型
    joblib.dump(best_metrics['model'], result_folder / 'best_rf_model.pkl')  # NEW

    X_train, X_test, y_train, y_test, cif_train, cif_test = best_metrics['split']
    y_pred_train = best_metrics['model'].predict(X_train)
    y_pred_test = best_metrics['model'].predict(X_test)

    # 保存散点图数据
    for data, filename in [
        ((cif_train, y_train, y_pred_train), result_folder / 'train_scatter_data.xlsx'),  # NEW
        ((cif_test, y_test, y_pred_test), result_folder / 'test_scatter_data.xlsx')  # NEW
    ]:
        pd.DataFrame({
            'CIF_File': data[0].values,
            'True': data[1],
            'Predicted': data[2]
        }).to_excel(filename, index=False)

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_train, y=y_pred_train, alpha=0.5, label='Training Data')
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5, label='Testing Data')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Best Iteration (R²_test: {best_metrics["r2_test"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(result_folder / 'scatter_plot.png')  # NEW
    plt.show()

    # 特征重要性
    importances = best_metrics['model'].feature_importances_
    feature_names = X.columns
    importance_data = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_data.to_excel(result_folder / 'importance_data.xlsx', index=False)  # NEW

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_data.sort_values('Importance'))
    plt.title('Feature Importances')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(result_folder / 'feature_importance.png')  # NEW
    plt.show()

    print("\nBest Results:")
    for key, value in best_metrics.items():
        if key not in ['model', 'split']:
            print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
