import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
import os
from google.colab import files

def simulate_perfect_info_market(N=5000, m=3, T=200, alpha=2.1, eta=0.1):
    """
    仮説A: 完全情報状態におけるネットワーク・トポロジーからの解放と分離均衡の検証
    """
    os.makedirs('output_perfect_info', exist_ok=True)
    
    print(f"1. Scale-Free Network を生成中 (N={N})...")
    G = nx.barabasi_albert_graph(N, m)
    degrees = np.array([G.degree(i) for i in range(N)])
    
    # ネットワーク中心性（次数）によるエージェントの分類
    k_90 = np.percentile(degrees, 90) # 上位10% (ハブ)
    k_50 = np.percentile(degrees, 50) # 下位50% (ペリフェリ)
    
    hubs = np.where(degrees >= k_90)[0]
    periphery = np.where(degrees <= k_50)[0]
    middle = np.where((degrees > k_50) & (degrees < k_90))[0]
    
    # 品質（q_i）と価格（P_i）の初期化
    q = np.random.uniform(0, 1, N)
    P = np.full(N, 0.5)
    
    history_hubs = np.zeros(T)
    history_periph = np.zeros(T)
    history_middle = np.zeros(T)
    
    print("2. 完全情報状態（分離均衡）の力学系シミュレーションを実行中...")
    for t in range(T):
        # 完全情報状態：ネットワーク上の平均化を介さず、真の品質 q_i に基づいて入札が行われる
        B = alpha * q
        
        # 価格の更新（トポロジーからの解放）
        P = np.clip(P + eta * (B - P), 0.01, 1.0)
        
        # 各階層の平均価格を記録
        history_hubs[t] = np.mean(P[hubs])
        history_middle[t] = np.mean(P[middle])
        history_periph[t] = np.mean(P[periphery])

    print("3. 結果の保存とグラフ描画...")
    df = pd.DataFrame({
        'TimeStep': range(T),
        'Hubs_Top10%': history_hubs,
        'Middle': history_middle,
        'Periphery_Bottom50%': history_periph
    })
    csv_path = 'output_perfect_info/perfect_info_dynamics.csv'
    df.to_csv(csv_path, index=False)
    
    # グラフの描画 (3パネルで仮説を完全検証)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # パネル1: 時系列（階層的乖離の消失を確認）
    ax1.plot(df['TimeStep'], df['Hubs_Top10%'], label='Hubs', color='#d62728', linewidth=3, alpha=0.7)
    ax1.plot(df['TimeStep'], df['Middle'], label='Middle', color='#2ca02c', linewidth=2, linestyle='-.')
    ax1.plot(df['TimeStep'], df['Periphery_Bottom50%'], label='Periphery', color='#1f77b4', linewidth=2, linestyle=':')
    ax1.set_title('Price Dynamics by Centrality (Decoupled)', fontsize=12)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mean Local Price')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # パネル2: 次数 vs 最終価格（相関関係の崩壊を確認）
    ax2.scatter(degrees, P, alpha=0.3, color='purple', s=10)
    ax2.set_xscale('log')
    ax2.set_title('Final Price vs. Node Degree (t=T)', fontsize=12)
    ax2.set_xlabel('Degree (log scale)')
    ax2.set_ylabel('Final Price')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # パネル3: 品質 vs 最終価格（分離均衡の到達を確認）
    ax3.scatter(q, P, alpha=0.3, color='orange', s=10)
    ax3.set_title('Final Price vs. True Quality (Separating Eq)', fontsize=12)
    ax3.set_xlabel('True Quality (q)')
    ax3.set_ylabel('Final Price')
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    png_path = 'output_perfect_info/perfect_info_plot.png'
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    # ZIP化
    zip_filename = 'perfect_info_results.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(csv_path, arcname='perfect_info_dynamics.csv')
        zipf.write(png_path, arcname='perfect_info_plot.png')
        
    return zip_filename

# 実行とダウンロード
zip_file = simulate_perfect_info_market()
files.download(zip_file)
print(f"完了: {zip_file} を作成・ダウンロードしました。")
