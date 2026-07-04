import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
import os
from google.colab import files

def simulate_network_lemon_market(N=5000, m=3, T=200, alpha=2.1, eta=0.1, gamma=0.08):
    """
    スケールフリー・ネットワーク上の局所的レモン市場シミュレーション
    """
    os.makedirs('output_network', exist_ok=True)
    
    print(f"1. Scale-Free Network を生成中 (N={N})...")
    G = nx.barabasi_albert_graph(N, m)
    degrees = np.array([G.degree(i) for i in range(N)])
    
    # 中心性（次数）によるエージェントの分類
    k_90 = np.percentile(degrees, 90) # 上位10% (ハブ)
    k_50 = np.percentile(degrees, 50) # 下位50% (ペリフェリ)
    
    hubs = np.where(degrees >= k_90)[0]
    periphery = np.where(degrees <= k_50)[0]
    middle = np.where((degrees > k_50) & (degrees < k_90))[0]
    
    # 高速化のための隣接リスト抽出（自身も含む）
    adj = [np.array(list(G.neighbors(i)) + [i]) for i in range(N)]
    
    # 品質の初期化と価格の初期化
    q = np.random.uniform(0, 1, N)
    P = np.full(N, 0.5)
    
    # ノイズスケーリング（中心性が高いほどノイズが小さい）
    mean_k = np.mean(degrees)
    sigma = gamma * (mean_k / (degrees + 1))
    
    # 履歴保存用配列
    history_hubs = np.zeros(T)
    history_periph = np.zeros(T)
    history_middle = np.zeros(T)
    
    print("2. 力学系シミュレーションを実行中...")
    for t in range(T):
        P_new = np.zeros(N)
        for i in range(N):
            neighbors = adj[i]
            # 自身の提示価格以下の品質を持つ隣接エージェントのみが市場に参加
            active_q = q[neighbors][q[neighbors] <= P[i]]
            
            if len(active_q) == 0:
                mean_q = 0.0
            else:
                mean_q = np.mean(active_q)
            
            # 中心性に依存した局所ノイズ
            noise = np.random.normal(0, sigma[i])
            perceived_q = max(0.0, mean_q + noise)
            
            # 入札価格と価格更新
            B = alpha * perceived_q
            P_new[i] = max(0.01, min(1.0, P[i] + eta * (B - P[i])))
            
        P = P_new
        # 各階層の平均価格を記録
        history_hubs[t] = np.mean(P[hubs])
        history_middle[t] = np.mean(P[middle])
        history_periph[t] = np.mean(P[periphery])

    print("3. 結果の保存とグラフ描画...")
    # CSVの保存
    df = pd.DataFrame({
        'TimeStep': range(T),
        'Hubs_Top10%': history_hubs,
        'Middle': history_middle,
        'Periphery_Bottom50%': history_periph
    })
    csv_path = 'output_network/network_lemon_dynamics.csv'
    df.to_csv(csv_path, index=False)
    
    # グラフの描画 (2パネル)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # パネル1: 時系列ダイナミクス
    ax1.plot(df['TimeStep'], df['Hubs_Top10%'], label='Hubs (High Centrality)', color='#d62728', linewidth=2)
    ax1.plot(df['TimeStep'], df['Middle'], label='Middle', color='#2ca02c', linewidth=2)
    ax1.plot(df['TimeStep'], df['Periphery_Bottom50%'], label='Periphery (Low Centrality)', color='#1f77b4', linewidth=2)
    ax1.set_title('Price Dynamics by Network Centrality', fontsize=14)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Mean Local Price', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # パネル2: 次数 vs 最終価格の散布図
    ax2.scatter(degrees, P, alpha=0.3, color='purple', edgecolors='none', s=20)
    ax2.set_xscale('log')
    ax2.set_title('Final Price vs. Node Degree (t=T)', fontsize=14)
    ax2.set_xlabel('Degree (log scale)', fontsize=12)
    ax2.set_ylabel('Final Price', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    png_path = 'output_network/network_lemon_plot.png'
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    # ZIP化
    zip_filename = 'network_lemon_results.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(csv_path, arcname='network_lemon_dynamics.csv')
        zipf.write(png_path, arcname='network_lemon_plot.png')
        
    return zip_filename

# 実行とダウンロード
zip_file = simulate_network_lemon_market()
files.download(zip_file)
print(f"完了: {zip_file} を作成・ダウンロードしました。")
