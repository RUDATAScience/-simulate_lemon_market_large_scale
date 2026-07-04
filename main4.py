import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
import os
from google.colab import files

def simulate_dynamic_lemon_market_fast(N=2000, m=4, T=150, alpha=2.1, eta=0.1, gamma=0.08, p_drop=0.3):
    """
    仮説B: 動的ネットワークにおける適応的リンク更新とレモン・エコチェンバーの創発
    （NumPyテンソル演算による超高速化版）
    """
    os.makedirs('output_dynamic', exist_ok=True)
    
    print(f"1. 初期 Scale-Free Network を生成中 (N={N})...")
    G = nx.barabasi_albert_graph(N, m)
    
    initial_degrees = np.array([d for n, d in G.degree()])
    k_90 = np.percentile(initial_degrees, 90)
    k_50 = np.percentile(initial_degrees, 50)
    
    hubs = np.where(initial_degrees >= k_90)[0]
    periphery = np.where(initial_degrees <= k_50)[0]
    middle = np.where((initial_degrees > k_50) & (initial_degrees < k_90))[0]
    
    q = np.random.uniform(0, 1, N)
    P = np.full(N, 0.5)
    
    history_hubs = np.zeros(T)
    history_periph = np.zeros(T)
    history_middle = np.zeros(T)
    history_gcc = np.zeros(T)
    
    print("2. 動的ネットワーク力学系を計算中（行列演算による高速化適用）...")
    for t in range(T):
        # ------------------------------------------------------------
        # 1. 不完全情報下での価格更新フェーズ (行列演算による完全ベクトル化)
        # ------------------------------------------------------------
        current_degrees = np.array([d for n, d in G.degree()])
        mean_k = np.mean(current_degrees) if np.mean(current_degrees) > 0 else 1.0
        
        # 隣接行列の取得 (N x N) と対角成分(自身)の追加
        A = nx.to_numpy_array(G, nodelist=range(N))
        np.fill_diagonal(A, 1)
        
        # 条件マスク: 接続されており、かつ q_j <= P_i であるか (N x N の論理配列)
        valid_mask = A * (q[None, :] <= P[:, None])
        
        # 各ノードの有効な q の和とカウントを一括計算
        sum_q = np.sum(valid_mask * q[None, :], axis=1)
        count_q = np.sum(valid_mask, axis=1)
        
        # 平均品質の計算 (ゼロ割り防止)
        mean_q = np.zeros(N)
        valid_nodes = count_q > 0
        mean_q[valid_nodes] = sum_q[valid_nodes] / count_q[valid_nodes]
        
        # 動的次数に基づくノイズと価格の更新
        sigma = gamma * (mean_k / (current_degrees + 1))
        noise = np.random.normal(0, sigma)
        
        perceived_q = np.maximum(0.0, mean_q + noise)
        B = alpha * perceived_q
        P_new = np.clip(P + eta * (B - P), 0.01, 1.0)
        
        P = P_new
        
        # ------------------------------------------------------------
        # 2. 適応的トポロジー更新フェーズ (一括切断と一括再配線)
        # ------------------------------------------------------------
        edges = np.array(G.edges())
        if len(edges) > 0:
            u = edges[:, 0]
            v = edges[:, 1]
            
            # 互いの品質が相手の要求を満たさない条件を一括評価
            dissatisfied = (q[u] < P[v] * 0.8) | (q[v] < P[u] * 0.8)
            
            # 確率的切断のマスク
            drop_mask = dissatisfied & (np.random.rand(len(edges)) < p_drop)
            edges_to_drop = edges[drop_mask]
            
            if len(edges_to_drop) > 0:
                # 一括切断
                G.remove_edges_from(edges_to_drop)
                
                # 再配線対象のノードリスト展開
                nodes_to_rewire = edges_to_drop.flatten()
                num_rewire = len(nodes_to_rewire)
                
                # ランダム候補を一括生成 (num_rewire x 5)
                candidates = np.random.randint(0, N, size=(num_rewire, 5))
                P_candidates = P[candidates]
                P_target = P[nodes_to_rewire].reshape(-1, 1)
                
                # Pの差分が最小の候補を一括特定
                best_idx = np.argmin(np.abs(P_candidates - P_target), axis=1)
                best_candidates = candidates[np.arange(num_rewire), best_idx]
                
                # 自己ループを除外して一括再配線
                valid = nodes_to_rewire != best_candidates
                new_edges = np.column_stack((nodes_to_rewire[valid], best_candidates[valid]))
                G.add_edges_from(new_edges)
        
        # ------------------------------------------------------------
        # 3. 記録フェーズ
        # ------------------------------------------------------------
        history_hubs[t] = np.mean(P[hubs])
        history_middle[t] = np.mean(P[middle])
        history_periph[t] = np.mean(P[periphery])
        
        if len(G.edges()) > 0:
            # 巨大連結成分のみ抽出して割合を計算
            gcc_size = len(max(nx.connected_components(G), key=len))
        else:
            gcc_size = 1
        history_gcc[t] = gcc_size / N

    print("3. 結果の保存とグラフ描画...")
    df = pd.DataFrame({
        'TimeStep': range(T),
        'Hubs_InitTop10': history_hubs,
        'Middle': history_middle,
        'Periph_InitBottom50': history_periph,
        'GCC_Ratio': history_gcc
    })
    csv_path = 'output_dynamic/dynamic_lemon_market.csv'
    df.to_csv(csv_path, index=False)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.plot(df['TimeStep'], df['Hubs_InitTop10'], label='Initial Hubs', color='#d62728', linewidth=2)
    ax1.plot(df['TimeStep'], df['Middle'], label='Initial Middle', color='#2ca02c', linewidth=2)
    ax1.plot(df['TimeStep'], df['Periph_InitBottom50'], label='Initial Periphery', color='#1f77b4', linewidth=2)
    ax1.set_title('Price Dynamics with Adaptive Rewiring', fontsize=12)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mean Local Price')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(df['TimeStep'], df['GCC_Ratio'], color='purple', linewidth=2)
    ax2.set_title('Collapse of Giant Connected Component', fontsize=12)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Fraction of Nodes in GCC')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(0, 1.05)
    
    ax3.scatter(q, P, alpha=0.4, c=P, cmap='coolwarm', s=15)
    ax3.set_title('Emergence of Echo Chambers (t=T)', fontsize=12)
    ax3.set_xlabel('True Quality (q)')
    ax3.set_ylabel('Final Price (P)')
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    png_path = 'output_dynamic/dynamic_lemon_plot.png'
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    zip_filename = 'dynamic_lemon_results.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(csv_path, arcname='dynamic_lemon_market.csv')
        zipf.write(png_path, arcname='dynamic_lemon_plot.png')
        
    return zip_filename

zip_file = simulate_dynamic_lemon_market_fast()
files.download(zip_file)
print(f"完了: {zip_file} を作成・ダウンロードしました。")
