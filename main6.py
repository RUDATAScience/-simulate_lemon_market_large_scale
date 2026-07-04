import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
import os
from google.colab import files

def run_intervention_strategy(strategy, N=2000, m=4, T=150, alpha=2.1, eta=0.1, gamma=0.08, p_drop=0.3):
    """
    指定された介入戦略に基づく動的ネットワークシミュレーション
    strategy: 'hub' (S1), 'periphery' (S2), 'random_edge' (S3)
    """
    G = nx.barabasi_albert_graph(N, m)
    initial_degrees = np.array([d for n, d in G.degree()])
    
    k_90 = np.percentile(initial_degrees, 90)
    k_50 = np.percentile(initial_degrees, 50)
    
    hubs = np.where(initial_degrees >= k_90)[0]
    periphery = np.where(initial_degrees <= k_50)[0]
    
    q = np.random.uniform(0, 1, N)
    P = np.full(N, 0.5)
    
    # 介入対象の設定（予算：ネットワーク全体の10%）
    budget = int(N * 0.1)
    Protected_Adj = np.zeros((N, N), dtype=bool)
    intervened_nodes = np.array([], dtype=int)
    
    if strategy == 'hub':
        # S1: トップダウン（次数上位10%を透明化）
        intervened_nodes = np.argsort(initial_degrees)[-budget:]
    elif strategy == 'periphery':
        # S2: ボトムアップ（次数下位10%を透明化）
        intervened_nodes = np.argsort(initial_degrees)[:budget]
    elif strategy == 'random_edge':
        # S3: 弱いつながりの強制（ランダムなエッジを予算分追加し、切断不可＋透明化）
        u_nodes = np.random.randint(0, N, budget)
        v_nodes = np.random.randint(0, N, budget)
        G.add_edges_from(zip(u_nodes, v_nodes))
        Protected_Adj[u_nodes, v_nodes] = True
        Protected_Adj[v_nodes, u_nodes] = True
        intervened_nodes = np.unique(np.concatenate([u_nodes, v_nodes]))

    history_macro = np.zeros(T)
    history_gap = np.zeros(T)
    
    for t in range(T):
        current_degrees = np.array([d for n, d in G.degree()])
        mean_k = np.mean(current_degrees) if np.mean(current_degrees) > 0 else 1.0
        
        # 1. 価格更新フェーズ（高速行列演算）
        A = nx.to_numpy_array(G, nodelist=range(N))
        np.fill_diagonal(A, 1)
        
        valid_mask = A * (q[None, :] <= P[:, None])
        sum_q = np.sum(valid_mask * q[None, :], axis=1)
        count_q = np.sum(valid_mask, axis=1)
        
        mean_q = np.zeros(N)
        valid_nodes = count_q > 0
        mean_q[valid_nodes] = sum_q[valid_nodes] / count_q[valid_nodes]
        
        # ノイズの計算と介入効果の適用（介入対象ノードはノイズ0）
        sigma = gamma * (mean_k / (current_degrees + 1))
        sigma[intervened_nodes] = 0.0 # 介入による透明化
        
        noise = np.random.normal(0, sigma)
        perceived_q = np.maximum(0.0, mean_q + noise)
        B = alpha * perceived_q
        P_new = np.clip(P + eta * (B - P), 0.01, 1.0)
        P = P_new
        
        # 2. 適応的再配線フェーズ
        edges = np.array(G.edges())
        if len(edges) > 0:
            u = edges[:, 0]
            v = edges[:, 1]
            
            dissatisfied = (q[u] < P[v] * 0.8) | (q[v] < P[u] * 0.8)
            drop_mask = dissatisfied & (np.random.rand(len(edges)) < p_drop)
            
            # S3の場合、保護されたエッジ（ランダムブリッジ）は切断させない
            if strategy == 'random_edge':
                is_prot = Protected_Adj[u, v]
                drop_mask = drop_mask & (~is_prot)
                
            edges_to_drop = edges[drop_mask]
            
            if len(edges_to_drop) > 0:
                G.remove_edges_from(edges_to_drop)
                nodes_to_rewire = edges_to_drop.flatten()
                num_rewire = len(nodes_to_rewire)
                
                candidates = np.random.randint(0, N, size=(num_rewire, 5))
                P_candidates = P[candidates]
                P_target = P[nodes_to_rewire].reshape(-1, 1)
                
                best_idx = np.argmin(np.abs(P_candidates - P_target), axis=1)
                best_candidates = candidates[np.arange(num_rewire), best_idx]
                
                valid = nodes_to_rewire != best_candidates
                new_edges = np.column_stack((nodes_to_rewire[valid], best_candidates[valid]))
                G.add_edges_from(new_edges)
                
        # 記録
        history_macro[t] = np.mean(P)
        history_gap[t] = np.mean(P[hubs]) - np.mean(P[periphery])
        
    return history_macro, history_gap, q, P

def compare_interventions():
    os.makedirs('output_intervention', exist_ok=True)
    print("戦略1 (Hubs) の計算中...")
    m_h, g_h, q_h, P_h = run_intervention_strategy('hub')
    print("戦略2 (Periphery) の計算中...")
    m_p, g_p, q_p, P_p = run_intervention_strategy('periphery')
    print("戦略3 (Random Edge) の計算中...")
    m_r, g_r, q_r, P_r = run_intervention_strategy('random_edge')
    
    T = len(m_h)
    
    # CSV保存
    df = pd.DataFrame({
        'TimeStep': range(T),
        'Macro_S1_Hub': m_h, 'Gap_S1_Hub': g_h,
        'Macro_S2_Periph': m_p, 'Gap_S2_Periph': g_p,
        'Macro_S3_Random': m_r, 'Gap_S3_Random': g_r
    })
    csv_path = 'output_intervention/intervention_comparison.csv'
    df.to_csv(csv_path, index=False)
    
    # プロット（2x3の複合グラフ）
    fig = plt.figure(figsize=(18, 10))
    
    # 上段: マクロ指標と格差
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df['TimeStep'], df['Macro_S1_Hub'], label='S1: Hubs', color='#d62728', lw=2)
    ax1.plot(df['TimeStep'], df['Macro_S2_Periph'], label='S2: Periphery', color='#1f77b4', lw=2)
    ax1.plot(df['TimeStep'], df['Macro_S3_Random'], label='S3: Random Bridges', color='#2ca02c', lw=2)
    ax1.set_title('Macro Economic Efficiency (Mean Price)', fontsize=14)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Global Mean Price')
    ax1.legend()
    ax1.grid(True, alpha=0.6)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df['TimeStep'], df['Gap_S1_Hub'], label='S1: Hubs', color='#d62728', lw=2)
    ax2.plot(df['TimeStep'], df['Gap_S2_Periph'], label='S2: Periphery', color='#1f77b4', lw=2)
    ax2.plot(df['TimeStep'], df['Gap_S3_Random'], label='S3: Random Bridges', color='#2ca02c', lw=2)
    ax2.set_title('Exclusion Gap (Hub Mean P - Periphery Mean P)', fontsize=14)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Price Gap')
    ax2.legend()
    ax2.grid(True, alpha=0.6)
    
    # 下段: 最終状態の散布図（エコチェンバーの可視化）
    axes = [plt.subplot(2, 3, 4), plt.subplot(2, 3, 5), plt.subplot(2, 3, 6)]
    titles = ['S1: Hub Intervention', 'S2: Periphery Intervention', 'S3: Random Bridge Intervention']
    qs = [q_h, q_p, q_r]
    Ps = [P_h, P_p, P_r]
    colors = ['Reds', 'Blues', 'Greens']
    
    for i in range(3):
        sc = axes[i].scatter(qs[i], Ps[i], alpha=0.5, c=Ps[i], cmap=colors[i], s=10)
        axes[i].set_title(titles[i], fontsize=12)
        axes[i].set_xlabel('True Quality (q)')
        axes[i].set_ylabel('Final Price (P)')
        axes[i].grid(True, alpha=0.6)
        axes[i].set_ylim(0, 1.05)
        
    plt.tight_layout()
    png_path = 'output_intervention/intervention_plot.png'
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    # ZIP化
    zip_filename = 'intervention_results.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(csv_path, arcname='intervention_comparison.csv')
        zipf.write(png_path, arcname='intervention_plot.png')
        
    return zip_filename

zip_file = compare_interventions()
files.download(zip_file)
print(f"完了: {zip_file} を作成・ダウンロードしました。")
