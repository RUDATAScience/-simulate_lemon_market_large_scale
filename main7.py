import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
import os
from google.colab import files

def run_delayed_intervention(t_intv, N=2000, m=4, T=150, alpha=2.1, eta=0.1, gamma=0.08, p_drop=0.3):
    """
    介入タイミング t_intv を変化させたランダムブリッジ介入シミュレーション
    """
    G = nx.barabasi_albert_graph(N, m)
    initial_degrees = np.array([d for n, d in G.degree()])
    
    k_90 = np.percentile(initial_degrees, 90)
    k_50 = np.percentile(initial_degrees, 50)
    
    hubs = np.where(initial_degrees >= k_90)[0]
    periphery = np.where(initial_degrees <= k_50)[0]
    
    q = np.random.uniform(0, 1, N)
    P = np.full(N, 0.5)
    
    budget = int(N * 0.1) # 10%の予算
    Protected_Adj = np.zeros((N, N), dtype=bool)
    intervened_nodes = np.array([], dtype=int)
    
    history_macro = np.zeros(T)
    history_gap = np.zeros(T)
    
    for t in range(T):
        # ------------------------------------------------------------
        # 介入の実行（指定されたタイミング t_intv で一回だけ発動）
        # ------------------------------------------------------------
        if t == t_intv:
            u_nodes = np.random.randint(0, N, budget)
            v_nodes = np.random.randint(0, N, budget)
            
            # 自己ループは除外
            valid = u_nodes != v_nodes
            u_nodes, v_nodes = u_nodes[valid], v_nodes[valid]
            
            G.add_edges_from(zip(u_nodes, v_nodes))
            Protected_Adj[u_nodes, v_nodes] = True
            Protected_Adj[v_nodes, u_nodes] = True
            intervened_nodes = np.unique(np.concatenate([u_nodes, v_nodes]))

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
        
        # ノイズ計算（介入済みのノードはノイズ0に固定される）
        sigma = gamma * (mean_k / (current_degrees + 1))
        if len(intervened_nodes) > 0:
            sigma[intervened_nodes] = 0.0
            
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
            
            # 保護されたエッジ（介入ブリッジ）は切断しない
            if len(intervened_nodes) > 0:
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

def compare_hysteresis():
    os.makedirs('output_hysteresis', exist_ok=True)
    print("t=0 (初期) での介入を計算中...")
    m_0, g_0, q_0, P_0 = run_delayed_intervention(t_intv=0)
    print("t=50 (中期) での介入を計算中...")
    m_50, g_50, q_50, P_50 = run_delayed_intervention(t_intv=50)
    print("t=100 (事後) での介入を計算中...")
    m_100, g_100, q_100, P_100 = run_delayed_intervention(t_intv=100)
    
    T = len(m_0)
    
    # CSV保存
    df = pd.DataFrame({
        'TimeStep': range(T),
        'Macro_t0': m_0, 'Gap_t0': g_0,
        'Macro_t50': m_50, 'Gap_t50': g_50,
        'Macro_t100': m_100, 'Gap_t100': g_100
    })
    csv_path = 'output_hysteresis/hysteresis_dynamics.csv'
    df.to_csv(csv_path, index=False)
    
    # プロット（2x3の複合グラフ）
    fig = plt.figure(figsize=(18, 10))
    
    # 上段: マクロ指標と格差
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df['TimeStep'], df['Macro_t0'], label='t=0 (Proactive)', color='#2ca02c', lw=2)
    ax1.plot(df['TimeStep'], df['Macro_t50'], label='t=50 (Delayed)', color='#ff7f0e', lw=2)
    ax1.plot(df['TimeStep'], df['Macro_t100'], label='t=100 (Post-Collapse)', color='#d62728', lw=2)
    ax1.axvline(x=50, color='#ff7f0e', linestyle='--', alpha=0.5)
    ax1.axvline(x=100, color='#d62728', linestyle='--', alpha=0.5)
    ax1.set_title('Macro Economic Efficiency: Effect of Intervention Timing', fontsize=14)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Global Mean Price')
    ax1.legend()
    ax1.grid(True, alpha=0.6)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df['TimeStep'], df['Gap_t0'], label='t=0 (Proactive)', color='#2ca02c', lw=2)
    ax2.plot(df['TimeStep'], df['Gap_t50'], label='t=50 (Delayed)', color='#ff7f0e', lw=2)
    ax2.plot(df['TimeStep'], df['Gap_t100'], label='t=100 (Post-Collapse)', color='#d62728', lw=2)
    ax2.axvline(x=50, color='#ff7f0e', linestyle='--', alpha=0.5)
    ax2.axvline(x=100, color='#d62728', linestyle='--', alpha=0.5)
    ax2.set_title('Exclusion Gap (Hysteresis & Irreversibility)', fontsize=14)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Price Gap (Hub - Periphery)')
    ax2.legend()
    ax2.grid(True, alpha=0.6)
    
    # 下段: 最終状態の散布図（エコチェンバーの残存確認）
    axes = [plt.subplot(2, 3, 4), plt.subplot(2, 3, 5), plt.subplot(2, 3, 6)]
    titles = ['Intervention at t=0', 'Intervention at t=50', 'Intervention at t=100']
    qs = [q_0, q_50, q_100]
    Ps = [P_0, P_50, P_100]
    colors = ['Greens', 'Oranges', 'Reds']
    
    for i in range(3):
        sc = axes[i].scatter(qs[i], Ps[i], alpha=0.5, c=Ps[i], cmap=colors[i], s=10)
        axes[i].set_title(titles[i], fontsize=12)
        axes[i].set_xlabel('True Quality (q)')
        axes[i].set_ylabel('Final Price (P)')
        axes[i].grid(True, alpha=0.6)
        axes[i].set_ylim(0, 1.05)
        
    plt.tight_layout()
    png_path = 'output_hysteresis/hysteresis_plot.png'
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    # ZIP化
    zip_filename = 'hysteresis_results.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(csv_path, arcname='hysteresis_dynamics.csv')
        zipf.write(png_path, arcname='hysteresis_plot.png')
        
    return zip_filename

zip_file = compare_hysteresis()
files.download(zip_file)
print(f"完了: {zip_file} を作成・ダウンロードしました。")
