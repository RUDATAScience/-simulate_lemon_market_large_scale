import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
from google.colab import files

def simulate_lemon_market_large_scale(alpha=2.1, eta=0.1, gamma=0.08, T=300, N_list=[10, 1000, 100000, 10000000, 100000000]):
    """
    参加者数 N=1億までのスケールアップに伴うレモン市場の不安定化シミュレーション
    高速化のため、事前ソートと累積和を活用。
    """
    results = {}
    os.makedirs('output', exist_ok=True)
    
    for N in N_list:
        print(f"Calculating for N = {N:,}...")
        
        # 1. 乱数生成と事前ソート (メモリ節約のためfloat32を使用)
        q = np.random.uniform(0, 1, N).astype(np.float32)
        q.sort()
        
        # 2. 累積和の事前計算 (平均値の高速計算 O(1) 用)
        # 累積和を計算（インデックス0に0を追加しておく、精度落ち防止のためfloat64）
        cumsum_q = np.zeros(N + 1, dtype=np.float64)
        np.cumsum(q, out=cumsum_q[1:])
        
        P = np.zeros(T)
        P[0] = 0.5  # 初期価格
        
        for t in range(T - 1):
            # 3. 二分探索で現在の価格 P[t] 以下の品質を持つ車の数を特定 (計算量 O(log N))
            idx = np.searchsorted(q, P[t], side='right')
            
            if idx == 0:
                mean_q = 0.0
            else:
                # 累積和を使って平均品質を計算
                mean_q = cumsum_q[idx] / idx
            
            # 情報過多によるノイズ。Nが大きいほど対数的に増大
            noise = np.random.normal(0, gamma * np.log(N + 1))
            
            # 買い手の推定品質とそれに基づく入札価格
            perceived_q = max(0.0, mean_q + noise)
            B = alpha * perceived_q
            
            # 価格のアップデート
            P[t+1] = max(0.01, min(1.0, P[t] + eta * (B - P[t])))
            
        results[f'N={N}'] = P
        
    # データフレーム化とCSV保存
    df = pd.DataFrame(results)
    csv_path = 'output/lemon_market_100M.csv'
    df.to_csv(csv_path, index_label='TimeStep')
    
    # グラフのプロットとPNG保存
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, col in enumerate(df.columns):
        plt.plot(df.index, df[col], label=col, color=colors[i], alpha=0.8, linewidth=1.5)
        
    plt.title('Lemon Market Dynamics: Scaling up to 100 Million Agents', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Market Price', fontsize=12)
    plt.legend(title='Market Size (N)', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    png_path = 'output/lemon_market_100M_plot.png'
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    # ZIPファイルにまとめる
    zip_filename = 'lemon_market_100M_results.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(csv_path, arcname='lemon_market_100M.csv')
        zipf.write(png_path, arcname='lemon_market_100M_plot.png')
        
    return zip_filename

# シミュレーションの実行とダウンロード
zip_file = simulate_lemon_market_large_scale()
files.download(zip_file)
print(f"シミュレーション完了: {zip_file} をダウンロードしました。")
