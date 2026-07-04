# Lemon Market Dynamics on Adaptive Networks

## Overview
This repository contains high-performance Python simulation codes to analyze the non-linear dynamics of information asymmetry (Akerlof's Lemon Market) on complex adaptive networks. 

By scaling up multi-agent simulations to large numbers (up to 100 million agents), this project explores how cognitive limits and network topologies lead to structural social exclusion, the emergence of "echo chambers of lemons," and tests various optimal intervention strategies to restore market health and prevent irreversible fragmentation.

## Key Hypotheses Tested
This project systematically verifies four main hypotheses regarding information ecosystems and social exclusion:

*   **Hypothesis A (Perfect Information):** In a state of perfect information, price dynamics decouple completely from network topology (centrality), leading to a separating equilibrium[cite: 16].
*   **Hypothesis B (Adaptive Rewiring & Echo Chambers):** Under incomplete information with adaptive link rewiring, the network self-organizes into an irreversible divide: a "premium subgraph" monopolized by hubs and an "echo chamber of lemons" in the periphery[cite: 17].
*   **Hypothesis C (Optimal Intervention):** Under resource constraints, top-down intervention (transparentizing hubs) maximizes inequality, whereas protecting "weak ties" (random bridges) is the mathematical optimum for destroying echo chambers and achieving social inclusion[cite: 18, 19].
*   **Hypothesis D (Hysteresis & Tipping Points):** Social exclusion dynamics possess a distinct tipping point. Delayed post-collapse interventions fail to restore the system to its healthy state, proving the irreversibility (hysteresis) of scapegoating and ostracism[cite: 20].

## File Structure & Usage

The repository includes several standalone simulation scripts designed for Google Colab or local Python environments. Each script automatically generates CSV datasets, visualization plots (PNGs), and packages them into a downloadable ZIP file.

*   `main.py`: **Large-Scale Baseline Simulation** 
    Simulates the lemon market scaling up to 100,000,000 agents. Utilizes pre-sorting and cumulative sums (`numpy.cumsum`) to optimize calculations to $O(\log N)$[cite: 14].
*   `main2.py`: **Static Scale-Free Network Market** 
    Runs the localized lemon market simulation on a static Barabási-Albert scale-free network, comparing price dynamics across hubs, middle, and periphery nodes[cite: 15].
*   `main3.py`: **Hypothesis A (Perfect Information)** 
    Verifies the decoupling of price dynamics from node degrees when information is perfectly transparent[cite: 16].
*   `main4.py`: **Hypothesis B (Dynamic Adaptive Network)** 
    Implements ultra-fast tensor operations (adjacency matrix masking) to simulate dynamic edge dropping and rewiring, demonstrating the collapse of giant connected components and the emergence of echo chambers[cite: 17].
*   `main5.py` / `main6.py`: **Hypothesis C (Intervention Strategies)** 
    Compares three targeted intervention strategies (Hubs, Periphery, Random Bridges) allocating a 10% network budget to observe macro-economic efficiency and exclusion gaps[cite: 18, 19].
*   `main7.py`: **Hypothesis D (Hysteresis & Delayed Intervention)** 
    Tests the Random Bridge intervention at different time steps ($t=0, 50, 100$) to demonstrate hysteresis and the irreversibility of delayed interventions[cite: 20].

## Requirements
*   Python 3.7+
*   `numpy`
*   `pandas`
*   `networkx`
*   `matplotlib`

## How to Run
All scripts are optimized for [Google Colab](https://colab.research.google.com/). Simply copy and paste the code into a Colab cell and execute. The script will automatically compute the dynamics, generate 3-panel plotting charts, and prompt a download of the zipped results containing both the `.csv` data and `.png` visualizations.

If running locally, remove the `from google.colab import files` and `files.download(zip_file)` lines, and retrieve the results directly from the generated `output/` directories.

## License
MIT License. See `LICENSE` for more information.
