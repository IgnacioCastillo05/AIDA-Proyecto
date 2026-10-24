# UrbanStyle Project Plan

## 1. Kaggle Dataset

Recommended dataset matching the required fields (Transaction ID,
Customer ID, Gender, Age, Product Category, Price, Quantity, Total
Amount):

https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset

**Interesting Simulation project for your review:**

https://github.com/KG0116/store-sim

## 2. Step-by-step Project Tasks (UrbanStyle Business Case)

UrbanStyle = a fashion retailer. All tasks focus on this business.
Pygame is obligatory for interactivity (buttons/sliders), but the
central focus is on algorithms (search, forecasting, RL, GAN).

### Task A --- Customer Pathfinding & Store Layout (Search Algorithms)

**Goal:** Show how customers move inside the store and identify crowded
spots.

**Required:**

1.  Represent the store as a grid (e.g., 8×6 cells).
2.  Map product categories to sections.
3.  Customers enter and move to product sections.
4.  Implement at least two search algorithms (BFS and A\*).
5.  Visualize high-traffic areas.

**Pygame component:** Display the grid, moving customers, and allow
sliders/buttons to change algorithm or number of customers.

**Optional:** Add heatmap overlay or test different layouts.

### Task B --- Demand Forecasting (Time-Series with Small LSTM)

**Goal:** Predict future sales to support inventory decisions.

**Required:**

1.  Select 5--8 categories.
2.  Preprocess as weekly time-series.
3.  Train a small LSTM or encoder-decoder.
4.  Predict next week's sales.
5.  Evaluate with MAE and RMSE.

**Pygame component:** Show forecast plot, with sliders to select
category or training length.

**Optional:** Add moving average baseline or CSV export of predictions.

### Task C --- Dynamic Pricing with Reinforcement Learning (RL + Gym)

**Goal:** Adjust prices dynamically to maximize revenue.

**Required:**

1.  Define a Gym environment (state: price, segment, recent sales;
    actions: adjust price; reward: revenue).
2.  Use stable-baselines3 (DQN or PPO) to train an agent.
3.  Run episodes and measure revenue.
4.  Compare to static pricing.

**Pygame component:** Display live revenue plot, allow sliders to set
initial price or reset environment.

**Optional:** Add multiple products or test multiple RL algorithms.

### Task D --- Simple GAN for Promotion Images

**Goal:** Create new promotional fashion images.

You can use MNIST-Fashion
(https://github.com/zalandoresearch/fashion-mnist).

**Required:**

1.  Train a very small GAN (e.g., DCGAN) on product images.
2.  Generator: noise → image (28×28).
3.  Discriminator: classify real/fake.
4.  Train a few epochs and save outputs.

**Pygame component:** Button to generate images, sliders for noise seed.

**Optional:** Conditional GAN (cGAN) or combine images into promo
poster.

## 3. Evaluation Rubric

  ------------------------------------------------------------------------------
  Task                            Weight Required criteria      Optional (bonus)
  ---------------- --------------------- ---------------------- ----------------
  Task A ---                         15% Grid model, BFS & A\*  Heatmap overlay,
  Pathfinding                            implemented, paths     layout
                                         visualized,            experiments
                                         high-traffic zones     
                                         identified             

  Task B ---                         20% Time-series prep, LSTM Baseline
  Forecasting                            model runs, MAE & RMSE comparison, CSV
                                         reported               export

  Task C --- RL                      25% Gym env built, RL      Multi-product or
  Pricing                                agent trained          multiple RL
                                         (stable-baselines3),   algorithms
                                         revenue vs static      
                                         shown                  

  Task D --- GANs                    20% Small GAN trained,     Conditional GAN
                                         sample outputs         or promo poster
                                         generated              

  Pygame                             10% Buttons/sliders for    Extra UI polish
  integration                            interactivity, at      
                                         least one              
                                         visualization per task 

  Report & Code                      10% Clear report,          ---
                                         reproducible code,     
                                         short presentation     
  ------------------------------------------------------------------------------
