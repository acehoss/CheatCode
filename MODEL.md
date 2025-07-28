# CheatCode Model

## Overview

The CheatCode Model is a custom transformer neural network designed for futures market trading. It processes 1-minute candle data from ES and NQ futures to predict "dubba" trading signals - trades with at least 2:1 reward-to-risk ratio across multiple time horizons and price bands.

## Model Inputs

The model takes in data sets for two ticker symbols (NQ and ES) covering the same time period, to discover hidden correlations.

Each data set contains 1 day of 1 minute candles

Each candle contains:

- open price 
- high price
- low price
- close price
- volume

Additionally, each candle contains sinusoidal embeddings for:

- time of day
- day of week
- day of month
- day of year
- minute of the hour

The sinusoidal embeddings have a sine and cosine component, meaning each embedding contributes 2 features.

This results in a total of 15 data points per candle.

**Positional Encoding**

To allow the transformer to learn the relationships between candles within the same training sample, a positional encoding is applied to each candle within the transformer architecture. This positional encoding does not increase the input data size but enhances the model's ability to capture temporal relationships.

### Handling Market Closed Periods

The training data includes periods when the futures markets are closed. During these market closed periods, each candle is represented as a "zero volume, point price" candle. This means:

- **Open**, **High**, **Low**, **Close** (OHLC) values are set to the price of the previous market close.
- **Volume** is set to zero.
- **Sinusoidal Embeddings**: Time embeddings are included as usual to represent the time features.

This approach maintains a continuous time series without gaps, allowing the model to learn temporal patterns across both open and closed market periods.

### Model Input Size

Futures markets are open 24 hours, Sunday through Friday, but close from Friday evening to Sunday evening. By including market closed periods, the total number of inputs per day remains:

- **1 day of 1-minute candles**: \( 15 \text{ data points} \times 60 \text{ minutes} \times 24 \text{ hours} = 21,600 \text{ data points per ticker symbol} \).
- Since there are two ticker symbols, the total input size is \( 21,600 \times 2 = 43,200 \) floating point data elements per day.

**Note**: The inclusion of market closed periods slightly alters the distribution of data points where volume is zero and prices remain constant.

## Model Outputs

The model infers a trading signal, which we call a dubba. A dubba is signalled when the model predicts the price is likely to reach a target price without moving opposite more than half the range of the target price before the target price is reached. (Called dubba because the target price is at least double the drawdown).

The model outputs a dubba signal by price for every 1 minute for the next 30 minutes for each ticker symbol.

Output price bands are relative to the current price (which is represented as 1.0) and are one tick wide, with **30 bands per time scale**, half above the current price and half below. The width of a tick in relative price terms is calculated based on the current absolute price.

For each ticker, that means there are **30 bands × 30 minutes = 900 output values per ticker**, or **1,800 output values for both tickers**.

## Training

Training data will be derived from the complete history of trades made for each of the tickers for a 57-week period. From this trade data, input data with expected outputs can be computed for a full year.

Training will use binary cross-entropy on each output value to measure loss.

### **1. Data Generation with Fixed Time Steps**

**Total Historical Data**: 57 weeks.

**Usable Data for Input Generation**: 52 weeks (after excluding the initial data required for input history due to data limitations).

**Input Generation Method**:

- **Fixed Sliding Window Approach**:
  - **Step Size**: A fixed interval of **1 minute**.
  - **Days per Week**: 7 days (including weekends due to market closed periods).
  - **Minutes per Day**: 1,440 minutes (24 hours).

**Calculating Inputs per Week**:

- **Total Days per Week**: 7 days.
- **Estimated Inputs per Week**:
  - \( 7 \text{ days/week} \times 1,440 \text{ inputs/day} = 10,080 \text{ inputs/week} \).

**Total Number of Inputs**:

- **Total Weeks**: 52 weeks.
- **Total Inputs**:
  - \( 52 \text{ weeks} \times 10,080 \text{ inputs/week} = 524,160 \text{ inputs} \).

### **2. Data Splitting**

The dataset uses a deterministic splitting strategy to ensure reproducibility:

**Data Shuffling and Splitting**:

- **Shuffling**: Optional, with seed preservation for reproducibility
- **Split Strategy**: Every 11th sample goes to validation (indices where i % 11 == 10)
- **Data Split Ratios**:
  - **Training Set**: ~**90.9%** of the data
  - **Validation Set**: ~**9.1%** of the data
- **Test Set**: No separate test set in current implementation

**Actual Split (from 524,160 total inputs)**:

- **Training Set**: ~476,509 inputs
- **Validation Set**: ~47,651 inputs

This approach ensures:

- Even distribution of validation samples throughout the dataset
- Reproducible splits when using the same shuffle seed
- No data leakage between training and validation

### **3. Training and Evaluation Process**

**Model Training**:

- **Training Set**:
  - **Number of Inputs**: Approximately **471,744** inputs.
  - **Batch Size**: Adjust based on hardware capabilities (e.g., batch size of 64 or 128).
  - **Epochs**: Begin with a reasonable number (e.g., 10 epochs) and adjust based on validation performance.

- **Validation**:

  - **Validation Set**: Approximately **52,416** inputs.
  - **Use**: For hyperparameter tuning and early stopping decisions.

- **Testing**:

  - There is **no separate test set** or testing phase at this time.

**Note**:

- Despite standard practices for time-series data, we are shuffling the samples before splitting to simplify implementation. This may affect the model's ability to capture temporal dependencies across longer periods.
- We may revisit data segmentation and temporal ordering in the future to enhance model performance.

### **4. Incorporating Market Closed Periods**

**Input Generation Process**:

- **Start Time**: The beginning of each day, including weekends.
- **While Loop**:
  - Generate an input at the current time.
  - Advance the current time by **1 minute**.
  - Continue until the end of the day.

**Representation During Market Closed Periods**:

- **OHLC Values**: Set to the price of the previous market close.
- **Volume**: Set to zero.
- **Sinusoidal Embeddings**: Included as usual to represent time features.
- **Purpose**: Allows the model to learn patterns related to market closures and openings, and maintains continuity in the time series data.

**Preserving Temporal Order Within Segments**:

- Inputs are generated sequentially every minute, respecting the temporal order of the data.
- There is no randomness in step sizes, ensuring consistent time intervals between inputs.

## Model Architecture Overview

The model implements a **Dual-Stage Transformer Architecture** with cross-ticker attention.

**Key Components:**

### 1. Input Processing Stage

- **Ticker Subnetworks** (`TickerSubNetwork`):
  - Each ticker (NQ and ES) has its own transformer encoder
  - Input projection: Linear(15, d_model) to project candle features
  - Positional encoding: Custom sinusoidal encoding for 1440-minute sequences
  - Transformer encoder with configurable layers
  - Temporal aggregation via mean pooling
  - Mish activation throughout

### 2. Cross-Attention Stage

- **Cross-Ticker Attention**:
  - Multi-head attention mechanism between ticker representations
  - Learns correlations between ES and NQ futures
  - Uses same dimensions as input transformers (d_model_input, nhead_input)

### 3. Post-Processing Stage

- **Dimension Adjustment** (if needed):
  - Linear layer to transition from d_model_input to d_model_post
  - Identity layer if dimensions match

- **Post-Cross-Attention Transformer**:
  - Additional transformer encoder layers
  - Refines the combined ticker representations
  - Separate hyperparameters (d_model_post, nhead_post) for flexibility

### 4. Output Generation

- **Output Module**:
  - FC layer with Mish activation and dropout
  - Final linear projection to 1800 outputs (900 per ticker)
  - Binary classification for each price band × time horizon

**Architecture Parameters**:

- `num_tickers`: 2 (ES and NQ)
- `seq_len`: 1440 (24 hours of 1-minute candles)
- `d_model_input`: Input transformer dimension (default: 128)
- `nhead_input`: Input transformer attention heads (default: 8-16)
- `num_encoder_layers`: Layers per ticker encoder (default: 2-4)
- `d_model_post`: Post-transformer dimension (can differ from input)
- `nhead_post`: Post-transformer attention heads
- `num_post_layers`: Post-cross-attention layers (default: 2)
- `dim_feedforward_input/post`: Feedforward network dimensions
- `dropout`: Dropout rate throughout the model

**Implementation Details**:

- All transformers use batch_first=True for efficiency
- Xavier uniform initialization for linear layers
- Layer normalization with eps=1e-4
- Supports multi-GPU training via DataParallel
