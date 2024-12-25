# AI-Powered Stock Analysis Dashboard

An interactive dashboard combining technical analysis, sentiment analysis, and price predictions for stocks.

## Features

- Real-time stock data analysis
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- News sentiment analysis using LLM
- Price predictions using SARIMAX
- Interactive charts and visualizations

## Requirements

- Python 3.8+
- Ollama (for sentiment analysis)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-dashboard.git
cd stock-dashboard
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install and run Ollama:
- Follow instructions at https://ollama.ai/
- Pull the llama3 model: `ollama pull llama3`

## Usage

1. Make sure Ollama is running
2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to http://localhost:8501

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
