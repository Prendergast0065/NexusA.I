import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import os
import openai
from openai import OpenAI
import time
import logging
import json
from datetime import timedelta
import random
from tqdm import tqdm
from dotenv import load_dotenv  # <--- Added for .env support

# Load environment variables from .env file at the start of the module
# This will be effective if this script is run standalone or when imported.
load_dotenv()

# === CONFIGURATION (Defaults) ===
LOG_RAW_GPT_RESPONSE_ON_ERROR_MODULE = True

logger = logging.getLogger('BacktesterLogicModule')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_GPT_MODEL = "gpt-4o"
DEFAULT_API_CALL_BUFFER_SECONDS = 2


def call_hosted_prompt(
    variables: dict,
    *,
    api_key: str,
    model: str,
    prompt_id: str,
    prompt_version: str,
    temperature: float = 0.3,
):
    """Execute an OpenAI hosted prompt and return the raw JSON string."""

    client = OpenAI(api_key=api_key)
    try:
        resp = client.responses.create(
            prompt={"id": prompt_id, "version": prompt_version, "variables": variables},
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    except TypeError:
        # Older openai versions do not support response_format
        resp = client.responses.create(
            prompt={"id": prompt_id, "version": prompt_version, "variables": variables},
            model=model,
            temperature=temperature,
        )
    return resp.choices[0].message.content



# --- Technical Indicator Functions ---
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rs[loss == 0] = float('inf')
    rs[(gain == 0) & (loss == 0)] = 0
    rsi = 100 - (100 / (1 + rs))
    rsi[rs == float('inf')] = 100
    return rsi


def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


# --- Data Loading and Preparation ---
def load_and_prepare_data_from_path(csv_file_path, num_days_to_process, randomize_period_bool):
    logger.info(f"Attempting to load data from: {csv_file_path}")
    try:
        df_full_history = pd.read_csv(csv_file_path, parse_dates=['timestamp'])
        df_full_history.set_index('timestamp', inplace=True)
        df_full_history.sort_index(inplace=True)
    except FileNotFoundError:
        logger.error(f"Data file not found: {csv_file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading CSV {csv_file_path}: {e}")
        return None

    if df_full_history.empty:
        logger.error(f"CSV file {csv_file_path} is empty or failed to load.")
        return None

    df_selected_period = None
    window_duration = pd.Timedelta(days=int(num_days_to_process))

    if randomize_period_bool:
        min_ts_in_data = df_full_history.index.min()
        max_ts_in_data = df_full_history.index.max()
        if (max_ts_in_data - min_ts_in_data) < window_duration:
            logger.warning(f"Total duration in CSV < requested duration. Using entire CSV.")
            df_selected_period = df_full_history.copy()
        else:
            latest_possible_start_ts = max_ts_in_data - window_duration
            possible_start_timestamps = df_full_history.index[
                df_full_history.index <= latest_possible_start_ts].unique()
            if not possible_start_timestamps.empty:
                random_start_ts = random.choice(possible_start_timestamps)
                end_ts_for_slice = min(random_start_ts + window_duration, max_ts_in_data)
                df_selected_period = df_full_history.loc[random_start_ts:end_ts_for_slice].copy()
                logger.info(
                    f"Randomly selected period from CSV: {df_selected_period.index.min()} to {df_selected_period.index.max()}")
            else:
                logger.warning("Could not find valid random start period in CSV. Using latest.")
                start_date_for_slice = max(min_ts_in_data, max_ts_in_data - window_duration)
                df_selected_period = df_full_history.loc[start_date_for_slice:].copy()
    else:
        start_date_for_slice = max(df_full_history.index.min(), df_full_history.index.max() - window_duration)
        df_selected_period = df_full_history.loc[start_date_for_slice:].copy()
        logger.info(
            f"Selected latest period from CSV: {df_selected_period.index.min()} to {df_selected_period.index.max()}")

    if df_selected_period is None or df_selected_period.empty:
        logger.error("No data selected for backtesting after period selection.")
        return None

    logger.info("Adding technical indicators to selected period...")
    if not all(col in df_selected_period.columns for col in ['high', 'low', 'close']):
        logger.error("CSV must contain 'high', 'low', 'close' columns for ATR.")
        return None

    df_selected_period['sma_30'] = df_selected_period['close'].rolling(window=30, min_periods=1).mean()
    df_selected_period['sma_60'] = df_selected_period['close'].rolling(window=60, min_periods=1).mean()
    df_selected_period['volatility'] = df_selected_period['close'].rolling(window=30, min_periods=1).std()
    df_selected_period['rsi'] = calculate_rsi(df_selected_period['close'])
    df_selected_period['atr'] = calculate_atr(df_selected_period['high'], df_selected_period['low'],
                                              df_selected_period['close'])

    df_selected_period.dropna(inplace=True)
    if df_selected_period.empty:
        logger.error("Data empty after TIs and dropping NaNs.")
        return None

    logger.info(
        f"Data prepared. Shape: {df_selected_period.shape}. Period: {df_selected_period.index.min()} to {df_selected_period.index.max()}")
    return df_selected_period


# --- AI Interaction ---
def get_gpt_action_for_web(sub_df, current_balance, current_btc_holdings,
                           trade_amount_btc_val, current_entry_price,
                           user_strategy_prompt_str,
                           openai_api_key_param,
                           api_call_buffer_seconds_param,
                           gpt_model_param):
    """Return trading action using either a hosted prompt or a raw prompt."""
    openai.api_key = openai_api_key_param  # Key is set per call from parameter

    # Log the key being used (masked)
    if openai.api_key:
        logger.debug(f"Using OpenAI Key: {openai.api_key[:5]}...{openai.api_key[-4:]} for API call.")
    else:
        logger.error("OpenAI API key is NOT SET before API call in get_gpt_action_for_web!")
        return "HOLD", 0.0, "OpenAI API key missing in function call", user_strategy_prompt_str

    if sub_df.shape[0] < 10:
        formatted_data = sub_df.to_string(index=True)
    else:
        formatted_data = sub_df.tail(10).to_string(index=True)

    prompt_to_send = user_strategy_prompt_str
    prompt_to_send = prompt_to_send.replace("{{DATA}}", formatted_data)
    prompt_to_send = prompt_to_send.replace("{{CURRENT_BTC_HOLDINGS}}", f"{current_btc_holdings:.8f}")
    prompt_to_send = prompt_to_send.replace("{{CURRENT_USD_BALANCE}}", f"{current_balance:.2f}")
    prompt_to_send = prompt_to_send.replace("{{TRADE_AMOUNT_BTC}}", f"{trade_amount_btc_val:.8f}")
    prompt_to_send = prompt_to_send.replace("{{ENTRY_PRICE}}", f"{current_entry_price:.2f}")

    logger.debug(f"Populated prompt for GPT (first 500 chars):\n{prompt_to_send[:500]}...")
    action, confidence, reasoning = "HOLD", 0.0, "Error in LLM call or default"
    content_from_llm = ""

    hosted_prompt_id = os.getenv("HOSTED_PROMPT_ID")
    hosted_prompt_version = os.getenv("HOSTED_PROMPT_VERSION", "1")

    try:
        if hosted_prompt_id:
            # Match variable names expected by the hosted prompt template
            variables = {
                "strategy_prompt": user_strategy_prompt_str,
                "data_block": formatted_data,
                "user_message": "json"
            }
            content_from_llm = call_hosted_prompt(
                variables=variables,
                api_key=openai_api_key_param,
                model=gpt_model_param,
                prompt_id=hosted_prompt_id,
                prompt_version=hosted_prompt_version,
            )
            logger.debug(f"Hosted prompt raw response: {content_from_llm}")
        else:
            response = openai.chat.completions.create(
                model=gpt_model_param,
                messages=[{"role": "user", "content": (prompt_to_send, "json")}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            content_from_llm = response.choices[0].message.content
            logger.debug(f"GPT raw response: {content_from_llm}")

        if api_call_buffer_seconds_param > 0:
            logger.debug(f"Waiting for {api_call_buffer_seconds_param}s after API call...")
            time.sleep(api_call_buffer_seconds_param)

        result = json.loads(content_from_llm)
        action = result.get("action", "HOLD").upper()
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "No reasoning provided by LLM.")
        if action not in ["BUY", "SELL", "HOLD"]:
            logger.warning(f"LLM returned invalid action: '{action}'. Defaulting to HOLD.")
            action, confidence = "HOLD", 0.0;
            reasoning += " (LLM invalid action)"
    # ... (rest of error handling same as before) ...
    except json.JSONDecodeError as e:
        action, confidence = "HOLD", 0.0;
        logger.error(f"GPT JSONDecodeError: {e}. Raw: '{content_from_llm}'");
        reasoning = f"JSON Decode Error: {e}"
    except openai.APIError as e:
        action, confidence = "HOLD", 0.0;
        logger.error(f"OpenAI API Error: {e}");
        reasoning = f"OpenAI API Error: {e}"
        if "Rate limit" in str(e): logger.warning("Rate limit possibly hit.")
    except Exception as e:
        action, confidence = "HOLD", 0.0;
        logger.error(f"Generic error in get_gpt_action: {e}", exc_info=True);
        reasoning = f"Generic LLM Error: {e}"
        if LOG_RAW_GPT_RESPONSE_ON_ERROR_MODULE and content_from_llm: logger.error(
            f"Problematic content: {content_from_llm}")
    return action, confidence, reasoning, prompt_to_send


# --- Main Backtesting Orchestration Function ---
def execute_backtest_strategy(
        openai_api_key_param,
        user_strategy_prompt_str,
        csv_data_path_param,
        num_days_param,
        randomize_period_from_csv_param,
        job_id_param,
        output_base_dir,
        start_balance=10000.0, trade_amount_btc=0.01, transaction_fee_percent=0.075,
        min_confidence_threshold=60, use_python_managed_sltp_safety_net=True,
        stop_loss_atr_multiplier=2.2, take_profit_atr_multiplier=3.0,
        api_call_buffer_seconds=DEFAULT_API_CALL_BUFFER_SECONDS,
        gpt_model=DEFAULT_GPT_MODEL
):
    logger.info(
        f"[{job_id_param}] Initializing backtest. OpenAI Key: {'Provided' if openai_api_key_param else 'MISSING'}")
    if not openai_api_key_param: return {"error": "OpenAI API Key was not provided to backtester."}

    df_full = load_and_prepare_data_from_path(csv_data_path_param, num_days_param, randomize_period_from_csv_param)
    if df_full is None or df_full.empty: return {"error": "Data preparation failed."}

    balance_usd = start_balance;
    btc_holdings = 0.0;
    entry_price = 0.0;
    active_position = False
    portfolio_history, trade_logs_list, logged_prompts_list = [], [], []
    decision_window = timedelta(minutes=30)
    simulation_start_time, simulation_end_time = df_full.index.min(), df_full.index.max()
    current_time = simulation_start_time
    total_steps = max(1, int((
                                         simulation_end_time - simulation_start_time).total_seconds() / decision_window.total_seconds())) if decision_window.total_seconds() > 0 else 1
    logger.info(
        f"[{job_id_param}] Backtest period: {simulation_start_time} to {simulation_end_time} ({total_steps} steps)")

    with tqdm(total=total_steps, desc=f"Job {job_id_param}", unit="step", disable=not (__name__ == '__main__')) as pbar:
        while current_time <= simulation_end_time:
            # ... (rest of the backtesting loop logic from previous version is largely the same) ...
            # This loop uses get_gpt_action_for_web, which now takes the API key as a parameter.
            window_end = current_time
            window_start = max(current_time - decision_window, simulation_start_time)
            current_market_data_df = df_full.loc[window_start:window_end].copy()

            if current_market_data_df.shape[0] < 2:
                logger.debug(
                    f"[{job_id_param}] Skipping {current_time}, not enough data ({current_market_data_df.shape[0]} rows)")
                current_time += decision_window;
                pbar.update(1);
                continue

            execution_price = current_market_data_df['close'].iloc[-1]
            current_atr = current_market_data_df['atr'].iloc[-1] if pd.notna(
                current_market_data_df['atr'].iloc[-1]) else 0.01

            final_action_for_step = "HOLD";
            final_reasoning_for_step = "Default Hold";
            llm_confidence_for_log = 0.0
            system_exit_triggered = False

            if active_position and use_python_managed_sltp_safety_net and current_atr > 0:
                sl_target = entry_price - (stop_loss_atr_multiplier * current_atr)
                tp_target = entry_price + (take_profit_atr_multiplier * current_atr)
                if execution_price <= sl_target:
                    final_action_for_step, llm_confidence_for_log = "SELL", 100.0
                    final_reasoning_for_step = f"SYSTEM SL @ {execution_price:.2f} (Entry:{entry_price:.2f}, Target:{sl_target:.2f}, ATR:{current_atr:.2f})"
                    system_exit_triggered = True
                elif execution_price >= tp_target:
                    final_action_for_step, llm_confidence_for_log = "SELL", 100.0
                    final_reasoning_for_step = f"SYSTEM TP @ {execution_price:.2f} (Entry:{entry_price:.2f}, Target:{tp_target:.2f}, ATR:{current_atr:.2f})"
                    system_exit_triggered = True

            if not system_exit_triggered:
                llm_action, llm_confidence, llm_reasoning, populated_prompt = get_gpt_action_for_web(
                    current_market_data_df, balance_usd, btc_holdings,
                    trade_amount_btc, entry_price if active_position else 0.0,
                    user_strategy_prompt_str, openai_api_key_param,
                    api_call_buffer_seconds, gpt_model
                )
                logged_prompts_list.append((current_time, populated_prompt));
                llm_confidence_for_log = llm_confidence
                final_reasoning_for_step = llm_reasoning

                if active_position:
                    if llm_action == "SELL" and llm_confidence >= min_confidence_threshold:
                        final_action_for_step = "SELL";
                        final_reasoning_for_step = f"LLM EXIT: {llm_reasoning}"
                    else:
                        final_action_for_step = "HOLD"
                        if llm_action == "SELL":
                            final_reasoning_for_step = f"LLM Low Conf SELL ({llm_confidence:.1f}%), holding. LLM: {llm_reasoning}"
                        elif llm_action == "BUY":
                            final_reasoning_for_step = "LLM BUY (in pos), holding. LLM: " + llm_reasoning
                else:
                    if llm_action == "BUY" and llm_confidence >= min_confidence_threshold:
                        final_action_for_step = "BUY"
                    else:
                        final_action_for_step = "HOLD"
                        if llm_action == "BUY":
                            final_reasoning_for_step = f"LLM Low Conf BUY ({llm_confidence:.1f}%), holding. LLM: {llm_reasoning}"
                        elif llm_action == "SELL":
                            final_reasoning_for_step = "LLM SELL (no pos), holding. LLM: " + llm_reasoning

            log_display = final_action_for_step
            if final_action_for_step == "BUY" and not active_position:
                cost_pre_fee = trade_amount_btc * execution_price;
                fee = cost_pre_fee * (transaction_fee_percent / 100.0)
                if balance_usd >= cost_pre_fee + fee:
                    btc_holdings += trade_amount_btc;
                    balance_usd -= (cost_pre_fee + fee)
                    entry_price = execution_price;
                    active_position = True
                else:
                    log_display = "HOLD (Buy Fail Bal)"; final_reasoning_for_step += " (Insuff. bal)"
            elif final_action_for_step == "SELL" and active_position:
                proceeds_pre_fee = btc_holdings * execution_price;
                fee = proceeds_pre_fee * (transaction_fee_percent / 100.0)
                balance_usd += (proceeds_pre_fee - fee);
                btc_holdings = 0.0
                active_position = False;
                entry_price = 0.0
            else:
                if final_action_for_step != "HOLD": log_display = f"HOLD ({final_action_for_step} Fail)"
                if final_reasoning_for_step == "Default Hold / No signal" and llm_reasoning: final_reasoning_for_step = llm_reasoning

            action_log_message = f"[{log_display}] {current_time}: Px={execution_price:.2f}, Bal={balance_usd:.2f}, BTC={btc_holdings:.4f}, Conf={llm_confidence_for_log:.1f}%, R: {final_reasoning_for_step}"
            if system_exit_triggered: action_log_message = f"[SYSTEM {final_action_for_step}] {current_time}: {final_reasoning_for_step}"
            logger.info(action_log_message)

            trade_logs_list.append((current_time, log_display, llm_confidence_for_log, execution_price, balance_usd,
                                    btc_holdings, final_reasoning_for_step, current_atr))
            portfolio_history.append((current_time, balance_usd + (btc_holdings * execution_price)))
            pbar.update(1);
            pbar.set_postfix({"Equity": f"${portfolio_history[-1][1]:.2f}", "Last": log_display}, refresh=True)
            if current_time == simulation_end_time and pbar.n < pbar.total: pbar.update(pbar.total - pbar.n)
            current_time += decision_window
            if not portfolio_history and current_time > simulation_end_time + timedelta(hours=1): logger.error(
                f"[{job_id_param}] Safety break loop."); break

    log_df_output = pd.DataFrame(trade_logs_list,
                                 columns=["timestamp", "action", "confidence", "price", "balance_usd", "btc_holdings",
                                          "reasoning", "atr_at_trade"])
    equity_df_output = pd.DataFrame(portfolio_history, columns=["timestamp", "equity_usd"])
    if not equity_df_output.empty: equity_df_output.set_index("timestamp", inplace=True)

    job_output_dir = os.path.join(output_base_dir, job_id_param)
    os.makedirs(job_output_dir, exist_ok=True)
    equity_curve_filename = "equity_curve.png";
    trade_log_filename = "trade_log.csv"
    equity_curve_path = os.path.join(job_output_dir, equity_curve_filename)
    trade_log_path = os.path.join(job_output_dir, trade_log_filename)

    try:
        if not equity_df_output.empty:
            fig, ax = plt.subplots(figsize=(12, 7));
            equity_df_output['equity_usd'].plot(ax=ax, title=f"Equity Curve - Job {job_id_param}",
                                                ylabel="Portfolio Value (USD)", color="#3B82F6")
            ax.grid(True, linestyle='--', alpha=0.7);
            ax.set_facecolor('#1F2937');
            fig.patch.set_facecolor('#1F2937')
            ax.tick_params(colors='white');
            ax.title.set_color('white');
            ax.yaxis.label.set_color('white');
            ax.xaxis.label.set_color('white')
            plt.tight_layout();
            plt.savefig(equity_curve_path, facecolor=fig.get_facecolor());
            plt.close(fig)
        else:
            raise ValueError("No equity data to plot.")
    except Exception as e_img:
        logger.error(f"Error generating plot for {job_id_param}: {e_img}. Placeholder.")
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (600, 300), color=(31, 41, 55));
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        d.text((10, 10), f"Plot Error {job_id_param}\n{str(e_img)[:100]}", fill=(229, 231, 235), font=font);
        img.save(equity_curve_path)

    if not log_df_output.empty:
        log_df_output.to_csv(trade_log_path, index=False)
    else:
        pd.DataFrame(columns=["timestamp", "action", "confidence", "price", "balance_usd", "btc_holdings", "reasoning",
                              "atr_at_trade"]).to_csv(trade_log_path, index=False)

    final_equity = equity_df_output['equity_usd'].iloc[-1] if not equity_df_output.empty else start_balance
    net_pl = final_equity - start_balance
    num_trades = len(log_df_output[log_df_output['action'] == 'BUY']) if not log_df_output.empty else 0

    logger.info(f"[{job_id_param}] Backtest finished. Final Equity: {final_equity:.2f}")
    return {
        "equity_curve_url": f"/static/results/{job_id_param}/{equity_curve_filename}",
        "trade_log_url": f"/static/results/{job_id_param}/{trade_log_filename}",
        "final_equity": final_equity, "net_pl": net_pl, "total_trades": num_trades
    }


# Standalone testing block
if __name__ == '__main__':
    logger.info("Running my_backtester_logic.py standalone for testing.")
    # Attempt to load OPENAI_API_KEY from .env for standalone test
    # load_dotenv() is already called at the top of the module
    test_openai_key = os.getenv("OPENAI_API_KEY")

    if not test_openai_key:
        logger.error("OPENAI_API_KEY not found in .env for standalone test. Please create a .env file with your key.")
        # Fallback to a dummy key if not found, which will cause API errors
        test_openai_key = "YOUR_DUMMY_OPENAI_KEY_FOR_TESTING_IF_NO_ENV"
        logger.warning(f"Using placeholder key: {test_openai_key}. OpenAI calls will fail.")
    else:
        logger.info(f"Using OpenAI key from .env for standalone test: {test_openai_key[:5]}...{test_openai_key[-4:]}")

    dummy_job_id = "standalone_test_003"
    dummy_csv_path = "dummy_btc_data_standalone.csv"  # Ensure this file exists or is created

    # Create a more substantial dummy CSV for better testing of indicators
    timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=300, freq='min')  # 5 hours of data
    data = {
        'timestamp': timestamps,
        'open': [10000 + i * 0.2 - (i % 10) * 0.5 for i in range(len(timestamps))],
        'high': [10000 + i * 0.2 + 5 - (i % 10) * 0.5 for i in range(len(timestamps))],
        'low': [10000 + i * 0.2 - 5 - (i % 10) * 0.5 for i in range(len(timestamps))],
        'close': [10000 + (i * 0.22) - ((i * 3) % 15) for i in range(len(timestamps))],
        'volume': [10 + i % 10 for i in range(len(timestamps))]
    }
    pd.DataFrame(data).to_csv(dummy_csv_path, index=False)
    logger.info(f"Created dummy CSV for testing: {dummy_csv_path}")

    # Ensure output directory exists for standalone test
    # Assuming 'static/results' is relative to where this script is, or adjust path
    standalone_output_dir = os.path.join('static', 'results')
    if not os.path.exists(standalone_output_dir):
        os.makedirs(standalone_output_dir, exist_ok=True)

    # Example prompt string (same as the one from your web form)
    test_user_prompt = """
You are a crypto trading analyst for BTC/USDT, focusing on short-term trend-following opportunities.
Your goal is to identify clear BUY or SELL signals based on the provided data.

Current Portfolio State:
- BTC Holdings: {{CURRENT_BTC_HOLDINGS}} BTC
- Available Balance: {{CURRENT_USD_BALANCE}} USD
- Standard Trade Amount: {{TRADE_AMOUNT_BTC}} BTC
- Entry Price of Current Position: {{ENTRY_PRICE}}

Market Data (10 most recent 1-minute candles, includes open, high, low, close, volume, sma_30, sma_60, volatility, RSI, ATR):
{{DATA}}

Your Task: Analyze the market data AND current portfolio state.

If NO BTC IS HELD (Holdings = 0.0):
  Consider a BUY if:
    1. Price is above both sma_30 and sma_60.
    2. sma_30 is above sma_60 (golden cross confirmation).
    3. RSI is above 50 and ideally rising.
    4. Volume shows some support for the move.
  If these conditions strongly suggest an uptrend, recommend "action": "BUY".

If BTC IS HELD (Holdings > 0.0):
  Consider a SELL if:
    1. Price falls below sma_30.
    2. RSI drops below 50.
    3. OR if current unrealized profit is >= 2 * current ATR from entry AND RSI shows bearish divergence or is overbought (e.g., >70) and turning down.
  If these conditions strongly suggest the uptrend is ending or reversing, recommend "action": "SELL".

If NEITHER strong BUY nor strong SELL conditions are met, recommend "action": "HOLD".

Output Format:
Respond ONLY with a valid JSON object. No other text.
{"action": "BUY" | "SELL" | "HOLD", "confidence": 0-100, "reasoning": "Your concise analysis (max 1-2 sentences)."}

Confidence Score:
- For BUY/SELL: Reflects the strength of the signal. Aim for >60% for action.
- For HOLD: Can reflect uncertainty OR conviction that holding is best.
"""

    results = execute_backtest_strategy(
        openai_api_key_param=test_openai_key,
        user_strategy_prompt_str=test_user_prompt,
        csv_data_path_param=dummy_csv_path,
        num_days_param=1,
        randomize_period_from_csv_param=False,
        job_id_param=dummy_job_id,
        output_base_dir=standalone_output_dir
    )
    logger.info(f"Standalone test results: {results}")
    if "error" not in results and results.get("results"):  # Check 'results' sub-dictionary
        logger.info(f"Equity curve should be at: {results.get('equity_curve_url')}")  # Access from top level
        logger.info(f"Trade log should be at: {results.get('trade_log_url')}")  # Access from top level
    elif results.get("error"):
        logger.error(f"Standalone test failed: {results.get('error')}")

