# Weather Markets


Live weather
https://www.weather.gov/wrh/timeseries?site=KATL


Forecast



TODO:

Debug https://www.weather.gov/wrh/timeseries?site=KATL

Why did this end at 83???




# data

* Live data only records outputs to whole C. Metar data every hour is more accurate to 0.1C
* 6h highs and lows correspond to the 24h highs and lows and vary per station


│ Backtest data shows naive_is_high creates a near-perfect structural partition:                                                                                                              │
│ - naive_is_high == 0 (naive = f_low): offset is never −1 (0.1%, 1/679 days)                                                                                                                 │
│ - naive_is_high == 1 (naive = f_high): offset is rarely +1 (5.4%, 40/740 days)                                                                                                              │
│ - n_possible_f == 1 (exact): offset is almost always 0 (92.8%)   

naive_is_low → clamps −1 to 0 [-1, 0] bracket
naive_is_high → clamps +1 to 0  [0, 1] bracket
metar_above_boundary → clamps −1 to 0 (existing rule, still applied) [0, 1] bracket
metar_gap_c >= 0.25, force to +1
metar_gap_c >= 0.2 & nih==0  to +1

# Temps

https://www.reddit.com/r/Kalshi/comments/1hfvnmj/an_incomplete_and_unofficial_guide_to_temperature/

Interpreting NWS Time Series: The NWS reports data from two different types of weather stations. They have technical names, but for our purposes, I’ll refer to them as hourly stations, and 5-minute stations. They record and report data very differently, so understanding the difference is critical. 

Hourly stations are constantly recording the temperature, but only report the temperature once at hour. These temperatures are recorded to the nearest 0.1F, then converted to the nearest 0.1C and sent to the NWS. The NWS then reports the temperature celcius, and converts it back to fahrenheit and reports that as well. Technically, the Fahrenheit value you see isn’t correct, because some rounding/conversion error has been introduced by converting it to the nearest 0.1C and back again. But the error is typically minimal. Every 6 to 24 hours, these stations will send the highest temperature they’ve measured to the NWS, and this will be recorded as the high for the day. Importantly, because you only see the temperature at hourly points in time AND because the rounding/conversion error is minor, the daily high will typically be greater than, and sometimes equal to, the highest hourly reading you see.

5-minute stations are very different. They record temperature in 1 minute long averages. 5 of these 1 minute long averages are then averaged into a 5 minute average. This 5 minute average is rounded to the nearest whole degree fahrenheit, which is then converted to the nearest whole degree celsius and sent to the NWS. Note: This is apparently due to the DOS limitations in the software running the stations, which, are you serious? But I digress. When the NWS reports the temperature in fahrenheit, it converts the celsius value to fahrenheit, with no regard for the original fahrenheit value. Again, yes, this is for real. This process introduces a significant amount of rounding and conversion error that makes the NWS reported fahrenheit values nearly meaningless, and can be a degree or more higher or lower than the official high for the day ends up being. Instead, what you’d want to do is work backwards to determine which original fahrenheit values would then get rounded and converted to the celsius value you see on the NWS time series. Most celsius values reported by the 5-minute stations could be the result of two different actual fahrenheit temperatures. If those two fahrenheit temperatures fall in the same two degree range on Kalshi, you can be confident whatever the underlying temperature, it’s within that range. However, if the two underlying values span two different Kalshi ranges, you’ll need to use your intuition and understanding of the temperatures to make an educated guess which of the two fahrenheit values occurred, and from that, which Kalshi range to bet on.

How the actual high is determined by the NWS: The NWS records the high for the day by checking both stations for the highest temperature recorded over the course of the entire day and taking the higher of the two. The hourly station records temperature constantly (I imagine this means 1 minute averages, like the 5-minute stations, but I’m not 100% certain) and the 5-minute station records 1 minute averages. The highest of these values, rounded to the nearest whole degree fahrenheit, is the high for the day. These values are not converted to celsius and back again, avoiding rounding/conversion error that plagues the 5-minute stations. And they are not averaged over 5 minutes, so the high can potentially be higher than any value that could be rounded to the whole degree celsius reported by the 5-minute stations.

# Peak pred

max auto = 72
bracket (0, +1) = [72, 73]
bracket (+2, +3) = [74, 75]


The auto_c → Settlement Pipeline                                                                                                                                                             
                                                                                                                                                                                               
  1. Raw Temperature Data                                                                                                                                                                      
                                                                                                                                                                                               
  ASOS stations report two types of readings:                                                                                                                                                  
  - Auto-obs (every 5 min): whole-degree °C only (e.g., 21°C, 22°C) — this is running_auto_max_c
  - METAR (at :53 each hour): 0.1°C precision via T-group (e.g., 21.7°C)

  The auto-obs are coarser but more frequent. The running max of whole-°C auto readings is tracked as running_auto_max_c in backtest.py:88-96.

  2. The Rounding Problem

  Kalshi settles on NWS CLI reports which give whole-degree °F. The conversion from °C max to °F is:

  naive_f = round(max_c * 9/5 + 32)

  But a whole-°C auto reading (e.g., 22°C) maps to two possible °F values because NWS tracks internally at 0.1°C. For example, 22°C could be anything from 21.5°C to 22.4°C, which spans 70.7°F
   to 72.3°F — so the settlement could be 71°F or 72°F. This is what c_to_possible_f() computes.

  3. The Offset Model

  The bracket model (bracket_model.py) predicts which °F the settlement will actually be, as an offset from naive_f:
  - offset = -1: settlement = naive_f - 1 (rare, the true max was slightly below the auto reading boundary)
  - offset = 0: settlement = naive_f (most common)
  - offset = +1: settlement = naive_f + 1 (true max was just above the next °C boundary)

  4. Two-Stage Architecture

  Stage 1 (auto-only GBM): Uses only features available from auto-obs — peak shape, dwell count at max °C, naive_is_high (whether naive_f is the higher of the two candidate °F values),
  fractional position of the C→F conversion, etc. Outputs 3-class probabilities P(offset=-1), P(offset=0), P(offset=+1).

  Stage 2 (combined RF): Takes METAR features (when available) + stacked stage-1 probabilities + consensus features. METAR at 0.1°C precision can directly resolve the ambiguity in many cases.
   Outputs refined 3-class probabilities.

  Clamp rules then override model output for structurally certain cases (e.g., if METAR confirms the temp is ≥0.25°C above the rounding boundary, force offset to +1).

  5. get_probability() — Mapping to Brackets

  Given the offset probabilities and a list of Kalshi bracket bounds, it:

  1. Computes naive_f = round(max_c * 9/5 + 32) from the current max_c (auto-obs max °C)
  2. Maps each offset to a candidate temp: {-1: naive_f-1, 0: naive_f, +1: naive_f+1}
  3. For each bracket (low, high), sums the offset probabilities for candidates that fall within [low, high]

  So if naive_f = 72 and the bracket is (72, 73), it gets P(offset=0) + P(offset=+1) because both 72 and 73 fall in that range.

  # backtest peak

    This is a separate question: given the current running auto max °C, will the actual 24h max end up ≥1°C higher? If yes, naive_f will shift and the right bracket changes entirely.

  - Label: will_increase_c = 1 if actual_max_c >= running_auto_max_c + 1
  - Uses time-of-day, momentum, forecast gap, and meteorological features
  - If will_increase_c = 1, the bot should wait before betting (the peak hasn't been reached yet)
  - If will_increase_c = 0, it's safe to apply the bracket model to the current running_auto_max_c

  In Practice (strategy.py)

  The strategy extracts current auto-obs features, calls get_probability() with the nearby Kalshi brackets, and bets on the bracket with highest model probability — but only if the peak
  detection model says the temperature is done rising.

  ======================================================================
  AdaBoost — DETAILED RESULTS
======================================================================
               precision    recall  f1-score   support

  no_increase       0.89      0.94      0.91     10742
will_increase       0.79      0.66      0.72      3818

     accuracy                           0.86     14560
    macro avg       0.84      0.80      0.82     14560
 weighted avg       0.86      0.86      0.86     14560


clamps

==========================================================================================
SUMMARY: PERFECT OR NEAR-PERFECT RULES
==========================================================================================
  n_possible_f == 1 (exact)                     n= 152  -1= 0.0%  0=92.8%  +1= 7.2%  clamp -1 -> 0 (100%)
  naive_is_low & n_poss==2                      n= 679  -1= 0.1%  0=45.8%  +1=54.1%  clamp -1→0 (99.9%)
  naive_is_high == 1                            n= 740  -1=46.1%  0=48.5%  +1= 5.4%  clamp +1→0 (95%)
  metar_above_boundary                          n= 396  -1= 0.0%  0=46.7%  +1=53.3%  clamp -1→0 (99%)
  metar_gap_c >= 0.25                           n= 188  -1= 0.0%  0= 0.0%  +1=100.0%  force +1 (100%)
  metar_gap_c >= 0.2 & nih==0                   n= 183  -1= 0.0%  0= 0.0%  +1=100.0%  force +1 (100%)
  single_peak==1 & nih==0                       n= 139  -1= 0.7%  0=96.4%  +1= 2.9%  clamp +1→0? (96%)
  single_peak==1 & nih==1                       n=  95  -1=93.7%  0= 4.2%  +1= 2.1%  strong -1 signal (94%)
  consec>=30 & nih==0                           n= 115  -1= 0.0%  0= 9.6%  +1=90.4%  force +1 (90.4%)
  gap_below==1                                  n= 370  -1= 0.3%  0=45.4%  +1=54.3%  clamp -1→0 (0.3%)
  gap_above==1                                  n= 358  -1=43.9%  0=50.0%  +1= 6.1%  clamp +1→0 (94%)
  metar_gap_c<=-0.5 & nih==1                    n= 178  -1=89.3%  0= 9.6%  +1= 1.1%  force -1 (89.3%)
  metar_confirm>=0.2 & nih==0                   n= 183  -1= 0.0%  0= 0.0%  +1=100.0%  force +1 (100%)