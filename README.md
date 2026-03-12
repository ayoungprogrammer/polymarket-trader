# setup

pip install --no-build-isolation -e .


# run bot

sudo tailscale funnel 5000


# Weather Markets


Live weather
https://www.weather.gov/wrh/timeseries?site=KATL


Forecast



TODO:

Debug https://www.weather.gov/wrh/timeseries?site=KATL

Why did this end at 83???


# data

* 6h highs and lows correspond to the 24h highs and lows and vary per station


# Temps

https://www.reddit.com/r/Kalshi/comments/1hfvnmj/an_incomplete_and_unofficial_guide_to_temperature/

Interpreting NWS Time Series: The NWS reports data from two different types of weather stations. They have technical names, but for our purposes, I’ll refer to them as hourly stations, and 5-minute stations. They record and report data very differently, so understanding the difference is critical. 

Hourly stations are constantly recording the temperature, but only report the temperature once at hour. These temperatures are recorded to the nearest 0.1F, then converted to the nearest 0.1C and sent to the NWS. The NWS then reports the temperature celcius, and converts it back to fahrenheit and reports that as well. Technically, the Fahrenheit value you see isn’t correct, because some rounding/conversion error has been introduced by converting it to the nearest 0.1C and back again. But the error is typically minimal. Every 6 to 24 hours, these stations will send the highest temperature they’ve measured to the NWS, and this will be recorded as the high for the day. Importantly, because you only see the temperature at hourly points in time AND because the rounding/conversion error is minor, the daily high will typically be greater than, and sometimes equal to, the highest hourly reading you see.

5-minute stations are very different. They record temperature in 1 minute long averages. 5 of these 1 minute long averages are then averaged into a 5 minute average. This 5 minute average is rounded to the nearest whole degree fahrenheit, which is then converted to the nearest whole degree celsius and sent to the NWS. Note: This is apparently due to the DOS limitations in the software running the stations, which, are you serious? But I digress. When the NWS reports the temperature in fahrenheit, it converts the celsius value to fahrenheit, with no regard for the original fahrenheit value. Again, yes, this is for real. This process introduces a significant amount of rounding and conversion error that makes the NWS reported fahrenheit values nearly meaningless, and can be a degree or more higher or lower than the official high for the day ends up being. Instead, what you’d want to do is work backwards to determine which original fahrenheit values would then get rounded and converted to the celsius value you see on the NWS time series. Most celsius values reported by the 5-minute stations could be the result of two different actual fahrenheit temperatures. If those two fahrenheit temperatures fall in the same two degree range on Kalshi, you can be confident whatever the underlying temperature, it’s within that range. However, if the two underlying values span two different Kalshi ranges, you’ll need to use your intuition and understanding of the temperatures to make an educated guess which of the two fahrenheit values occurred, and from that, which Kalshi range to bet on.

How the actual high is determined by the NWS: The NWS records the high for the day by checking both stations for the highest temperature recorded over the course of the entire day and taking the higher of the two. The hourly station records temperature constantly (I imagine this means 1 minute averages, like the 5-minute stations, but I’m not 100% certain) and the 5-minute station records 1 minute averages. The highest of these values, rounded to the nearest whole degree fahrenheit, is the high for the day. These values are not converted to celsius and back again, avoiding rounding/conversion error that plagues the 5-minute stations. And they are not averaged over 5 minutes, so the high can potentially be higher than any value that could be rounded to the whole degree celsius reported by the 5-minute stations.