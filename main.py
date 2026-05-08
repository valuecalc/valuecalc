import cloudscraper
import os
import re
import json
import html
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from termcolor import cprint

CACHE_PATH = os.environ["HOME"] + "/.valuecalc"
ALPHASPEAD_BASEURL = "https://www.alphaspread.com/security"

#  ----- Stock data scraping & caching

#  convert alphaspread JSON format to dataframe
def raw_fin_record_to_df(content: dict):
    rows, all_records, f_data = [], [], content["fieldsData"][0]
    for k in f_data.keys():
        if k:
            rows += f_data[k][0]

    for field in rows:
        field_data = field[0]
        name = field_data['name']
                
        for data_col in field_data["values"][0]:
            date = data_col[0]["date"][0].split("T00:")[0]
            value = data_col[0]["value"]
            all_records.append({'date': date, 'metric': name, 'value': value})

    fin_record = pd.DataFrame(all_records)
    #  put dates as indexes
    return fin_record.pivot(index='date', columns='metric', values='value')
    
#  fetch alphaspread financial statements
def get_alphaspread_data(url: str, statement_name: str):
    scraper = cloudscraper.create_scraper()
    response = scraper.get(f"{url}/financials/{statement_name}")

    if response.status_code != 200:
        raise Exception(f"Failed to fetch alphaspread data: status code {response.status_code}")
    
    statement_raw_dict = None
 
    for match in re.finditer(r'wire:snapshot="({.*?})"', response.text):
        raw_json = html.unescape(match.group(1))
        snapshot = json.loads(raw_json)
        data = snapshot.get('data', {})

        if type(data) is dict and data.get("statementCode") == statement_name:
            statement_raw_dict = data
            break
    
    if statement_raw_dict is None:
        raise Exception(f"Failed to find statement data on page {url}")
    
    return raw_fin_record_to_df(statement_raw_dict)

def load_statement(baseurl, statement_name):
    statement = get_alphaspread_data(f"{baseurl}", statement_name)
    statement.index = pd.to_datetime(statement.index)
    return statement

def load_cached_stock_data(ticker: str):
    ticker_cache_basedir = f"{CACHE_PATH}/{ticker}"
    general_data_path = f"{CACHE_PATH}/{ticker}/general_infos.json"

    if not os.path.isdir(ticker_cache_basedir):
        AS_baseurl = build_AS_url(ticker)

        #  fetch price history from yfinance (needs higher granularity to match reporting months)
        stock_data = yf.Ticker(ticker)
        prices = stock_data.history(period="20y", interval="1wk")
        prices.index = prices.index.tz_localize(None).normalize()
        
        #  fetch financial statements
        balance_sheet = load_statement(AS_baseurl, "balance-sheet")

        #  shift balance sheet index for japaneese stock (ex: 2010-03-31 -> 2009-12-31)
        #  allows to merge with quarterly eaning report
        new_index = []
        for date in balance_sheet.index:
            if date.month == 3:
                new_date = date - pd.DateOffset(months=3)
                new_index.append(new_date + pd.offsets.MonthEnd(0))  #  to 12-31
            else:
                new_index.append(date)
        balance_sheet.index = pd.to_datetime(new_index)
        balance_sheet = balance_sheet.groupby(balance_sheet.index).last()

        #  resample other statement to yearly data
        income_statement = load_statement(AS_baseurl, "income-statement").resample('YE').sum()
        cash_flow_statement = load_statement(AS_baseurl, "cash-flow-statement").resample('YE').sum()

        #  combine all statement to get market cap on every statement (will duplicate data to fill)
        combined_fin = pd.concat([balance_sheet, income_statement, cash_flow_statement], axis=1)
        combined_fin = combined_fin.loc[:, ~combined_fin.columns.duplicated()]
        combined_fin = combined_fin.sort_index().ffill()

        #  merge prices with statements and calculate market cap
        combined_fin = pd.merge_asof(
                combined_fin.sort_index(),
                prices[['Close']].sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest'
            )
        combined_fin['Common Shares Outstanding'] = combined_fin['Common Shares Outstanding'].ffill()
        combined_fin['MarketCap'] = combined_fin['Close'] * combined_fin['Common Shares Outstanding']

        #  save in cache
        os.makedirs(ticker_cache_basedir, exist_ok = True)
        combined_fin.to_csv(f"{ticker_cache_basedir}/all_statements.csv")
        f = open(general_data_path, "w")
        f.write(json.dumps(stock_data.info))
        f.close()

        return FinDataWrapper(combined_fin, stock_data.info)
    else:
        f = open(general_data_path, "r")
        general_infos = json.loads(f.read())
        f.close()
        combined_fin = pd.read_csv(f"{ticker_cache_basedir}/all_statements.csv", index_col=0)
        return FinDataWrapper(combined_fin, general_infos)

#  convert yahoo finance tickers to alphaspread compatible base URL
def build_AS_url(ticker):
    parts = ticker.split(".")
    exchange = parts[1]
    as_company_id = parts[0]
    as_country_code = ""
    
    if exchange in ["KS", "KQ"]:
        as_country_code = "krx"
    elif exchange == "T":
        as_country_code = "tse"
    #  TODO: build compat for other exchanges

    if as_country_code == "":
        raise Exception(f"Failed to create alphaspread url for exchange {exchange}")

    return f"{ALPHASPEAD_BASEURL}/{as_country_code}/{as_company_id}"

class FinDataWrapper:
    def __init__(self, unified_df, header):
        self.raw_data = unified_df
        self.header = header
        self.raw_data.index = pd.to_datetime(self.raw_data.index)
    
    def get(self, key):
        if key in self.raw_data.columns:
            return self.raw_data[key].fillna(0)
        else:
            cprint(f"Warning: missing column '{key}', defaulted to '0'. May alter valuation ratios", "yellow")
            return pd.Series(0, index=self.raw_data.index)

#  ----- Stock valuation reporting

def valuation_reporting(fin_data):
    header = fin_data.header

    #  FIXME: balance sheet metrics should not take the last list as earnings may be reported 
    #  the next months after balance sheet publication
 
    #  VANT: Net Tangible Asset Value
    VANT = (
            fin_data.get('Total Assets')
            - fin_data.get('Goodwill')
            - fin_data.get('Intangible Assets')
            - fin_data.get('Total Liabilities')
        )

    #  NCAV: Net Curent Asset Value
    NCAV = fin_data.get('Total Current Assets') - fin_data.get('Total Liabilities')

    #  NNWC : Net-Net Working Capital
    NNWC = (
            fin_data.get('Cash & Cash Equivalents') + 
            fin_data.get('Short-Term Investments') +
            0.75 * fin_data.get('Total Receivables') + 
            0.5 * fin_data.get('Inventory') - 
            fin_data.get('Total Liabilities')
        )

    market_cap = fin_data.get('MarketCap')
    share_count = fin_data.get('Common Shares Outstanding')
    NCAV_per_share = NCAV / share_count
    NNWC_per_share = NNWC / share_count
    VANT_per_share = VANT / share_count

    #  render current valuation results
    last_row = fin_data.raw_data.iloc[-1]
    current_price = int(last_row['Close'])
    currency = header.get("currency")
    company_name = header.get("longName")

    #  Render graphs for asset valuation
    fig, ax1 = plt.subplots(figsize=(14, 8))
    yr_index = fin_data.raw_data.index[:-1]

    ax1.plot(yr_index, fin_data.get('Close').iloc[:-1], label=f"Prix ({currency})", color='blue', linewidth=2)
    ax1.plot(yr_index, NCAV_per_share.iloc[:-1], label='Net Current Asset Value', color='orange', linestyle='--')
    ax1.plot(yr_index, NNWC_per_share.iloc[:-1], label='Net-Net Working Capital', color='red', linestyle=':')
    ax1.plot(yr_index, VANT_per_share.iloc[:-1], label='Net Tangible Asset Value', color='green', linestyle=':')

    current_VANT_per_share = int(VANT_per_share.iloc[-2])
    current_NCAV_per_share = int(NCAV_per_share.iloc[-2])
    current_NNWC_per_share = int(NNWC_per_share.iloc[-2])

    valuation_details = f"Cours actuel: {current_price} {currency}\n"
    
    valuation_details += f" - P/VANT: {round((current_price/current_VANT_per_share), 2)} (VANT= {current_VANT_per_share} {currency})\n"
    valuation_details += f" - P/NCAV: {round((current_price/current_NCAV_per_share), 2)} (NCAV= {current_NCAV_per_share} {currency})\n"
    valuation_details += f" - P/NNWC: {round((current_price/current_NNWC_per_share), 2)} (NNWC= {current_NNWC_per_share} {currency})\n"
    valuation_details += f" - Dividend yield: {header.get('dividendYield', 'unknown')}% (payout ratio: {int(header.get('payoutRatio', 0)* 100)} %)"
    
    ax1.text(0.02, 0.95, valuation_details,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
        )
    ax1.set_ylabel(f"Valeur par action ({currency})", fontweight='bold')
    ax1.legend()
    
    #  render cash flow valuation ratios
    #  TODO: make sure to exclude investing cash flow
    fcf_yield = 100 * fin_data.get('Free Cash Flow') / fin_data.get("MarketCap")

    ax2 = ax1.twinx()
    colors = ['g' if x > 0 else 'r' for x in fcf_yield]
    bars = ax2.bar(fcf_yield.index, fcf_yield, width=120, color=colors, alpha=0.3, label='FCF Yield %', zorder=1)
    #ax2.bar(, fcf_yield, color=colors, alpha=0.3, label='FCF yield')
    ax2.set_ylabel("FCF Yield (%)", color='gray', fontweight='bold')
    
    plt.title(f"{company_name} - {ticker}")

    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    ticker = "7399.T"
    valuation_reporting(load_cached_stock_data(ticker))
