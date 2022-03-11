import random

import pandas as pd
from datetime import date, datetime, timedelta
import yfinance as yf
# import pymongo
import mongoengine as me
from dotenv import load_dotenv
import os
import numpy as np


class BalanceWarning(Exception):

    def __init__(self, balance, change, message="Balance is too low for this action"):
        self.balance = balance
        self.change = change
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Balance: {self.balance}, Change: {self.change} -> {self.message}"


class SellWarning(Exception):

    def __init__(self, ticker, current_amount, sell_amount, message="You cannot sell more than you have."):
        self.ticker = ticker
        self.current_amount = current_amount
        self.sell_amount = sell_amount
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Ticker: {self.ticker}, Current Holdings: {self.current_amount}, " \
               f"Sell Amount: {self.sell_amount} -> {self.message}"


class Assets(me.Document):
    AssetId = me.IntField(required=True, unique=True)
    BuyDate = me.DateTimeField(required=True)
    Ticker = me.StringField(required=True, unique=True, max_length=10)
    # AssetName = me.StringField(required=True, max_length=30)
    CurrentPrice = me.FloatField(required=True)
    Amount = me.FloatField(required=True)
    PositionValue = me.FloatField(required=True)
    PfWeight = me.FloatField(required=True)
    meta = {
        'indexes': ['AssetId']
    }


class Trades(me.Document):
    TradeId = me.IntField(required=True, unique=True)
    AssetId = me.IntField(required=True)
    Ticker = me.StringField(required=True, max_length=10)
    # AssetName = me.StringField(required=True, max_length=30)
    Type = me.StringField(required=True, max_length=5)
    TradeTime = me.DateTimeField(required=True)
    TradePrice = me.FloatField(required=True)
    Amount = me.FloatField(required=True)
    TradeValue = me.FloatField(required=True)
    meta = {
        'indexes': ['TradeId']
    }

"""
class Balance(me.Document):
    BalanceId = me.IntField(required=True, unique=True)
    AsOfDate = me.DateTimeField(required=True)
    Balance = me.FloatField(required=True)
    Action = me.StringField(required=True, max_length=30)
    CapitalId = me.FloatField()
    TradeId = me.IntField()
    FundSize = me.FloatField(required=True)
    meta = {
        'indexes': ['BalanceId']
    }
"""

class Balance(me.Document):
    Balance = me.DictField(required=True)  # Balance contains: {AsOfDate: Balance, FundSize, ActionId, ActionType}


class Capital(me.Document):
    CapitalId = me.IntField(required=True)
    CapitalAmount = me.FloatField(required=True)
    Note = me.StringField()


class Prices(me.Document):
    PriceId = me.IntField(required=True, unique=True)
    Ticker = me.StringField(required=True)
    PriceData = me.DictField(required=True)  # PriceData contains: Keys=PriceDate, Values=Price
    meta = {
        'indexes': ['PriceId']
    }


class NAV(me.Document):
    NavData = me.DictField(required=True)  # NavData contains: Keys=Date, Values=NAV, FundSize
    # FundSize = me.FloatField(required=True)


class Logging(me.Document):
    LogId = me.IntField(required=True, unique=True)
    Activity = me.StringField(required=True)
    # User = me.StringField(required=True)
    meta = {
        'indexes': ['LogId']
    }


def logger(func):
    def wrapper(*args, **kwargs):
        if Logging.objects.first() is None:
            log_id = 1
        else:
            log_id = Logging.objects().order_by('-LogId').first().LogId + 1
        activity = f"{datetime.now()} INFO:root: Calling {func.__name__} with arguments: {', '.join(args)}."
        Logging(LogId=log_id, Activity=activity).save()
        func(*args, **kwargs)
        log_id = log_id + 1
        activity = f"{datetime.now()} INFO:root: Success calling {func.__name__}"
        Logging(LogId=log_id, Activity=activity).save()

    return wrapper


class Asset:

    def __init__(self, ticker: str):
        """Init function that takes the Ticker of a asset

        Args:
            ticker (str): Ticker for the Asset. Could be bond or stock
        """
        self.ticker = ticker
        self.data = yf.Ticker(self.ticker)

    def historical_data(self, start_date: str, end_date: str = date.today().strftime("%Y-%m-%d")) -> pd.DataFrame:
        return self.data.history(start=start_date, end=end_date)

    def stats(self, stat: str):
        return self.data.get_info().get(stat)


class Portfolio:

    def __init__(self, capital: int = 100000, currency='USD', start: datetime = datetime(2021, 1, 1)):

        self._init_capital = capital
        self._capital = capital
        self._NAV = 100
        self._currency = currency
        self._start = start

        # Load .env variables
        load_dotenv()
        mongo_password = os.getenv('MONGO_DB_PASSWORD')
        data_url = f"mongodb+srv://sjh660:{mongo_password}@cluster0.lf4zu.mongodb.net/Cluster0?retryWrites=true&w=majority"

        # Create MongoDB connection
        client = me.connect(
            host=data_url,
            alias='default'
        )

        # Check if the collections are empty and we have to start a new portfolio
        if Balance.objects.first() is None:
            BalanceDict = dict(AsOfDate=start, Balance=capital, ActionType='Start capital',
                               ActionId=1, FundSize=capital)
            Balance(Balance=BalanceDict).save()

        if Capital.objects.first() is None:
            Capital(CapitalId=1, CapitalAmount=capital, Note='StartCapital').save()

        if NAV.objects.first() is None:
            NAV(NavData=dict(AsOfDate=start, NAV=100, FundValue=capital)).save()

        if Prices.objects.first() is None:
            Prices(PriceId=1, Ticker='CASH', PriceData=dict(Date=start, Price=1)).save()

        if Assets.objects.first() is None:
            Assets(AssetId=1, Ticker='CASH', CurrentPrice=1, Amount=capital, PositionValue=capital,
                   PfWeight=1, BuyDate=start).save()

    @property
    def init_capital(self):
        return self._init_capital

    @property
    def capital(self):
        return self._capital

    @property
    def currency(self):
        return self._currency

    @property
    def nav(self):
        return self._NAV

    def single_trade(self, ticker: str, trade_time: str, amount: int or float, action_type: str):
        if action_type == 'sell':
            amount = -amount
        trade_time = datetime.strptime(trade_time, '%Y-%m-%d %H:%M:%S')
        first_date = Balance.objects().first().Balance
        try:
            first_date = list(first_date.get('AsOfDate'))[0]
        except TypeError:
            first_date = first_date.get('AsOfDate')
        if trade_time < first_date:
            raise ValueError('You cannot buy something before you started the portfolio')
        asset = Asset(ticker)
        data = pd.DataFrame()
        price_time = (trade_time + timedelta(days=1)).strftime('%Y-%m-%d')
        while len(data) == 0:
            try:
                data = asset.historical_data(start_date=price_time, end_date=price_time)
                price = data['Close'].values[0]
                trade_date = data.index[0]
            except IndexError:
                price_time = (datetime.strptime(price_time, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

        balance_dict = Balance.objects().first().Balance
        balance = balance_dict.get('Balance')
        if not isinstance(balance, int):
            balance = balance[-1]

        if balance < price*amount:
            raise BalanceWarning(balance[-1], price*amount)

        old_asset = Assets.objects(Ticker=ticker).first()

        try:
            trade_id = Trades.objects().order_by('-TradeId').first().TradeId + 1
        except AttributeError:
            trade_id = 1
        try:
            try:
                asset_id = old_asset.AssetId
            except AttributeError:
                asset_id = Assets.objects().order_by('-AssetId').first().AssetId + 1
        except AttributeError:
            asset_id = 1

        # Register the asset
        try:
            new_amount = old_asset.Amount + amount

            # Make sure we aren't selling more than we have.
            if new_amount < 0:
                raise SellWarning(ticker=ticker, current_amount=old_asset.Amount, sell_amount=amount)
            elif new_amount == 0:
                Assets.objects(Ticker=ticker).delete()
            else:
                new_position = new_amount * price
                # Make sure that asset.BuyDate is the first date from when we hold the asset.
                if trade_date < old_asset.BuyDate:
                    buy_date = trade_date
                else:
                    buy_date = old_asset.BuyDate
                update_parameters = {"$set":
                    {
                        "Amount": round(new_amount, 2),
                        "PositionValue": new_position,
                        "BuyDate": buy_date
                    }
                }
                Assets.objects(Ticker=ticker).update(__raw__=update_parameters)

        except AttributeError:
            # Make sure we don't sell something we don't have
            if amount < 0:
                raise SellWarning(ticker=ticker, current_amount=0, sell_amount=amount)
            fund_size = Assets.objects.sum('PositionValue')
            pf_weight = (price * amount) / (fund_size + price * amount)
            asset_dict = {"AssetId": asset_id, "Ticker": asset.ticker,  # "AssetName": name,
                          "CurrentPrice": price, 'Amount': round(amount, 2), 'PositionValue': price * amount,
                          "PfWeight": pf_weight, 'BuyDate': trade_date}
            Assets(**asset_dict).save()

        # Register the trade
        trade = dict(TradeId=trade_id, Ticker=asset.ticker,  # AssetName=name,
                     Type=action_type, TradeTime=trade_date, Amount=amount, TradePrice=price,
                     TradeValue=price * amount, AssetId=asset_id)
        Trades(**trade).save()

        self.update_prices()
        self.update_balance(trade_id, trade_date, 'Trade')
        self.update_nav()
        self.update_pf_weights()

    def buy_asset(self, ticker: str, buy_time: str, amount: float):
        self.single_trade(ticker=ticker, trade_time=buy_time, amount=amount, action_type='buy')

    # TODO: Add so that you cannot sell asset before you buy. i.e., in relation to trade_time
    def sell_asset(self, ticker: str, sell_time: str, amount: float):
        self.single_trade(ticker=ticker, trade_time=sell_time, amount=amount, action_type='sell')

    def capital_injection(self, injection: int):
        # TODO: Make it so that it also updated Balance collection + Cash
        self._capital += injection

    @staticmethod
    def update_price_one(ticker: str, buy_date: datetime):
        asset = Asset(ticker)
        data = asset.historical_data(start_date=buy_date.strftime("%Y-%m-%d"))

        prices_series = pd.Series(data.Close, index=data.index)
        idx = pd.date_range(start=prices_series.index[0], end=date.today())
        prices_series = prices_series.reindex(idx, method='ffill')
        prices_df = prices_series.to_frame()

        # Get amount of asset
        trade_data = Trades.objects(Ticker=ticker)
        trade_dates = [i.TradeTime.date() for i in trade_data]
        trade_amount = [i.Amount for i in trade_data]
        trade_df = pd.DataFrame(data=trade_amount, index=trade_dates, columns=['Amount'])

        prices_df = prices_df.join(trade_df)
        prices_df['cum. amount'] = prices_df['Amount'].cumsum()
        prices_df = prices_df.ffill().fillna(value=0, axis=1)
        prices_df = prices_df[~prices_df.index.duplicated(keep='last')]
        prices_df.columns = ['Price', 'Amount', 'Cum. Amount']
        prices_df = prices_df[['Price', 'Cum. Amount']]
        prices_df = prices_df[prices_df['Cum. Amount'] != 0]

        ts = prices_df.index.values.astype(datetime) // 10 ** 9
        dates = [datetime.utcfromtimestamp(i) for i in ts]

        data_dict = dict(Date=dates, Price=list(prices_df.Price.values), Amount=list(prices_df['Cum. Amount'].values))

        try:
            price_id = Prices.objects().order_by('-PriceId').first().PriceId + 1
        except AttributeError:
            price_id = 1

        if Prices.objects(Ticker=ticker).count() == 1:
            update_parameters_prices = {"$set":
                {
                    "PriceData": data_dict
                }
            }
            Prices.objects(Ticker=ticker).update(__raw__=update_parameters_prices)
        else:
            Prices(PriceId=price_id, Ticker=ticker, PriceData=data_dict).save()

        update_parameters_asset = {"$set":
            {
                "CurrentPrice": prices_df.Price[-1],
                "PositionValue": Assets.objects(Ticker=ticker).first().Amount * prices_df.Price[-1]
            }
        }

        Assets.objects(Ticker=ticker).update(__raw__=update_parameters_asset)

    def update_prices(self):
        list_of_assets = []
        for i in Assets.objects:
            list_of_assets.append(i)

        # Remove cash from the list of assets
        list_of_assets = [i for i in list_of_assets if i.AssetId != 1]

        for i in list_of_assets:
            self.update_price_one(i.Ticker, i.BuyDate)

        for i in Prices.objects:
            if Assets.objects(Ticker=i.Ticker).first() is None:
                Prices.objects(Ticker=i.Ticker).delete()

        self.update_cash()

    @staticmethod
    def update_cash():
        first_date = Balance.objects().first().Balance
        try:
            first_date = list(first_date.get('AsOfDate'))[0]
        except TypeError:
            first_date = first_date.get('AsOfDate')
        d_range = pd.date_range(start=first_date, end=date.today())
        cash_df = pd.DataFrame(1, index=d_range, columns=['Price'])
        temp_df = pd.DataFrame()
        for i in Trades.objects().order_by('TradeTime'):
            ddf = pd.DataFrame({'Type': [i.Type], 'TradeValue': [i.TradeValue]}, index=[i.TradeTime])
            temp_df = pd.concat([temp_df, ddf])
        cash_df = cash_df.join(temp_df)
        start_capital = Capital.objects(CapitalId=1).first().CapitalAmount
        cash_df.loc[first_date, 'Amount'] = start_capital
        cash_df[['TradeValue', 'Amount']] = cash_df[['TradeValue', 'Amount']].fillna(0)
        TradeValue_temp = cash_df.groupby(cash_df.index).sum()['TradeValue']
        cash_df = cash_df[~cash_df.index.duplicated(keep='last')]
        cash_df['TradeValue'] = TradeValue_temp
        cash_df['Amount'] = cash_df['Amount'] - cash_df['TradeValue']
        cash_df['Cum. Amount'] = cash_df['Amount'].cumsum()
        cash_df['Price'] = 1.0

        ts = cash_df.index.values.astype(datetime) // 10 ** 9
        dates = [datetime.utcfromtimestamp(i) for i in ts]

        data_dict = dict(Date=dates, Price=list(cash_df['Price'].values), Amount=list(cash_df['Cum. Amount'].values))
        update_parameters_cash = {"$set":
            {
                "PriceData": data_dict
            }
        }
        Prices.objects(Ticker='CASH').update(__raw__=update_parameters_cash)

        # Update Cash in Assets
        new_cash = list(cash_df['Cum. Amount'].values)[-1]
        update_parameters = {"$set":
            {
                "Amount": new_cash,
                "PositionValue": new_cash
            }
        }
        Assets.objects(Ticker='CASH').update(__raw__=update_parameters)

    def update_balance(self, action_id: int, action_time: datetime, action_type: str):
        cash_dict = Prices.objects(Ticker='CASH').first().PriceData
        cash_df = pd.DataFrame(cash_dict)

        balance_dict = Balance.objects().first().Balance
        try:
            balance_df = pd.DataFrame(balance_dict)
        except ValueError:
            balance_df = pd.DataFrame({'AsOfDate': balance_dict.get('AsOfDate'), 'Balance': [balance_dict.get('Balance')],
                                       'FundSize': [balance_dict.get('FundSize')], 'ActionId': [balance_dict.get('ActionId')],
                                       'ActionType': [balance_dict.get('ActionType')]})
        ts = balance_df.AsOfDate.values.astype(datetime) // 10 ** 9
        dates = [datetime.utcfromtimestamp(i) for i in ts]
        balance_df.index = [i.date() for i in dates]
        balance_df.drop(['AsOfDate', 'Balance'], axis=1, inplace=True)
        try:
            first_date = list(balance_dict.get('AsOfDate'))[0]
        except TypeError:
            first_date = balance_dict.get('AsOfDate')
        d_range = pd.date_range(start=first_date, end=date.today())
        temp_df = pd.DataFrame(dict(FundSize=None, ActionId=None, ActionType=None), index=d_range)
        temp_df.update(balance_df, overwrite=True)
        temp_df.FundSize = temp_df.FundSize.ffill()

        cash_df = cash_df.set_index('Date').rename({'Amount': 'Balance'}, axis=1)['Balance'].to_frame()
        init_capital = Capital.objects(CapitalId=1).first().CapitalAmount
        cash_df['FundSize'] = init_capital
        cash_df[['ActionId', 'ActionType']] = None
        cash_df.loc[action_time, 'ActionId'] = action_id
        cash_df.loc[action_time, 'ActionType'] = action_type
        cash_df.update(temp_df, overwrite=True)

        balance_id = Balance.objects().first().id
        ts2 = cash_df.index.values.astype(datetime) // 10 ** 9
        dates2 = [datetime.utcfromtimestamp(i) for i in ts2]
        data_dict = dict(AsOfDate=dates2, Balance=list(cash_df['Balance'].values),
                         FundSize=list(cash_df['FundSize'].values), ActionId=list(cash_df['ActionId'].values),
                         ActionType=list(cash_df['ActionType'].values))
        update_parameters = {"$set":
            {
                "Balance": data_dict
            }
        }
        Balance.objects(id=balance_id).update(__raw__=update_parameters)

    # TODO: Create a method for getting exchange rates and calculate prices in USD.
    # TODO: Also consider creating an exchange rate PnL

    @staticmethod
    def update_pf_weights():
        list_of_assets = []
        for i in Assets.objects:
            list_of_assets.append(i)

        fund_size = Assets.objects.sum('PositionValue')

        for asset in list_of_assets:
            ticker = asset.Ticker
            position_value = asset.PositionValue
            new_pf_weight = position_value / fund_size

            update_parameters = {"$set":
                {
                    "PfWeight": new_pf_weight
                }
            }

            Assets.objects(Ticker=ticker).update(__raw__=update_parameters)

    @staticmethod
    def update_nav():
        temp_dict = dict()
        for asset in Prices.objects():
            ticker = asset.Ticker
            dates, prices, amounts = asset.PriceData.values()
            temp_dict[ticker] = dict(Date=dates, Value=np.multiply(prices, amounts))
        nav_df = pd.DataFrame(temp_dict.get('CASH'))
        nav_df.set_index('Date', inplace=True)
        for k, v in temp_dict.items():
            if k != 'CASH':
                ddf = pd.DataFrame(v.get('Value'), index=v.get('Date'))
                nav_df = nav_df.join(ddf, rsuffix=k)
        nav_df['FundValue'] = nav_df.sum(axis=1)

        # Get FundSize (not appreciated with price changes)
        balance_dict = Balance.objects().first().Balance
        try:
            balance_df = pd.DataFrame(balance_dict)
        except ValueError:
            balance_df = pd.DataFrame(
                {'AsOfDate': balance_dict.get('AsOfDate'), 'Balance': [balance_dict.get('Balance')],
                 'FundSize': [balance_dict.get('FundSize')], 'ActionId': [balance_dict.get('ActionId')],
                 'ActionType': [balance_dict.get('ActionType')]})
        balance_df.set_index('AsOfDate', inplace=True)
        nav_df['NAV'] = nav_df['FundValue'] / balance_df['FundSize'] * 100

        nav_id = NAV.objects().first().id
        ts = nav_df.index.values.astype(datetime) // 10 ** 9
        dates = [datetime.utcfromtimestamp(i) for i in ts]
        update_parameters = {"$set":
            {
                "NavData": dict(AsOfDate=dates, NAV=list(nav_df['NAV'].values),
                                FundValue=list(nav_df['FundValue'].values))
            }
        }

        NAV.objects(id=nav_id).update(__raw__=update_parameters)

    @staticmethod
    def graph_nav(start: str = None, end: str = None):

        nav = NAV.objects().first().NavData
        nav_df = pd.DataFrame(nav).set_index('AsOfDate')

        if start:
            start_dt = datetime.strptime(start, '%Y-%m-%d')
        else:
            start_dt = nav_df.index[0]

        if end:
            start_dt = datetime.strptime(end, '%Y-%m-%d')
        else:
            end_dt = nav_df.index[-1]

        nav_df.loc[start_dt:end_dt, 'NAV'].plot();

    @staticmethod
    def delete_collection(collection):
        for i in collection.objects:
            i.delete()


if __name__ == "__main__":

    port = Portfolio()
    # port.buy_asset('TSLA', '2021-05-01 00:00:00', 5)
    # port.sell_asset('TSLA', '2021-12-01 00:00:00', 1)
    # port.buy_asset('AAPL', '2021-12-01 00:00:00', 1)
    #port.buy_asset('BTC-USD', '2021-01-01 00:00:00', 0.5)
    #port.sell_asset('BTC-USD', '2021-08-01 00:00:00', 0.1)
    port.graph_nav()
    port.update_nav()
    port.update_prices()
    port.update_pf_weights()

    import random
    list_of_assets = ['TSLA', 'AAPL', 'MSFT', 'BTC-USD', 'XRP-USD']
    dates = pd.date_range('2021-01-01 00:00:00', '2022-01-01 00:00:00')
    for date in dates:
        trade_or_not = np.random.binomial(1, 0.1)
        if trade_or_not:
            asset = random.choice(list_of_assets)
            date_str = date.strftime('%Y-%m-%d %H:%M:%S')
            short_date = date_str[:10]
            a = Asset(asset)
            data = pd.DataFrame()
            while len(data) == 0:
                try:
                    data = a.historical_data(start_date=short_date, end_date=short_date)
                    price = data['Close'].values[0]
                    amount = 5000/price
                except IndexError:
                    short_date = (datetime.strptime(short_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            buy_or_sell = np.random.binomial(1, 0.5)
            if buy_or_sell:
                try:
                    port.buy_asset(asset, date_str, amount)
                except BalanceWarning:
                    continue
            else:
                try:
                    port.sell_asset(asset, date_str, amount)
                except SellWarning:
                    continue


    DELETE = False
    if DELETE:
        for j in [Trades, NAV, Balance, Assets, Prices, Capital]:
            Portfolio.delete_collection(j)
