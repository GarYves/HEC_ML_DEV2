import numpy as np
import pandas as pd


COLUMNS = [
    "Date",
    "TimeZone",
    "ContractName",
    "Open",
    "Close",
    "High",
    "Low",
    "LastTradedPrice",
    "LagSameContractLastTradedPrice",
    "Volume",
    "NbTrade",
    "LogReturn"
]


class DataLoader:
    def __init__(self, year: int = 1990):
        self.year = year
        self.df = None
        self._load_df()
        self._format_df()

    def _load_df(self):
        self.df = pd.read_excel(
            io = f"data/{self.year}.xlsx", 
            names = COLUMNS,
            engine = "openpyxl"
        )
    
    def _format_df(self):
        self.df["Date"] = pd.to_datetime(self.df["Date"], format = "'%d-%b-%Y %H:%M:%S'")
        self.df["TimeZone"]  = self.df["TimeZone"].str.replace("'", "")
        self.df["ContractName"]  = self.df["ContractName"].str.replace("'", "")


class DataCleaner:
    def __init__(self, df):
        self.df = df
        self._remove_missing_contract_names()
        self._remove_missing_log_returns()
        self._remove_old_contracts_on_roll_dates()

    def _remove_missing_contract_names(self):
        self.df = self.df.loc[~(self.df.ContractName != "") | ~self.df.ContractName.isnull()]

    def _remove_missing_log_returns(self):
        self.df = self.df.loc[~self.df.LogReturn.isna()]

    def _remove_old_contracts_on_roll_dates(self):
        # check for more than one contract per day
        self.df["date"] = pd.to_datetime(self.df["Date"]).dt.date
        _df = self.df[["date","ContractName"]].drop_duplicates()
        
        roll_dates = _df.loc[_df.duplicated("date", keep=False)].date.unique()
        vol_sums = df.groupby(["date", "ContractName"]).NbTrade.sum().to_frame().reset_index()
        rolls = vol_sums.loc[vol_sums.date.isin(roll_dates), :].sort_values(["date", "NbTrade"])
        # remove the contract with the lowest number of trade 
        to_dump_on_roll_dates = rolls.groupby("date").apply(lambda x: x.loc[x["NbTrade"].idxmin()])
        self.df = self.df.loc[
            ~(self.df.date.isin(to_dump_on_roll_dates.date) & 
            self.df.ContractName.isin(to_dump_on_roll_dates.ContractName))
        ]



def realized_variance(r):
    return np.power(r, 2).sum()


def bipower_variation(r):
    return np.pi /2 * np.abs(r * r.shift(1))[1:].sum()


def subsample_apply(r, fun, n):
    return np.mean([fun(r.iloc[k::n]) for k in range(n)])


def format_variances(variances):
    variances.columns = ["rv", "bv", "ssj"]
    variances["interval"] = interval
    variances = variances.reset_index()
    return variances


INTERVALS = [1,2,3,4,5,6,7,8,9,10,15,20,30,45,60,90]
YEARS = [1990, 2001, 2007, 2018]

for YEAR in YEARS:
    df = DataLoader(YEAR).df
    df = DataCleaner(df).df

    date_contract = [df.date, df.ContractName]
    gdf = df.groupby(date_contract)
    _vars = []

    # NAIVE
    for interval in INTERVALS:
        # using .iloc[0::interval] on each day is going to give us values at each interval
        rv = gdf.apply(lambda x: realized_variance(x.iloc[0::interval].LogReturn))
        bv = gdf.apply(lambda x: bipower_variation(x.iloc[0::interval].LogReturn))
        ssj = np.maximum(rv - bv, 0)
        variances = pd.concat([rv, bv, ssj], axis=1)
        variances = format_variances(variances)
        variances["method"] = "naive"
        _vars.append(variances)

    # SUBSAMPLING
    for interval in INTERVALS:
        rv = gdf.apply(lambda x: subsample_apply(x.LogReturn, realized_variance, interval))
        bv = gdf.apply(lambda x: subsample_apply(x.LogReturn, bipower_variation, interval))
        ssj = np.maximum(rv - bv, 0)
        variances = pd.concat([rv, bv, ssj], axis=1)
        variances = format_variances(variances)
        variances["method"] = "subsampling"
        _vars.append(variances)

    vdf = pd.concat(_vars)
    vdf.to_csv(f"results/variance_calculations_{YEAR}.csv")
