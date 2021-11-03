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
        self._format_data()

    def _load_df(self):
        self.df = pd.read_excel(
            io = f"data/{self.year}.xlsx", 
            names = COLUMNS,
            engine = "openpyxl"
        )
    
    def _format_data(self):
        self.df["Date"] = pd.to_datetime(self.df["Date"], format = "'%d-%b-%Y %H:%M:%S'")
        self.df["TimeZone"]  = self.df["TimeZone"].str.replace("'", "")
        self.df["ContractName"]  = self.df["ContractName"].str.replace("'", "")


def realized_variance(r):
    return np.power(r, 2).sum()


def bipower_variation(r):
    return np.pi /2 * np.abs(r * r.shift(1))[1:].sum()


def subsample_apply(r, fun, n):
    return np.mean([fun(r.iloc[k::n]) for k in range(n)])


YEAR = 1990
df = DataLoader(YEAR).df


#### check for dirt
td = df.Date.diff()

# big time diff
td[td>pd.Timedelta("1 min")].sort_values() # looks for weird 

# small time diff
td[td<pd.Timedelta("1 min")].sort_values()

# missing time zones
df.TimeZone.unique()
df.loc[df.TimeZone == "", "TimeZone"] = df.TimeZone.iloc[0]

# missing LogReturn
df.LogReturn.isna().sum()

# missing ContractName
df.loc[df.ContractName == ""].isna().mean()
df = df.loc[df.ContractName != ""]

# missing LogReturn
df.loc[df.LogReturn.isna()]
df = df.loc[~df.LogReturn.isna()]

# check for more than one contract per day

# find days with early close


def format_variances(variances):
    variances.columns = ["rv", "bv", "ssj"]
    variances["interval"] = interval
    variances = variances.reset_index()
    # creating a variable to check the day before the conrtact expires or rolls to see if there are anomalies
    # .fillna is to us the last contract for the last fday of the year which will always have rolled into March.
    variances["rollDate"] = variances["ContractName"] != variances["ContractName"].shift(-1).fillna(method="ffill")
    return variances


date_contract = [df.Date.dt.date, df.ContractName]
gdf = df.groupby(date_contract)
_vars = []


# NAIVE
for interval in range(1,61):
    # using .iloc[0::interval] on each day is going to give us values at each interval
    rv = gdf.apply(lambda x: realized_variance(x.iloc[0::interval].LogReturn))
    bv = gdf.apply(lambda x: bipower_variation(x.iloc[0::interval].LogReturn))
    ssj = np.maximum(rv - bv, 0)
    variances = pd.concat([rv, bv, ssj], axis=1)
    variances = format_variances(variances)
    variances["method"] = "naive"
    _vars.append(variances)


# SUBSAMPLING
for interval in range(1,61):
    rv = gdf.apply(lambda x: subsample_apply(x.LogReturn, realized_variance, interval))
    bv = gdf.apply(lambda x: subsample_apply(x.LogReturn, bipower_variation, interval))
    ssj = np.maximum(rv - bv, 0)
    variances = pd.concat([rv, bv, ssj], axis=1)
    variances = format_variances(variances)
    variances["method"] = "subsampling"
    _vars.append(variances)


vdf = pd.concat(_vars)
vdf.to_csv("results/variance_calculations_{YEAR}.csv")
