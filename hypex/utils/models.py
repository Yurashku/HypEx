from sklearn.linear_model import LinearRegression, Ridge, Lasso

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

CUPAC_MODELS = {
    "linear": {
        "pandas": LinearRegression(),
        "polars": None,
    },
    "ridge": {
        "pandas": Ridge(),
        "polars": None,
    },
    "lasso": {
        "pandas": Lasso(),
        "polars": None,
    },
}

if CATBOOST_AVAILABLE:
    CUPAC_MODELS["catboost"] = {
        "pandas": CatBoostRegressor(verbose=0),
        "polars": None,
    }
