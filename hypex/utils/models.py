from sklearn.linear_model import LinearRegression, Ridge, Lasso

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

CUPAC_MODELS = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
    "lasso": Lasso(),
}
if CATBOOST_AVAILABLE:
    CUPAC_MODELS["catboost"] = CatBoostRegressor(verbose=0)
