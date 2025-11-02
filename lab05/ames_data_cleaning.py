import click
import pandas as pd


@click.command()
@click.option(
    "--file-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with Ames housing data. It will be transformed, cleaned, and replaced.",
)
def clean_ames_data(file_path: str) -> None:
    df = pd.read_parquet(file_path)

    # remove dots from names to match data_description.txt
    df.columns = [col.replace(".", "") for col in df.columns]

    # remove unnecessary columns, very rare neighborhoods, known outliers
    df = df.drop(["Order", "PID"], axis="columns")
    df = df.loc[~df["Neighborhood"].isin(["GrnHill", "Landmrk"]), :]
    df = df.loc[df["GrLivArea"] <= 4000, :]

    # replace a lot of missing values by reasonable defaults, following
    # https://www.kaggle.com/code/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    replace_all_missing_values(df)

    # encode categorical columns appropriately
    df = encode_categorical_columns(df)

    df.to_parquet(file_path)


def replace_all_missing_values(df: pd.DataFrame) -> None:
    # Alley : data description says NA means "no alley access"
    replace_na(df, "Alley", value="None")

    # BedroomAbvGr : NA most likely means 0
    replace_na(df, "BedroomAbvGr", value=0)

    # BsmtQual etc : data description says NA for basement features is "no basement"
    replace_na(df, "BsmtQual", value="No")
    replace_na(df, "BsmtCond", value="No")
    replace_na(df, "BsmtExposure", value="No")
    replace_na(df, "BsmtFinType1", value="No")
    replace_na(df, "BsmtFinType2", value="No")
    replace_na(df, "BsmtFullBath", value=0)
    replace_na(df, "BsmtHalfBath", value=0)
    replace_na(df, "BsmtUnfSF", value=0)

    # Condition : NA most likely means Normal
    replace_na(df, "Condition1", value="Norm")
    replace_na(df, "Condition2", value="Norm")

    # External stuff : NA most likely means average
    replace_na(df, "ExterCond", value="TA")
    replace_na(df, "ExterQual", value="TA")

    # Fence : data description says NA means "no fence"
    replace_na(df, "Fence", value="No")

    # Functional : data description says NA means typical
    replace_na(df, "Functional", value="Typ")

    # GarageType etc : data description says NA for garage features is "no garage"
    replace_na(df, "GarageType", value="No")
    replace_na(df, "GarageFinish", value="No")
    replace_na(df, "GarageQual", value="No")
    replace_na(df, "GarageCond", value="No")
    replace_na(df, "GarageArea", value=0)
    replace_na(df, "GarageCars", value=0)

    # HalfBath : NA most likely means no half baths above grade
    replace_na(df, "HalfBath", value=0)

    # HeatingQC : NA most likely means typical
    replace_na(df, "HeatingQC", value="Ta")

    # KitchenAbvGr : NA most likely means 0
    replace_na(df, "KitchenAbvGr", value=0)

    # KitchenQual : NA most likely means typical
    replace_na(df, "KitchenQual", value="TA")

    # LotFrontage : NA most likely means no lot frontage
    replace_na(df, "LotFrontage", value=0)

    # LotShape : NA most likely means regular
    replace_na(df, "LotShape", value="Reg")

    # MasVnrType : NA most likely means no veneer
    replace_na(df, "MasVnrType", value="None")
    replace_na(df, "MasVnrArea", value=0)

    # MiscFeature : data description says NA means "no misc feature"
    replace_na(df, "MiscFeature", value="No")
    replace_na(df, "MiscVal", value=0)

    # OpenPorchSF : NA most likely means no open porch
    replace_na(df, "OpenPorchSF", value=0)

    # PavedDrive : NA most likely means not paved
    replace_na(df, "PavedDrive", value="N")

    # PoolQC : data description says NA means "no pool"
    replace_na(df, "PoolQC", value="No")
    replace_na(df, "PoolArea", value=0)

    # SaleCondition : NA most likely means normal sale
    replace_na(df, "SaleCondition", value="Normal")

    # ScreenPorch : NA most likely means no screen porch
    replace_na(df, "ScreenPorch", value=0)

    # TotRmsAbvGrd : NA most likely means 0
    replace_na(df, "TotRmsAbvGrd", value=0)

    # Utilities : NA most likely means all public utilities
    replace_na(df, "Utilities", value="AllPub")

    # WoodDeckSF : NA most likely means no wood deck
    replace_na(df, "WoodDeckSF", value=0)

    # CentralAir : NA most likely means No
    replace_na(df, "CentralAir", value="N")

    # EnclosedPorch : NA most likely means no enclosed porch
    replace_na(df, "EnclosedPorch", value=0)

    # FireplaceQu : data description says NA means "no fireplace"
    replace_na(df, "FireplaceQu", value="No")
    replace_na(df, "Fireplaces", value=0)

    # SaleCondition : NA most likely means normal sale
    replace_na(df, "SaleCondition", value="Normal")

    # Electrical : NA most likely means standard circuit & breakers
    replace_na(df, "Electrical", value="SBrkr")


def replace_na(df: pd.DataFrame, col: str, value) -> None:
    df.loc[:, col] = df.loc[:, col].fillna(value)


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(
        {
            "MSSubClass": {
                20: "SC20",
                30: "SC30",
                40: "SC40",
                45: "SC45",
                50: "SC50",
                60: "SC60",
                70: "SC70",
                75: "SC75",
                80: "SC80",
                85: "SC85",
                90: "SC90",
                120: "SC120",
                150: "SC150",
                160: "SC160",
                180: "SC180",
                190: "SC190",
            },
            "MoSold": {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            },
        }
    )
    df = df.replace(
        {
            "Alley": {"None": 0, "Grvl": 1, "Pave": 2},
            "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
            "BsmtFinType1": {
                "No": 0,
                "Unf": 1,
                "LwQ": 2,
                "Rec": 3,
                "BLQ": 4,
                "ALQ": 5,
                "GLQ": 6,
            },
            "BsmtFinType2": {
                "No": 0,
                "Unf": 1,
                "LwQ": 2,
                "Rec": 3,
                "BLQ": 4,
                "ALQ": 5,
                "GLQ": 6,
            },
            "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "Functional": {
                "Sal": 1,
                "Sev": 2,
                "Maj2": 3,
                "Maj1": 4,
                "Mod": 5,
                "Min2": 6,
                "Min1": 7,
                "Typ": 8,
            },
            "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
            "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
            "PavedDrive": {"N": 0, "P": 1, "Y": 2},
            "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
            "Street": {"Grvl": 1, "Pave": 2},
            "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
        }
    )
    return df


if __name__ == "__main__":
    clean_ames_data()
