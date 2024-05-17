import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


def outlier_thresholds(dataframe, column, q1=0.05, q3=0.95):
    """
    Aykiri degerlerin alt limitini ve ust limitini donduren fonksiyon
    Parameters
    ----------
    dataframe: Pandas.series
        Aykiri degerin bulunmasi istediginiz dataframe'i giriniz
    column: str
        Hangi degisken oldugunu belirtiniz
    q1:  float
        Alt limit ceyrekligini belirtin
    q3: float
        Ust limit ceyrekligini belirtin

    Returns
    -------
    low_th: float
        alt eşik değer
    up_th: float
        üst eşik değer
    """
    quartile1 = dataframe[column].quantile(q1)
    quartile3 = dataframe[column].quantile(q3)
    iqr = quartile3 - quartile1
    up_th = quartile3 + 1.5 * iqr
    low_th = quartile1 - 1.5 * iqr
    return low_th, up_th


def grab_col_names(dataframe, cat_th=10, car_th=30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes not in ["O", 'datetime64[ns]']]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    binary_cols = [col for col in dataframe.columns if
                   dataframe[col].dtype not in [int, float] and dataframe[col].nunique() == 2]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'binary_cols: {len(binary_cols)}')

    return cat_cols, num_cols, cat_but_car, binary_cols


def check_outlier(dataframe, column):
    """
    Veri setindeki değişkenlerin içerisinde outlier değer var mı yok mu kontrol eder geriye bool değer döndürür.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    column: str
        Hangi değişken içerisinde sorgulama yapmak istiyorsak o değişkeni giriniz

    Returns
    -------


    """
    if dataframe[column].dtype in ['float', 'int']:
        low, up = outlier_thresholds(dataframe, column)
        if dataframe[(dataframe[column] > up) | (dataframe[column] < low)].any(axis=None):
            return True
    return False


def load_df(path):
    df = pd.read_csv(path)
    return df


def for_check(df, cols):
    """
    Bütün değişkenlerde döner ve bize ilgili değişkende aykırı değer olup olmadığını verir
    Parameters
    ----------
    df: dataframe
        Değişken isimleri alınmak istenilen dataframe
    cols: list
        Kontrol edilmek istenilen değişkenlerin listesini veriniz

    Returns
    -------

    """

    for col in cols:
        print(col, check_outlier(df, col))


def grab_outliers(dataframe, col_name, index=False):
    """
    Bu fonksiyon bize ilgili değişkendeki aykırı değerleri döndürür.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    col_name: str
        Hangi değişkenin bilgilerini istiyorsanız o değişkeni yazınız
    index: bool
        Fonksiyonun geriye aykırı değerleri dönmesini istiyorsanız True verin


    Returns
    -------

    """
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_specific_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Bu fonskiyon istediğmiz kolondaki aykırı değerleri siler
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    col_name: str
        dataframe deki aykırı değerleri silmek istediğimiz değişkenin ismi

    Returns
    -------
        df_without_outliers: dataframe
            Aykırı değerlerin silindiği dataframe i döner

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def remove_top_frequency(dataframe, col_name, threshold=0.01):
    """
    Belirli bir sütundaki en çok tekrar eden değerleri siler.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        İşlenecek veri seti.
    col_name: str
        İşlenecek sütun adı.
    threshold: float, optional
        Silme eşiği, varsayılan olarak %1 olarak ayarlanmıştır.

    Returns
    -------
    dataframe: pandas.DataFrame
        Aykırı değerlerin kaldırıldığı veri seti.
    """
    # Sütun içindeki değerlerin frekanslarını hesapla
    value_counts = dataframe[col_name].value_counts(normalize=True)

    # Eşik değeri aşan en yüksek frekanslı değerleri belirle
    top_values_to_remove = value_counts[value_counts > threshold].index.tolist()

    # Belirlenen değerleri içermeyen veri setini döndür
    dataframe = dataframe[~dataframe[col_name].isin(top_values_to_remove)]

    return dataframe


def remove_all_outliers(df, cols):
    """
    Veri setindeki bütün aykırı değerleri siler
    Parameters
    ----------
    df: dataframe
        Değişken isimleri alınmak istenilen dataframe
    cols: list
        Değişkenlerde bulunan silinmesini istediğiniz aykırı değerli değişkenleri liste halinde veriniz.

    Returns
    -------
        df: dataframe
            Aykırı değerlerden kurtulmuş yeni bir dataframe döner
    """
    for col in cols:
        df = remove_specific_outlier(df, col)
    return df


def replace_with_thresholds(dataframe, variable):
    """
    Belli bir değişkendeki aykırı değerleri baskılamak yerini alt ve üst limitlerle doldurmak için kullanınız.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    variable: str
        Baskılamak istediğiniz değişkenin ismi

    Returns
    -------
        dataframe: dataframe
            Baskılanmış yerini alt ve üst limitlerle doldurulmuş bir dataframe döndürür

    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    return dataframe


def replace_all_outliers(df, cols):
    """
    Veri setindeki bütün değişkenleri gezer ve aykırı değerleri alt ve üst limitlerle doldurur.
    Parameters
    ----------
    df: dataframe
        Değişken isimleri alınmak istenilen dataframe
    cols: list
        Baskılamak istediğiniz değişkenleri liste halinde veriniz

    Returns
    -------
        df: dataframe
            İstenilen değişkenlerdeki aykırı değerleri alt ve üst değerlerle baskılanamış dataframe döndürür.

    """
    for col in cols:
        replace_with_thresholds(df, col)
    return df


def lof(dataframe, th_index=None, neighbors=20):
    """
    İstenilen veri setine LOF (Local Outlier Factor) yöntemini uygular.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    th_index: int
        İçerisine index bilgisi alır o indeksten sonraki verileri kırpar eğer bilgi girilmezse istenilen veriler getirilmez
    neighbors: int
        Komşuluk sayısını belirtiniz.

    Returns
    -------
        df_scores: np.Array
            Geriye değişkenlerin skorlarını döndürür.
    """
    dataframe = dataframe.select_dtypes(include=['float64', 'int64'])
    clf = LocalOutlierFactor(n_neighbors=neighbors)
    clf.fit_predict(dataframe)
    df_scores = clf.negative_outlier_factor_
    if th_index is not None:
        th = np.sort(df_scores)[th_index]
        print(dataframe[df_scores < th])
        print(dataframe.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T)
    return df_scores

# mkdir -p "`python -m site --user-site`"
# python -m site --user-site
# /Users/mrakar/.local/lib/python3.11/site-packages/
