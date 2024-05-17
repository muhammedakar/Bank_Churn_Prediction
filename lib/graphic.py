import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display



def plot_numerical_col(dataframe, num_cols, plot_type='hist'):
    num_cols_count = len(num_cols)
    num_rows = num_cols_count // 3
    num_rows += 1 if num_cols_count % 3 != 0 else 0  # Eğer sütun sayısı 3'e tam bölünmüyorsa bir ek satır oluştur.


    col_groups = [num_cols[i:i+12] for i in range(0, num_cols_count, 12)]

    for group in col_groups:
        fig, axes = plt.subplots(num_rows, 3, figsize=(10, 10))
        axes = axes.flatten()

        for i, col in enumerate(group):
            if plot_type == 'hist':
                sns.histplot(data=dataframe[col], ax=axes[i])
            elif plot_type == 'kde':
                sns.kdeplot(data=dataframe[col], ax=axes[i])
            elif plot_type == 'box':
                sns.boxplot(data=dataframe[col], ax=axes[i])
            else:
                print("Geçersiz grafik türü. Lütfen 'hist', 'kde', veya 'box' olarak belirtin.")
                return
            axes[i].set_xlabel(col)

        for j in range(len(group), num_rows * 3):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def plot_categoric_col(dataframe, cat_cols):
    cat_cols_count = len(cat_cols)
    cat_rows = cat_cols_count // 3
    cat_rows += 1 if cat_cols_count % 3 != 0 else 0  # Eğer sütun sayısı 3'e tam bölünmüyorsa bir ek satır oluştur.

    fig, axes = plt.subplots(cat_rows, 3, figsize=(10, 10), squeeze=True)
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.countplot(data=dataframe, x=col, ax=axes[i], order=dataframe[col].value_counts().index)
        axes[i].set_xlabel(col)

    plt.tight_layout()
    plt.show()

# Kullanım örneği:
# plot_categoric_col(df, ["col1", "col2", "col3", "col4", "col5", "col6"])
