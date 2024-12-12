########################################################################################################################
## ISYE 6740 project
## Predicting Water Source Type of Tapwater exposed to Per- and Polyfluoroalkyl Substances (PFAS) in the United States
## Predicting whether PFAS concentration in Tapwater is above Maximum Contamination Levels
########################################################################################################################


## import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


## declare constants: EPA standards
MCL = {
    "PFOA": 4.0,
    "PFOS": 4.0,
    "PFHxS": 10.0,
    "PFNA": 10.0,
    "HFPO-DA;\nGenX": 10.0,
    "mix": 1.0,  # mixtures containing two or more of PFHxS, PFNA, HFPO-DA, and PFBS; in terms of hazard index
    "PFBS": 2000.0
}


## define classifiers
names = [
    "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
    "Logistic Regression", "Decision Tree", "Random Forest", "Neural Net",
    "AdaBoost", "Naive Bayes", "QDA"
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    LogisticRegression(solver='liblinear'),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]


## count occurrences per class
def count_classes(labels):
    print(labels.value_counts())
    print('----------------------------------------')


## count NAs per column
def count_NA(df):
    print("Number of NAs per PFAS type:")
    print(df.isna().sum())
    print('----------------------------------------')


## standardization
def standardize_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)


## remove rows that have outliers in at least one column
def remove_outliers(df):
    ## keep if at least 3 stds from the mean
    return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]


def show_corr_matrix(df):
    sns.heatmap(df.corr(), annot=True, fmt=".1f")
    plt.tight_layout()
    plt.show()


## remove columns with all zeros
def remove_zero_cols(df):
    return df.loc[:, (df != 0).any(axis=0)]


## two sample t-test with equal variance
def two_sample_t_test(A, B, alpha=0.05):
    res = stats.ttest_ind(a=A, b=B, equal_var=True)
    print("two sample t-test:")
    if res.pvalue > alpha:
        print(f"{res.pvalue} > {alpha}, failed to reject null hypothesis")
    else:
        print(f"{res.pvalue} < {alpha}, reject null hypothesis")
    print('----------------------------------------')


## PCA
def apply_PCA(df, comp=6):
    pca = PCA(n_components=comp)
    return pca.fit_transform(df)


## Isomap
def apply_Isomap(df):
    embedding = Isomap(n_components=6, n_neighbors=5)
    return embedding.fit_transform(df)


## plot reduced dimensional data
def plot_reduced_dim(df, labels, name, q2):
    fig, axs = plt.subplots(figsize=(6,14), nrows=3)
    axs[0].set_xlabel('PC1')
    axs[0].set_ylabel('PC2')
    axs[1].set_xlabel('PC3')
    axs[1].set_ylabel('PC4')
    axs[2].set_xlabel('PC5')
    axs[2].set_ylabel('PC6')

    ## question 2: above MCL?
    if q2:
        sns.scatterplot(df[:, 0], df[:, 1], labels.map({0:False, 1:True}), hue_order=[False, True], ax=axs[0])
        sns.scatterplot(df[:, 2], df[:, 3], labels.map({0:False, 1:True}), hue_order=[False, True], ax=axs[1])
        sns.scatterplot(df[:, 4], df[:, 5], labels.map({0:False, 1:True}), hue_order=[False, True], ax=axs[2])
    ## question 1: public/private?
    else:
        sns.scatterplot(df[:, 0], df[:, 1], labels, hue_order=["Public", "Private"], ax=axs[0])
        sns.scatterplot(df[:, 2], df[:, 3], labels, hue_order=["Public", "Private"], ax=axs[1])
        sns.scatterplot(df[:, 4], df[:, 5], labels, hue_order=["Public", "Private"], ax=axs[2])

    fig.tight_layout()
    # plt.savefig("out/"+ name +".png")
    plt.show()


## PCA method 1
def PCA_replace_nd_and_NA_with_zero(df, labels, q2=False):
    ## replace NA and nd with zeroes
    df_clean = df.fillna(0.0)
    df_clean = df_clean.replace("nd", 0.0)

    ## remove outliers
    df_clean = remove_outliers(df_clean)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## get indices of remaining rows
    labels_clean = labels[df_clean.index]

    ## standardize data
    df_std = standardize_data(df_clean)

    ## apply PCA
    X = apply_PCA(df_std)

    ## plot data
    plot_reduced_dim(X, labels_clean, "PCA_replace_nd_and_NA_with_zero", q2)


## Isomap method 1
def Isomap_replace_nd_and_NA_with_zero(df, labels, q2=False):
    ## replace NA and nd with zeroes
    df_clean = df.fillna(0.0)
    df_clean = df_clean.replace("nd", 0.0)

    ## remove outliers
    df_clean = remove_outliers(df_clean)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## get indices of remaining rows
    labels_clean = labels[df_clean.index]

    ## standardize data
    df_std = standardize_data(df_clean)

    ## apply Isomap
    X = apply_Isomap(df_std)

    ## plot data
    plot_reduced_dim(X, labels_clean, "Isomap_replace_nd_and_NA_with_zero", q2)


## PCA method 2
def PCA_remove_NA_replace_nd(df, labels, q2=False):
    ## remove columns with NAs
    df_clean = df.drop(columns=['PFPeA', 'PFDA', 'PFHpS', 'PFDS', 'PFPrS', 'HFPO-DA;\nGenX'])
    ## replace nd with zero
    df_clean = df_clean.replace("nd", 0.0)

    ## remove outliers
    df_clean = remove_outliers(df_clean)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## show correlation matrix
    # if q2:
    #     show_corr_matrix(df_clean)

    ## get indices of remaining rows
    labels_clean = labels[df_clean.index]

    ## standardize data
    df_std = standardize_data(df_clean)

    ## apply PCA
    X = apply_PCA(df_std)

    ## apply t-test
    if(q2 == False):
        category = labels_clean.map({"Public":False, "Private":True})
        two_sample_t_test(X[category, 0], X[~category, 0])

    ## plot data
    plot_reduced_dim(X, labels_clean, "PCA_remove_NA_replace_nd", q2)


## Isomap method 2
def Isomap_remove_NA_replace_nd(df, labels, q2=False):
    ## remove columns with NAs
    df_clean = df.drop(columns=['PFPeA', 'PFDA', 'PFHpS', 'PFDS', 'PFPrS', 'HFPO-DA;\nGenX'])
    ## replace nd with zero
    df_clean = df_clean.replace("nd", 0.0)

    ## remove outliers
    df_clean = remove_outliers(df_clean)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## get indices of remaining rows
    labels_clean = labels[df_clean.index]

    ## standardize data
    df_std = standardize_data(df_clean)

    ## apply Isomap
    X = apply_Isomap(df_std)

    ## plot data
    plot_reduced_dim(X, labels_clean, "Isomap_remove_NA_replace_nd", q2)


## PCA method 3
def PCA_remove_some_NA_replace_nd(df, labels, q2=False):

    ## remove columns with NAs
    df_clean = df.drop(columns=['PFDS', 'PFPrS', 'HFPO-DA;\nGenX'])
    ## remove rows with NAs
    df_clean = df_clean.dropna()
    ## replace nd with zero
    df_clean = df_clean.replace("nd", 0.0)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## remove outliers
    df_clean = remove_outliers(df_clean)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## get indices of remaining rows
    labels_clean = labels[df_clean.index]

    ## standardize data
    df_std = standardize_data(df_clean)

    ## apply PCA
    X = apply_PCA(df_std)

    ## plot data
    plot_reduced_dim(X, labels_clean, "PCA_remove_some_NA_replace_nd", q2)


## Isomap method 3
def Isomap_remove_some_NA_replace_nd(df, labels, q2=False):
    ## remove columns with NAs
    df_clean = df.drop(columns=['PFDS', 'PFPrS', 'HFPO-DA;\nGenX'])
    ## remove rows with NAs
    df_clean = df_clean.dropna()
    ## replace nd with zero
    df_clean = df_clean.replace("nd", 0.0)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## remove outliers
    df_clean = remove_outliers(df_clean)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## get indices of remaining rows
    labels_clean = labels[df_clean.index]

    ## standardize data
    df_std = standardize_data(df_clean)

    ## apply Isomap
    X = apply_Isomap(df_std)

    ## plot data
    plot_reduced_dim(X, labels_clean, "Isomap_remove_some_NA_replace_nd", q2)


def above_MCL(row):
    ## define hazard index based on EPA guidelines
    hazard_index = (row['HFPO-DA;\nGenX'] / MCL['HFPO-DA;\nGenX']) + \
                    (row['PFBS'] / MCL['PFBS']) + \
                    (row['PFNA'] / MCL['PFNA']) + \
                    (row['PFHxS'] / MCL['PFHxS'])

    ## return 1 if above MCL
    if hazard_index > 1:
        return 1
    elif (row['PFOA'] > MCL['PFOA']) or (row['PFOS'] > MCL['PFOS']):
        return 1
    elif (row['PFHxS'] > MCL['PFHxS']) or (row['PFNA'] > MCL['PFNA']) or (row['HFPO-DA;\nGenX'] > MCL['HFPO-DA;\nGenX']):
        return 1

    return 0


def classify_pca(df, labels):
    ## remove columns with NAs
    df_clean = df.drop(columns=['PFPeA', 'PFDA', 'PFHpS', 'PFDS', 'PFPrS', 'HFPO-DA;\nGenX'])
    ## replace nd with zero
    df_clean = df_clean.replace("nd", 0.0)

    ## remove outliers
    df_clean = remove_outliers(df_clean)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## get indices of remaining rows
    labels_clean = labels[df_clean.index]
    ## count classes
    print("Above MCL class counts:")
    count_classes(labels_clean.map({0:False, 1:True}))

    ## standardize data
    df_std = standardize_data(df_clean)

    ## apply PCA
    X = apply_PCA(df_std, comp=2)

    ## split data set into test and train
    X_train, X_test, y_train, y_test = \
        train_test_split(X, labels_clean, test_size=.3, random_state=42)

    ## create mesh grid
    h = .02  ## mesh step size
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ## get CV scores for all the classifiers
    scores = []
    for clf in classifiers:
        scores.append(np.mean(cross_val_score(clf, X, labels_clean, cv=10)))
    print("classifiers CV score wrt to PCA data:")
    print(scores)
    print('----------------------------------------')

    plt.figure(figsize=(15, 20))
    i = 1
    ## plot dataset
    cm = plt.cm.coolwarm
    cm_bright = ListedColormap(['#0096FF', '#FF5733'])  ## blue, orange
    ax = plt.subplot(4, (len(classifiers) + 1) / 4, 1)
    ax.set_title("Input data", size=20)
    ## plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.2, edgecolors='k')
    ## plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    ## iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(4, (len(classifiers) + 1) / 4, i)
        clf.fit(X_train, y_train)

        ## plot the decision boundary
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        ## put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.7)

        ## plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', alpha=0.2)
        ## plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name, size=20)
        i += 1

    plt.tight_layout()
    # plt.savefig("classifiers.png")
    plt.show()


def classify(df, labels):
    ## remove columns with NAs
    df_clean = df.drop(columns=['PFPeA', 'PFDA', 'PFHpS', 'PFDS', 'PFPrS', 'HFPO-DA;\nGenX'])
    ## replace nd with zero
    df_clean = df_clean.replace("nd", 0.0)

    ## remove outliers
    df_clean = remove_outliers(df_clean)

    ## remove columns with all zeros
    df_clean = remove_zero_cols(df_clean)

    ## standardize data
    df_std = standardize_data(df_clean)

    ## apply PCA to remove correlation
    X = apply_PCA(df_std, comp=df_std.shape[1])

    ## get indices of remaining rows
    labels_clean = labels[df_clean.index]

    ## get CV scores for all the classifiers
    scores = []
    for clf in classifiers:
        scores.append(np.mean(cross_val_score(clf, X, labels_clean, cv=10)))
    print("classifiers CV score:")
    print(scores)
    print('----------------------------------------')


########################################################################################################################
## main
########################################################################################################################
if __name__ == '__main__':

    ## load data
    data = pd.read_excel("data/pfas.xlsx", skiprows=[0], skipfooter=1)

    ## drop last column
    df = data.iloc[:, :-1]

    ## parse relevant columns: pfas data and site type
    pfas = df.iloc[:, 4:-4]
    labels = df.iloc[:, 3]


    ####################################################################################################################
    ## Site Type
    ####################################################################################################################
    ## count site types
    print("Site Type counts:")
    count_classes(labels)

    ## count NANs
    count_NA(pfas)

    ## PCA method 1: replace NA and nd with zeroes
    PCA_replace_nd_and_NA_with_zero(pfas, labels)

    ## PCA method 2: remove columns with NAs
    PCA_remove_NA_replace_nd(pfas, labels)

    ## PCA method 3: remove columns with a lot of NAs and remaining rows with NAs
    PCA_remove_some_NA_replace_nd(pfas, labels)

    ## Isomap method 1:
    Isomap_replace_nd_and_NA_with_zero(pfas, labels)

    ## Isomap method 2: remove columns with NAs
    Isomap_remove_NA_replace_nd(pfas, labels)
    
    ## Isomap method 3: remove columns with a lot of NAs and remaining rows with NAs
    Isomap_remove_some_NA_replace_nd(pfas, labels)


    ####################################################################################################################
    ## Above Maximum Contamination Level (MCL)
    ####################################################################################################################

    ## replace NA and nd with zeros
    pfas_clean = pfas.fillna(0.0)
    pfas_clean = pfas_clean.replace("nd", 0.0)

    ## get labels based on EPA standards
    pfas_clean['above MCL'] = pfas_clean.apply(above_MCL, axis=1)

    ## PCA method 1: replace NA and nd with zeroes
    PCA_replace_nd_and_NA_with_zero(pfas, pfas_clean['above MCL'], True)

    ## PCA method 2: remove columns with NAs
    PCA_remove_NA_replace_nd(pfas, pfas_clean['above MCL'], True)

    ## PCA method 3: remove columns with a lot of NAs and remaining rows with NAs
    PCA_remove_some_NA_replace_nd(pfas, pfas_clean['above MCL'], True)

    ## Isomap method 1:
    Isomap_replace_nd_and_NA_with_zero(pfas, pfas_clean['above MCL'], True)

    ## Isomap method 2: remove columns with NAs
    Isomap_remove_NA_replace_nd(pfas, pfas_clean['above MCL'], True)

    ## Isomap method 3: remove columns with a lot of NAs and remaining rows with NAs
    Isomap_remove_some_NA_replace_nd(pfas, pfas_clean['above MCL'], True)

    ## apply classifiers on PCA data to predict above/below MCL --------------------------------------------------------
    classify_pca(pfas, pfas_clean['above MCL'])

    ## apply classifiers on data to predict above/below MCL ------------------------------------------------------------
    classify(pfas, pfas_clean['above MCL'])