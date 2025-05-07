from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

def select_anova(X, y, k):
    """select top k features with ANOVA F-test"""
    selector = SelectKBest(score_func=f_classif, k=k)
    return selector.fit_transform(X, y), selector.get_support(indices=True)

def select_chi2(X, y, k):
    """select top k features with Chi-Square"""
    selector = SelectKBest(score_func=chi2, k=k)
    return selector.fit_transform(X, y), selector.get_support(indices=True)

def select_mutual_info(X, y, k):
    """select top k features with Mutual Information"""
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    return selector.fit_transform(X, y), selector.get_support(indices=True)

FS_METHODS = {
    "anova": select_anova,
    "chi2": select_chi2,
    "mutual_info": select_mutual_info
}
