import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.iforest import IForest

def pu_learning(train_feats, test_feats):
    nn = NearestNeighbors(n_neighbors=20, metric='cosine') #
    nn.fit(train_feats)
    dists, _ = nn.kneighbors(test_feats)
    knn_scores = dists.mean(axis=1) 

    num_neg = 30
    pseudo_neg_indices = np.argsort(knn_scores)[-num_neg:]
    pseudo_neg_feats = test_feats[pseudo_neg_indices]

    X_train = np.vstack([train_feats, pseudo_neg_feats])
    y_train = np.array([1] * len(train_feats) + [0] * len(pseudo_neg_feats))
    
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
    )
    clf.fit(X_train, y_train)

    
    final_probs = clf.predict_proba(test_feats)[:, 1]
    threshold = 0.99
    final_labels = (final_probs >= threshold).astype(int)
    return final_labels

def one_class_svm(train_feats, test_feats):
    scaler = StandardScaler()
    clf = OneClassSVM(kernel="rbf", gamma='auto', nu=0.1)
    train_scaled = scaler.fit_transform(train_feats)
    clf.fit(train_scaled)
    test_scaled = scaler.transform(test_feats)
    scores = clf.decision_function(test_scaled)
    # threshold = -0.155
    # pred = (scores > threshold).astype(int)
    top_idx = np.argsort(-scores)[:40]  # 大到小排序
    pred = np.zeros_like(scores, dtype=int)
    pred[top_idx] = 1
    return pred

def iforest_learning(train_feats, test_feats):
    clf = IForest(contamination=0.2, random_state=42) 
    clf.fit(train_feats)
    pred = clf.predict(test_feats)
    return 1 - pred 

def lof_learning(train_feats, test_feats):
    clf = make_pipeline(
        StandardScaler(),
        LocalOutlierFactor(
            n_neighbors=90 ,
            metric="cosine",
            novelty=True
        )
    )
    clf.fit(train_feats)

    train_scores = -clf.decision_function(train_feats)
    test_scores = -clf.decision_function(test_feats)
    threshold = np.percentile(train_scores, 40) 
    pred = (test_scores <= threshold).astype(int) 
    return pred