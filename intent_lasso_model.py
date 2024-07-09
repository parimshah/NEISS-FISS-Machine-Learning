import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn import linear_model
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from nltk.tokenize import word_tokenize

# Saving data to DataFrame
df = pd.read_csv("C://Users//rohin//Documents//gen//Weil Cornell//Code//intentNEISSData.csv")
narratives = df.CMTX1
intent = df.CLASS_C

# Tokenizing corpus using NLTK
def tokenize_text(text):
    return word_tokenize(text)

# Vectorizing unique words into features with TF IDF values
vectorizer = TfidfVectorizer(tokenizer=tokenize_text, min_df=0.0005)
X = vectorizer.fit_transform(narratives)
features = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(data=X.toarray(), columns=features)
print(len(features))

# Splitting data into missing and non-missing
missing_df = intent[intent == 0]
missing_tfidf = tfidf_df.loc[missing_df.index]

nonmissing_df = intent[intent != 0]
nonmissing_tfidf = tfidf_df.loc[nonmissing_df.index]

X = nonmissing_tfidf.to_numpy()
y = nonmissing_df.to_numpy()

# Splitting data into Training: 70%, Validation: 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training model with 5-fold cross validation
reg = linear_model.Lasso(alpha=0.0001)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val = X_train[train_index], X_train[val_index]
    y_train_fold, y_val = y_train[train_index], y_train[val_index]

    reg.fit(X_train_fold, y_train_fold)

# Predicting validation set
y_preds = np.round(reg.predict(X_val), 0)


# Calculating prediction metrics
accuracy = accuracy_score(y_val, y_preds)
y_val_binarized = label_binarize(y_val, classes=np.unique(y_val))
y_preds_binarized = label_binarize(y_preds, classes=np.unique(y_val))
auc = roc_auc_score(y_val_binarized, y_preds_binarized, average='macro')

# Saving true values & predictions in DataFrame
val_df = pd.DataFrame(data=X_val, columns=features)
val_df['y_val'] = y_val
val_df['predictions'] = y_preds

# Splitting validation DataFrame by true intent values
dfs = {}
for i in range(1, 5):
    indexes = val_df.index[val_df['y_val'] == i].tolist()
    dfs[f'{i}_df'] = val_df.loc[indexes].copy()

# Calculate & print recall values by intent
recalls = []
for intent in [dfs['1_df'], dfs['2_df'], dfs['3_df'], dfs['4_df']]:
    recall = recall_score(intent['y_val'], intent['predictions'], average='weighted', zero_division=0)
    recalls.append(np.round(recall, 3))

print("Reccal by Intent:", recalls)
print("Accuracy:", np.round(accuracy, 3))
print("AUC:", np.round(auc, 3))


# Calculating the percentage & numbers of each intent in complete predicted dataset
missing_df = (pd.DataFrame(np.round(reg.predict(missing_tfidf), 0), columns=["CLASS_C"]))
missing_df.reset_index()
nonmissing_df = pd.DataFrame(nonmissing_df, columns=["CLASS_C"])
nonmissing_df.reset_index()
results = pd.concat([missing_df, nonmissing_df], ignore_index=True)

value_mapping = {
    1: "Unintentnl",
    2: "Assault",
    3: "Suicide",
    4: "Law enforce"
}
results['CLASS_C'] = results['CLASS_C'].replace(value_mapping)

intent_counts = results.CLASS_C.value_counts()
total_count = results.CLASS_C.count()
print("New Intent Values")
for x in intent_counts:
    print(intent_counts[intent_counts == x].index[0])
    print(x)
    print(np.round(x / total_count, 2))


# Calculating the percentage & numbers of each intent in original dataset
value_mapping = {
    0: "Unknown",
    1: "Unintentnl",
    2: "Assault",
    3: "Suicide",
    4: "Law enforce"
}
df['CLASS_C'] = df['CLASS_C'].replace(value_mapping)

intent_counts = df['CLASS_C'].value_counts()
total_count = df['CLASS_C'].count()
print("Oringial Intent Values:")
for x in intent_counts:
    print(intent_counts[intent_counts == x].index[0])
    print(x)
    print(np.round(x / total_count, 2))







