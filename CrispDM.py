import httpagentparser
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingClassifier

# Create a data from .json file in the same directory
df = pd.read_json('../../WebstormProjects/fingerprint-frontend/prediction/fingerprintsnew.json')
df = df.drop(['_id'], axis=1)
df = df.drop(['_class'], axis=1)
df = df.drop('process', axis=1)

response = requests.get('http://localhost:8080/api/bot-data')

# get response body as json
fingerprint = response.json()

df_latest = pd.json_normalize(fingerprint)
df_latest = df_latest.drop(['id.timestamp', 'id.date', 'process'], axis=1)

# Concatenate the two dataframes
df = pd.concat([df, df_latest], ignore_index=True)

# Replace "None" with a placeholder value (e.g., 0) in the 'bot' column
df['bot'] = df['bot'].replace({'None': 0})

# Fill NaN values with a default value or use other imputation techniques
df = df.fillna(0)

# Convert the 'languages' column to a more usable format
df = df.drop(['languages'], axis=1)

# Handle the 'distinctiveProps' column containing JSON-like data
df = df.drop(['distinctiveProps'], axis=1)
df = df.drop(['documentElementKeys'], axis=1)
df = df.drop(['functionBind'], axis=1)
df = df.drop(['windowExternal'], axis=1)

# New columns for slimerjs, pahntomjs, headless, electron check if these strings are in the appVersion if yes then 1 else 0
df['slimerjs'] = df['appVersion'].apply(lambda x: 1 if 'slimerjs' in x.lower() else 0)
df['phantomjs'] = df['appVersion'].apply(lambda x: 1 if 'phantomjs' in x.lower() else 0)
df['headless'] = df['appVersion'].apply(lambda x: 1 if 'headless' in x.lower() else 0)
df['electron'] = df['appVersion'].apply(lambda x: 1 if 'electron' in x.lower() else 0)
df = df.drop(['appVersion'], axis=1)

# Convert boolean columns to numeric
bool_columns = ['bot', 'android', 'documentFocus', 'notificationPermissions', 'pluginsArray', 'webDriver', 'slimerjs', 'phantomjs', 'headless', 'electron']
df[bool_columns] = df[bool_columns].astype('bool')

# Handle User Agent
df['userAgent'] = df['userAgent'].apply(lambda x: httpagentparser.detect(x))

# Create new columns from the dictionary in userAgent, if it exists give it a default value of 'Unknown'
df['browserNameUA'] = df['userAgent'].apply(lambda x: x['browser']['name'] if 'browser' in x else 'Unknown')
df['browserVersionUA'] = df['userAgent'].apply(lambda x: x['browser']['version'] if 'browser' in x else 'Unknown')
df['osNameUA'] = df['userAgent'].apply(lambda x: x['os']['name'] if 'os' in x else 'Unknown')
df['platformName'] = df['userAgent'].apply(lambda x: x['platform']['name'] if 'platform' in x else 'Unknown')
df['platformVersion'] = df['userAgent'].apply(lambda x: x['platform']['version'] if 'platform' in x else 'Unknown')

df = df.drop(['userAgent'], axis=1)

# One-hot encode categorical columns
categorical_columns = ['browserEngineKind', 'browserKind', 'webGlVendor', 'webGlRenderer', 'browserNameUA', 'browserVersionUA', 'osNameUA', 'platformName', 'platformVersion']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Sort the columns alphabetically
df_encoded = df_encoded.reindex(sorted(df_encoded.columns), axis=1)

df_latest = df_encoded.iloc[[-1]]
df_latest['bot'] = None
df_encoded = df_encoded.drop(df_encoded.tail(1).index)

# Feature matrix (X) and target variable (y)
X = df_encoded.drop('bot', axis=1)
y = df_encoded['bot']

# Initialize the Decision Tree model
clf = GradientBoostingClassifier(n_estimators=150, max_depth=5)

# Train the model
clf.fit(X, y)

print(clf.predict(df_latest.drop(['bot'], axis=1)))

# Evaluate the model on the test set
# print(confusion_matrix(y_test, clf.predict(X_test)))
# report = classification_report(y_test, clf.predict(X_test), output_dict=True)
# print('Accuracy: ', report['accuracy'])
# print('Precision: ', report['False']['precision'])
# print('Recall: ', report['False']['recall'])
# print('F1 Score: ', report['False']['f1-score'])
# print('Precision: ', report['True']['precision'])
# print('Recall: ', report['True']['recall'])
# print('F1 Score: ', report['True']['f1-score'])
#
# print(balanced_accuracy_score(y_test, clf.predict(X_test)))
