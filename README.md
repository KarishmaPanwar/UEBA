# UEBA
UEBA: Approaches and Strategies

In the realm of data analytics, User and Entity Behavior Analytics (UEBA) has emerged as a powerful approach for understanding and interpreting user behavior across various applications. Beyond its foundational role in cybersecurity, UEBA extends its capabilities to areas such as fraud detection, insider threat detection, and operational efficiency. By leveraging advanced machine learning algorithms, UEBA systems analyze patterns and anomalies in user behavior, providing valuable insights and enabling proactive decision-making. This blog delves into the core algorithms powering UEBA and explores their diverse applications.

To demonstrate the power of UEBA, we'll use a credit card transactions dataset that contains over 1.85 million records. Each transaction includes details such as the amount, merchant category, and demographic data about the cardholder. By applying various machine learning algorithms to this rich dataset, we can uncover unique patterns that help us understand user behavior, different spending patterns, and identify anomalies or suspicious activities. 

Link: https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset/data

Clustering for Anomaly Detection
Clustering is a technique used in data analysis to group sets of similar observations in a dataset. It is effective for anomaly detection because it identifies patterns and structures in data that may not be immediately apparent. By segmenting users into behavioral groups, organizations can pinpoint unusual behavior that warrants further investigation. Techniques like K-Means clustering,  a popular clustering algorithm, group similar user behavior profiles together, revealing distinct segments like frequent buyers versus casual browsers. K-Means works by partitioning the data into K clusters, where each data point belongs to the cluster with the nearest mean. This allows for identification of outliers that may indicate malicious activity. 

In the realm of UEBA, researchers have explored various clustering algorithms to reveal insightful patterns. A recent study  by Artioli et al. (2024) thoroughly investigates traditional and emerging clustering algorithms within the context of UEBA, using three user behavior-related datasets. The study examined both classical and contemporary clustering methods, with a focus on their scalability and suitability for UEBA scenarios through hyper-parameter tuning. Three algorithms were tested: 

K Nearest Neighbor (KNN): Assigns data points to the group of their nearest neighbors.
Scalable Spare Subspace Clustering by Orthogonal Matching Pursuit (SSC-OMP): Identifies sparse subspaces in high-dimensional data to group similar data points.
Elastic Net Subspace Clustering (EnSC): Combines sparse and low-rank subspace clustering to identify complex patterns in data.

The findings revealed that while KNN is straightforward and easy to implement, its effectiveness was limited by its computational intensity and sensitivity to the choice of k value. KNN was less scalable for large datasets compared to more advanced methods like SSC-OMP and EnSC. However, it provided a useful baseline for comparison and helped highlight the advantages and limitations of more sophisticated clustering algorithms in the context of UEBA.

'''
from sklearn.cluster import KMeans

# Standardize features
features = ['amt', 'city_pop', 'lat', 'long', 'age']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x=df['amt'], y=df['city_pop'], hue=df['cluster'], palette='Set1', s=100, alpha=0.7)
plt.title('Customer Segmentation Using K-Means')
plt.xlabel('Transaction Amount')
plt.ylabel('City Population')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
'''
![K means](image-2.png)

'''
# Split data for KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Extract the transaction hour from the timestamp
df['transaction_hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour

# Define features (X) and target (y)
X = df[['amt', 'transaction_hour']]
y = df['is_fraud']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN for anomaly detection
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict anomalies
y_pred = knn.predict(X_test)

# Plot KNN predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['amt'], y=X_test['transaction_hour'], hue=y_pred, palette='coolwarm')
plt.title('Anomalous Transactions Detected by KNN')
plt.xlabel('Transaction Amount')
plt.ylabel('Transaction Hour')
plt.show()
'''
![KNN](image-4.png)

Classifying Behavior for Threat Detection

Classification techniques are a powerful tool for identifying potential threats by leveraging historical data to train models that can accurately predict future behavior. One such technique is the Naive Bayes classifier, which can predict user categories (e.g., high-risk versus low-risk) or classify behavior as normal versus anomalous for security applications. In the context of UEBA, classification techniques like Naive Bayes can be useful in identifying and differentiating between legitimate and malicious users based on their behavior patterns.

Other approaches to classifying user behavior include using classical ML models such as Decision Trees and Random Forests. A research study by  Ranjan and Kumar (2022)  demonstrated the effectiveness of using big data analytics and machine learning to predict malicious users based on application-layer logs. The study used Random Forests and Decision Trees to process real-time data and classify users into high-risk and low-risk categories, effectively identifying suspicious IP addresses and user identification tokens (UITs). This approach enables proactive threat detection and targeted security measures, showcasing the potential of these methods to enhance organizational security by analyzing deviations in user behavior.

'''
# Naive Bayes model for fraud detection
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict fraud
y_pred_nb = nb.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Fraud Detection (Naive Bayes)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred_nb))
print(df.columns) 
'''
![Naive Bayes](image-3.png)

Modeling Sequential Behavior 

Modeling Sequential Behavior Techniques are designed to analyze and identify patterns in time-series data, such as user navigation paths or transaction sequences. Hidden Markov Models (HMMs) are a type of modeling sequential behavior technique that identify hidden states and transition probabilities in sequential data, capturing complex patterns and transitions in time-series data. This technique is particularly useful in UEBA where analyzing user behavior over time can detect subtle changes that may signal malicious activity.

Building on the capabilities of Modeling Sequential Behavior Techniques, research has further demonstrated their effectiveness in identifying anomalous behavior. For example, Zheng (2016)  proposed an effective contrast sequential pattern mining approach for taxpayer behavior analysis, which can be adapted for identifying anomalous sequences in user behavior. By analyzing taxpayers' static attributes and transaction sequences, Zheng predicted self-finalized (compliant) versus non-self-finalized (non-compliant) debt cases, identifying behavior patterns that indicated a higher likelihood of cases being self-finalized. This approach improved resource allocation and proactive intervention, enhancing the efficiency of tax debt collection.

'''
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder


# Encode the 'category' column into numerical values for HMM
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Prepare data for HMM (amt, transaction_hour, and encoded category)
hmm_data = df[['amt', 'cc_num', 'transaction_hour', 'category_encoded']]

# Group transactions by card number to simulate sequences
sequences = hmm_data.groupby('cc_num').apply(lambda x: x.values).tolist()

# Initialize the Hidden Markov Model with 3 components (3 hidden states)
model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=100)

# Fit the HMM model on all sequences concatenated
concatenated_data = np.concatenate(sequences)
model.fit(concatenated_data)

# Predict hidden states for each transaction sequence
hidden_states = model.predict(concatenated_data)

# Add hidden states back to the original DataFrame
df['hidden_state'] = np.concatenate([model.predict(seq) for seq in sequences])

# Visualize Hidden States: Transaction patterns by hidden states
plt.figure(figsize=(12, 8))

# Visualization: Transaction Amount vs Hidden States
plt.figure(figsize=(12, 8))
sns.scatterplot(x=df['amt'], y=df['transaction_hour'], hue=df['hidden_state'], palette='Set1', s=100)
plt.title('Transaction Amount vs Hidden States (HMM)')
plt.xlabel('Transaction Amount')
plt.ylabel('Transaction Hour')
plt.grid(True)
plt.show()

# Visualization: Count of Transactions per Hidden State
plt.figure(figsize=(8, 6))
sns.countplot(x='hidden_state', data=df, palette='Set2')
plt.title('Count of Transactions by Hidden State')
plt.xlabel('Hidden State')
plt.ylabel('Count')
plt.show()
'''
![hmm](image-5.png)
![hmm](image-6.png)


Uncovering Patterns with Apriori and Association Rules
Apriori is a popular algorithm for association rule learning, designed to identify frequent item sets and generate association rules. These techniques uncover patterns such as "users who viewed X also viewed Y" or "customers who bought A often buy B," enhancing strategies for cross-selling, bundle recommendations, and product placements.

Leveraging the power of association rule learning, researchers have explored ways to refine and optimize this technique for better insights. In their study, Sarker and Salim (2018) utilized association rule mining to extract behavioral association rules from smartphone data. While association rule learning is effective for revealing hidden relationships in data, it often produces a large number of redundant rules, complicating the decision-making process. To address this, Sarker and Salim proposed a method to identify and eliminate redundancy in association rules, generating a concise set of non-redundant behavioral rules. Their approach involved creating an association generation tree to prioritize contexts and streamline the extraction of significant rules. Through experiments on mobile phone datasets, they demonstrated that their method outperforms traditional association rule algorithms, providing clearer insights into user behavior.

'''
from mlxtend.frequent_patterns import apriori, association_rules

# Create a binary matrix for the 'category' column
basket = pd.get_dummies(df['category'])

# Group by 'cc_num' to represent transactions
basket = df.groupby(['cc_num'])['category'].apply(lambda x: ','.join(x)).str.get_dummies(',')

# Apply Apriori for frequent pattern mining
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

# Display results
print(frequent_itemsets)

# Visualization of frequent itemsets using a bar plot
# Plot only top 10 frequent itemsets by support
frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

plt.figure(figsize=(10, 6))
sns.barplot(x='support', y='itemsets_str', data=frequent_itemsets.sort_values('support', ascending=False).head(10))
plt.title('Top 10 Frequent Itemsets by Support')
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.show()
'''
![Apriori](image-1.png)

UEBA: A Proactive Approach to Security
UEBA is an advanced cybersecurity approach that uses machine learning, algorithms, and statistical analyses to detect real-time network attacks. Salitin and Zolait (2018) emphasize the value of behavior analytics in securing networks against novel threats, such as zero-day attacks. By creating baseline profiles of user and entity behavior, UEBA can identify deviations that signal potential malicious activity. This proactive approach allows organizations to respond to threats swiftly, safeguarding critical assets and minimizing damage.

UEBA's strength lies in its ability to detect insider threats through continuous monitoring and analysis of user behavior and network activities. It utilizes techniques like risk scoring, activity mapping, and profiling to prioritize alerts and automate responses, reducing the time-to-response cycle. This is crucial as malware constantly evolves to avoid detection, with a significant percentage of malware signatures appearing only once. Traditional signature-based security measures are increasingly ineffective, making behavior-based detection essential.

Khaliq, Tariq, and Masood (2018) discuss the various UEBA approaches and the features of top commercial solutions, highlighting the need for a clear understanding of an organization's assets and the specific activities that need monitoring. They point out that while UEBA technology is robust and widely adopted, its deployment can be complex and requires advanced analytic techniques to manage effectively.

Conclusion

In conclusion, UEBA stands as a versatile and essential tool across multiple domains, offering deep insights into user and entity behavior. By employing sophisticated machine learning techniques and real-time analytics, UEBA systems can detect anomalies, enhance security, and drive operational efficiency. The proactive nature of UEBA allows organizations to anticipate and respond to potential threats and irregularities before they escalate. As technology continues to evolve, advancing UEBA systems through research and development will be crucial in unlocking new applications and maintaining their effectiveness in an ever-changing digital landscape.

Works Cited

Artioli P, Maci A and Magrì A (2024) A comprehensive investigation of clustering algorithms for User and Entity Behavior Analytics. Front. Big Data 7:1375818. doi: 10.3389/fdata.2024.1375818

Rohit Ranjan, Shashi Shekhar Kumar (2022).  User behavior analysis using data analytics and machine learning to predict malicious user versus legitimate user. https://www.sciencedirect.com/science/article/pii/S2667295221000246

Zhao, X., Keikhosrokiani, P. (2022). Sales prediction and product recommendation model through user behavior analytics. Computers, Materials & Continua, 70(2), 3855-3874. https://doi.org/10.32604/cmc.2022.019750

Sarker, I.H., Salim, F.D. (2018). Mining User Behavioral Rules from Smartphone Data Through Association Analysis. In: Phung, D., Tseng, V., Webb, G., Ho, B., Ganji, M., Rashidi, L. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2018. Lecture Notes in Computer Science(), vol 10937. Springer, Cham. https://doi.org/10.1007/978-3-319-93034-3_36

M. A. Salitin and A. H. Zolait, "The role of User Entity Behavior Analytics to detect network attacks in real time," 2018 International Conference on Innovation and Intelligence for Informatics, Computing, and Technologies (3ICT), Sakhier, Bahrain, 2018, pp. 1-5, doi: 10.1109/3ICT.2018.8855782. keywords: {Security;Monitoring;Real-time systems;Machine learning;Analytical models;Adaptation models;Technological innovation;traditional security approaches;Security attacks;User Entity Behavior Analytics;Real-time}

S. Khaliq, Z. U. Abideen Tariq and A. Masood, "Role of User and Entity Behavior Analytics in Detecting Insider Attacks," 2020 International Conference on Cyber Warfare and Security (ICCWS), Islamabad, Pakistan, 2020, pp. 1-6, doi: 10.1109/ICCWS48432.2020.9292394. keywords: {Security;Organizations;Servers;Computer crime;Monitoring;Machine learning;Engines;Security Information and Event Management;Machine Learning (ML);Artificial Intelligence (AI)}
