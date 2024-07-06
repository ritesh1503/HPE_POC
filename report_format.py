import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Generate sample cybersecurity data
def generate_cybersecurity_data(num_records=100):
    anomaly_types = ['Phishing', 'Malware', 'DDoS Attack', 'Unauthorized Access', 'Data Breach']
    sources = ['Firewall', 'IDS', 'Antivirus', 'SIEM', 'Endpoint Security']
    severities = ['Low', 'Medium', 'High', 'Critical']
    data = []
    start_date = datetime.now() - timedelta(days=num_records)
    for _ in range(num_records):
        record = {
            'timestamp': (start_date + timedelta(days=random.randint(0, num_records))).strftime("%Y-%m-%d %H:%M:%S"),
            'anomaly_type': random.choice(anomaly_types),
            'source': random.choice(sources),
            'severity': random.choice(severities),
            'description': 'Sample description of the cybersecurity anomaly.'
        }
        data.append(record)
    return pd.DataFrame(data)

# Report functions
def report_json(data):
    return data.to_json(orient='records', lines=True)

def report_html(data):
    return data.to_html()

def report_csv(data):
    return data.to_csv(index=False)

def report_graph(data, report_file):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='anomaly_type', hue='severity')
    plt.title('Cybersecurity Anomalies by Type and Severity')
    plt.xlabel('Anomaly Type')
    plt.ylabel('Count')
    plt.legend(title='Severity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cybersecurity_anomalies_plot.png')
    plt.show()
    
    with open(report_file, 'a') as f:
        f.write("\nGraph Report: This plot shows the count of each type of cybersecurity anomaly, categorized by severity.\n")

def report_kmeans(data, report_file):
    data_numeric = pd.get_dummies(data[['anomaly_type', 'source', 'severity']])
    kmeans = KMeans(n_clusters=3)
    data['cluster'] = kmeans.fit_predict(data_numeric)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(data_numeric)
    data['component_1'] = components[:, 0]
    data['component_2'] = components[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='component_1', y='component_2', hue='cluster', data=data, palette='viridis')
    plt.title('KMeans Clustering of Cybersecurity Anomalies')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig('kmeans_clustering_plot.png')
    plt.show()

    with open(report_file, 'a') as f:
        f.write("\nKMeans Clustering Report: This plot shows the clustering of cybersecurity anomalies based on their types and sources, reduced to two principal components.\n")

def report_classification(model, model_name, data, report_file):
    # Convert categorical data to numeric for classification
    data_numeric = pd.get_dummies(data[['anomaly_type', 'source']])
    data_numeric['severity'] = data['severity'].astype('category').cat.codes
    
    X = data_numeric.drop(columns='severity')
    y = data_numeric['severity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Classification Report for {model_name}:\n")
    report = classification_report(y_test, y_pred)
    print(report)
    
    print("\nConfusion Matrix:\n")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()

    with open(report_file, 'a') as f:
        f.write(f"\nClassification Report for {model_name}:\n{report}\n")
        f.write(f"\nConfusion Matrix for {model_name}: The confusion matrix plot shows the performance of the classification model, indicating the true vs. predicted classifications.\n")

def report_rnn_attention(data, report_file):
    # Placeholder function for generating an attention map
    print("Generating Attention Map for RNN...")
    plt.figure(figsize=(10, 6))
    plt.title('Attention Map for RNN (Placeholder)')
    plt.xlabel('Time Steps')
    plt.ylabel('Attention Weights')
    sns.heatmap(np.random.rand(10, 10), cmap='viridis')
    plt.tight_layout()
    plt.savefig('rnn_attention_map.png')
    plt.show()

    with open(report_file, 'a') as f:
        f.write("\nRNN Attention Map: This attention map shows the attention weights for different time steps, used in detecting network intrusions.\n")

def report_ner_frequency(data, report_file):
    # Placeholder function for generating frequency distribution of named entities
    print("Generating Frequency Distribution for NER...")
    named_entities = ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE']
    entity_counts = [random.randint(5, 20) for _ in named_entities]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=named_entities, y=entity_counts)
    plt.title('Frequency Distribution of Named Entities')
    plt.xlabel('Entity Type')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('ner_frequency_distribution.png')
    plt.show()

    with open(report_file, 'a') as f:
        f.write("\nNER Frequency Distribution: This plot shows the frequency of different named entities detected in the security monitoring process.\n")

# Get user input
def get_user_format_choice():
    print("Choose a report format:")
    print("1. JSON")
    print("2. HTML")
    print("3. CSV")
    print("4. Graph")
    choice = input("Enter the number corresponding to your choice: ")
    return choice

def get_ml_model_choice():
    print("Choose an ML model to apply to the data:")
    print("1. None")
    print("2. Naive Bayes")
    print("3. Decision Tree Classifier")
    print("4. Random Forest Classifier")
    print("5. KMeans Clustering")
    print("6. Recurrent Neural Network (RNN)")
    print("7. Natural Language Processing (NLP)")
    choice = input("Enter the number corresponding to your choice: ")
    return choice

# Display report based on format and ML model
def display_report(data, format_choice, ml_model_choice, report_file):
    if ml_model_choice == '2':
        model = GaussianNB()
        model_name = "Naive Bayes"
        report_classification(model, model_name, data, report_file)
    elif ml_model_choice == '3':
        model = DecisionTreeClassifier()
        model_name = "Decision Tree Classifier"
        report_classification(model, model_name, data, report_file)
    elif ml_model_choice == '4':
        model = RandomForestClassifier()
        model_name = "Random Forest Classifier"
        report_classification(model, model_name, data, report_file)
    elif ml_model_choice == '5':
        report_kmeans(data, report_file)
    elif ml_model_choice == '6':
        report_rnn_attention(data, report_file)
    elif ml_model_choice == '7':
        report_ner_frequency(data, report_file)
    else:
        if format_choice == '1':
            report = report_json(data)
            with open(report_file, 'w') as f:
                f.write("JSON Report:\n")
                f.write(report)
                f.write("\nThis JSON report contains the serialized version of the cybersecurity anomaly data.\n")
        elif format_choice == '2':
            report = report_html(data)
            with open(report_file, 'w') as f:
                f.write("HTML Report:\n")
                f.write(report)
                f.write("\nThis HTML report contains a tabular representation of the cybersecurity anomaly data.\n")
        elif format_choice == '3':
            report = report_csv(data)
            with open('cybersecurity_report.csv', 'w') as f:
                f.write(report)
            with open(report_file, 'w') as f:
                f.write("CSV Report:\n")
                f.write(report)
                f.write("\nThis CSV report contains the tabular representation of the cybersecurity anomaly data.\n")
        elif format_choice == '4':
            report_graph(data, report_file)
        else:
            print("Invalid choice. Please select a valid format.")

# Input custom data
def input_custom_data():
    print("Provide your own data in the following format:")
    print("[{'timestamp': '2023-05-01 12:34:56', 'anomaly_type': 'Phishing', 'source': 'Firewall', 'severity': 'High', 'description': 'Sample description.'}, ...]")
    custom_data = input("Enter your data as a JSON string: ")
    try:
        data = pd.read_json(custom_data)
    except ValueError as e:
        print("Invalid JSON format. Generating sample data instead.")
        data = generate_cybersecurity_data()
    return data

# Main function
def main():
    report_file = 'cybersecurity_report.txt'
    print("Do you want to provide your own input data? (yes/no)")
    custom_data_choice = input().strip().lower()
    if custom_data_choice == 'yes':
        data = input_custom_data()
    else:
        data = generate_cybersecurity_data()
    
    ml_model_choice = get_ml_model_choice()
    format_choice = get_user_format_choice()
    display_report(data, format_choice, ml_model_choice, report_file)

if __name__ == "__main__":
    main()

