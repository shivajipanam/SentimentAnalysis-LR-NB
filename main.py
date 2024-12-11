import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import time
from datetime import datetime
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from math import log, exp

from collections import defaultdict, Counter

from sklearn.metrics import roc_curve, roc_auc_score, auc

#nltk.download('wordnet')

file1_path = r'archive (2)/twitter_training.csv'  # Replace with the locatoin on
file2_path = r'archive (2)/twitter_validation.csv'  # Replace with your file path

# Parse command-line arguments
def parse_args():
    try:
        algo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        train_size = int(sys.argv[2]) if len(sys.argv) > 2 else 80
        if train_size < 50 or train_size > 80:
            train_size = 80
    except ValueError:
        algo, train_size = 0, 80
    return algo, train_size

# Load and preprocess dataset
def load_dataset(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    #data = data.head(10)            # For debugging
    
    # Define schema (column names) you want to enforce
    schema = ['ID', 'Category', 'Sentiment', 'Text']

    # Read the CSV file, skip the first row, and apply the predefined schema
    data = pd.read_csv(file_path, names=schema, header=None, skiprows=1)
    
    # Drop rows with fewer than 4 non-NaN columns (or values)
    data = data.dropna(thresh=4)
    
    # Drop irrelevant columns
    data = data.drop(columns=['ID', 'Category'])
    
    #print("Dropping irrelevant columns", end='\r', flush=True)              # For debugging      
    data = data[data['Text'].str.strip() != '']
    
    # Drop rows with missing text
    data = data.dropna(subset=['Text']).reset_index(drop=True)
    
    # Normalize text (lowercase and remove special characters)
    data['Text'] = data['Text'].str.lower()  # Convert to lowercase
    data['Text'] = data['Text'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))  # Remove non-alphabetic characters

    #print("Convert corpus to lowercase and remove stop words", end='\r', flush=True)              # For debugging      
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    data['Text'] = data['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Lemmatization
    #print("Lemmatizing using WordNetLemmatizer" , end='\r', flush=True)              # For debugging      
    lemmatizer = WordNetLemmatizer()
    data['Text'] = data['Text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    # Encode labels
    #print("Encode and drop irrelevant labels" , end='\r', flush=True)              # For debugging      
    data['Sentiment'] = data['Sentiment'].map({'Positive': 1, 'Negative': 0})  # Example encoding

    # Remove rows with undefined sentiment (i.e., rows where Sentiment is NaN)
    data = data[data['Sentiment'].notna()]

    # Optionally, convert the Sentiment column to integers if needed
    data['Sentiment'] = data['Sentiment'].astype(int)
    
    # Shuffle the rows of the DataFrame
    #print("Shuffling", end='\r', flush=True)              # For debugging      
    data = data.sample(frac=1).reset_index(drop=True)
    
    data = data[data['Text'].str.strip() != '']
    
    # Drop rows with missing text
    data = data.dropna(subset=['Text']).reset_index(drop=True)
    
    #print(data.head(10))              # For debugging      
    
    return data

# Helper function to Tokenize the text 
def tokenizer(df):
    
    #print("Tokenize", end='\r', flush=True)              # For debugging      
    df['tokens'] = df['Text'].apply(lambda x: x.split())
    
    return df

# Simple non binary bag of words implementation
def nb_bag_of_words_vectorization(df, vocab):
    
    # Convert tokens to vector representation (Bag of Words with Add-1 Smoothing)
    def vectorize(tokens, vocab):
        # Initialize vector with 1 for Add-1 Smoothing
        vector = [0.01] * len(vocab)  # Add-1 smoothing by starting with 1 instead of 0
        for token in tokens:
            if token in vocab:
                idx = list(vocab.keys()).index(token)
                vector[idx] += 1  # Increment based on token frequency
        return vector

    #print("Calculating Bag of Words for vectorization", end='\r', flush=True)        # Debug
    df['vector'] = df['tokens'].apply(lambda tokens: vectorize(tokens, vocab))
    
    return df

# Helper function to build vocabulary
def build_vocab(df):
    # Build vocabulary: A set of all unique tokens in the dataset
    #print("Building Vocabulary with top 200 words", end='\r', flush=True)
    
    # Flatten all tokens into one list to count word frequencies
    all_tokens = [token for tokens in df['tokens'] for token in tokens]

    # Count word frequencies using Counter
    word_freq = Counter(all_tokens)

    # Get the top 200 most frequent words and their counts
    most_common_words = dict(word_freq.most_common(200))

    # `vocab` will be a dictionary where key = word, value = count (frequency)
    vocab = most_common_words

    return vocab

# Implement TF-IDF (Term Frequency-Inverse Document Frequency) vectorization 
def tf_idf_vectorization(df, vocab):
    
    N = len(df)
    
    #print("Calculating Term Frequencies for TF-IDF vectorization", end='\r', flush=True)        # Debug
    df['tf'] = df['tokens'].apply(lambda tokens: {token: tokens.count(token) / len(tokens) for token in tokens})
    
    # Inverse document frequency
    #print("Calculating TF-IDF for vectorization", end='\r', flush=True)        # Debug
    idf = {word: np.log(N / (1 + sum([1 for tokens in df['tokens'] if word in tokens]))) for word in vocab}

    # TF-IDF for each document
    def tf_idf_vector(tokens, tf, idf, vocab):
        vector = [0] * len(vocab)
        for word in tokens:
            if word in vocab:
                idx = list(vocab.keys()).index(word)
                vector[idx] = tf[word] * idf[word]
        return vector

    df['vector'] = df.apply(lambda row: tf_idf_vector(row['tokens'], row['tf'], idf, vocab), axis=1)
    
    return df

def calculate_metrics(y_true, y_pred):
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    sensitivity = recall_score(y_true, y_pred)  # Recall is the same as Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred)
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)
    
    # Print the metrics
    print(f"Number of true positives: {tp}")
    print(f"Number of true negatives: {tn}")
    print(f"Number of false positives: {fp}")
    print(f"Number of false negatives: {fn}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Negative Predictive Value: {negative_predictive_value:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F-score: {f_score:.4f}")
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': negative_predictive_value,
        'accuracy': accuracy,
        'f1_score': f_score
    }

# Implement Naïve Bayes Classifier
class NaiveBayes:
    def __init__(self, vocab):
        self.prior = {}
        self.cond_prob = defaultdict(lambda: defaultdict(float))
        self.vocab = vocab
        self.class_counts = {}
        self.regular_prob = {}
        self.classes = []
    
    def train(self, X, y):
        """
        Trains the Naive Bayes model using the given document-term matrix X and class labels y.
        
        The method performs the following steps:
        1. **Calculate Prior Probabilities (P(Class))**:
        - The prior probabilities for each class are computed by counting the occurrences of each class in the label vector `y` and dividing by the total number of documents. These probabilities are stored in `self.prior` in log-transformed form for numerical stability.
        
        2. **Calculate Word Frequencies for Each Class**:
        - The word counts for each class are calculated by summing the document-term matrix rows corresponding to each class. This gives the frequency of each word for each class. The total word count for each class is also computed.
        - The results are stored in `classwise_word_counts` and `total_words_in_class` dictionaries.
        
        3. **Calculate Conditional Probabilities (P(Word | Class))**:
        - For each word in the vocabulary, the conditional probability of that word given each class is computed using Laplace smoothing:
            P(Word | Class) = (count(Word, Class) + 1) / (total_words_in_class[Class] + len(vocab))
        - The result is stored in a 2D array `cond_prob`, where each row corresponds to a class, and each column corresponds to the probability of a word given that class.
        
        Args:
            X (ndarray): Document-term matrix (num_docs x num_words) representing the frequency of words in each document.
            y (ndarray): Array of class labels (length num_docs) corresponding to each document in `X`.
            
        Outputs:
            None: The model is trained, and the prior and conditional probabilities are stored internally in the model.
        """
        # Get unique classes
        self.classes = np.unique(y)

        # Calculate the prior probabilities P(Class)
        class_count = Counter(y)
        total_docs = len(y)
        self.class_counts = class_count

        for c in self.classes:
            self.prior[c] = np.log10(class_count[c] / total_docs)
        
        num_classes = len(self.prior)
            
        # Initialize dictionaries for word sums and total sums for each class
        classwise_word_counts = {}
        total_words_in_class = {}

        # Iterate through each class and calculate word sums and total sums
        for c in range(num_classes):
            # Get the document-term matrix rows where Y == c (class c)
            class_docs = X[y == c]
            
            # Sum the word counts for each word (sum along the vertical axis for class c)
            classwise_word_counts[c] = np.sum(class_docs, axis=0)
            
            # Total word count for class c (sum all words for the class)
            total_words_in_class[c] = np.sum(classwise_word_counts[c])
            
        # print(total_words_in_class, '\n class wise counts', classwise_word_counts) # Debug
        
        vocab_size = len(self.vocab)
        
        # Initialize a 2D array for conditional probabilities
        # Rows represent classes, columns represent words in the vocabulary
        #cond_prob = np.zeros((num_classes, vocab_size))

        # Calculate conditional probabilities for each class
        for c in range(num_classes):
            for word_idx in range(vocab_size):
                # Use the corresponding sum of word counts for the class c
                self.cond_prob[c][word_idx] = np.log10(
                    (classwise_word_counts[c][word_idx] + 1) / (total_words_in_class[c] + vocab_size)
                )
            
        
    def predict(self, doc_vector):
        """
        Predicts the probability distribution for all classes based on the given document vector (sentence_vector).
        
        Parameters:
        - doc_vector (np.array): A vector representation of the sentence, where each element corresponds to the frequency
                                    of a word in the document (non-binary bag-of-words).
        
        Returns:
        - prob_array (np.array): An array containing the probabilities for each class.
        """
        log_prob = {}  # Initialize log probabilities
        
        # Iterate over each class and calculate the log probabilities
        for c in self.classes:
            # Start with the log of the prior probabilities for each class
            log_prob[c] = self.prior[c]  # Use pre-calculated log of prior probabilities
            
            total_log_prob = 0
            # Add the log of the conditional probabilities P(Word | Class) for each word in the sentence
            for word_idx, word_count in enumerate(doc_vector):
                if word_count > 0:  # Only consider words that appear in the sentence
                    # Add the log-probability scaled by the word count (since it's non-binary bag-of-words)
                    log_prob[c] += word_count * self.cond_prob[c][word_idx]

        # To avoid overflow/underflow, use Log-Sum Exp trick
        # Find the maximum log probability to subtract from all probabilities
        max_log_prob = max(log_prob.values())
        
        # Exponentiate and calculate the regular probabilities
        self.regular_prob = {c:  np.power(10, (log_prob[c] - max_log_prob))  for c in log_prob}
        
        # Calculate total probability (sum of exponentiated probabilities)
        total_prob = sum(self.regular_prob.values())
        
        # Normalize the probabilities so they sum to 1
        if total_prob > 0:
            for c in self.regular_prob:
                self.regular_prob[c] /= total_prob  # Normalize to sum to 1
            #print("Line 312", max(self.regular_prob, key=self.regular_prob.get), self.regular_prob)
        else:
            # If total_prob is 0, perhaps return a default class
            print("Warning: Total probability is zero. Returning class with highest prior probability.")
            return self.prior
        
        return self.regular_prob
        
    def predict_class(self, doc_vector):
        """
        Predicts the class with the highest probability for a given sentence vector.

        Parameters:
        - sentence_vector (list or np.array): Vector representation of the sentence/document.
        - model (object): The Naive Bayes model instance that contains the `predict` method.

        Returns:
        - predicted_class (int): The class index with the highest probability.
        """
        # Step 1: Use the model's predict method to get the probabilities for each class
        self.predict(doc_vector)
        
        # Step 2: Use argmax to find the class with the highest probability
        predicted_class = max(self.regular_prob, key=self.regular_prob.get)
        
        return predicted_class

# Helper function to plot confusion matrics
def plot_confusion_matrices(y_test, y_pred_nb, y_pred_lr):
# Assuming y_test are true labels, and y_pred_nb, y_pred_lr are predictions from Naive Bayes and Logistic Regression
    # Generate confusion matrices
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    # Set up side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Confusion Matrix for Naive Bayes
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Naive Bayes Confusion Matrix')
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')

    # Confusion Matrix for Logistic Regression
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Logistic Regression Confusion Matrix')
    axes[1].set_xlabel('Predicted Labels')
    axes[1].set_ylabel('True Labels')

    # Adjust layout for readability
    plt.tight_layout()
    plt.show()

# Helper function to plot ROC curves
def plot_roc_curves(y_test, y_proba_nb, y_proba_lr):
    # For Naive Bayes (binary classification)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb, pos_label=1)  # No need to slice if it's a 1D array
    roc_auc_nb = auc(fpr_nb, tpr_nb)

    # For Logistic Regression (binary classification)
    # Extract the probabilities for class 1 (second column) from the 2D array
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr[:, 1], pos_label=1)  # Select probabilities for class 1
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    # Plot ROC curves for both models
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_nb, tpr_nb, color='blue', lw=2, label=f'Naive Bayes ROC curve (AUC = {roc_auc_nb:.2f})')
    plt.plot(fpr_lr, tpr_lr, color='green', lw=2, label=f'Logistic Regression ROC curve (AUC = {roc_auc_lr:.2f})')

    # Diagonal line for random classifier
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.show()

# Main execution
def main():
    algo, train_size = parse_args()
    print(f"Training set size: {train_size} %")
    print(f"Classifier type: {'Naïve Bayes' if algo == 0 else 'Logistic Regression'}")
    
    # Get and print the current time
    #current_time = datetime.now().time()
    #print("Current time:", current_time, end='\r', flush=True)          # Debug

    #print("Reading the file data: ", file_path , end='\r', flush=True)       # Debug
    
    # Load dataset
    df1 = load_dataset(file1_path)
    df2 = load_dataset(file2_path)    
    
    data = pd.concat([df1, df2], ignore_index=True)
    
    data = tokenizer(data)
    # Build vocab of unique tokens
    vocab = build_vocab(data)
    #print("Tokenizing into a NB Bag of words/ TF-IDF vectorization", end='', flush=True)        # Debug
    data = nb_bag_of_words_vectorization(data, vocab)
    #data = tf_idf_vectorization(data, vocab)
    
    print("Data length:", len(data))
    #print("Splitting the data vectorization:\n", end='\r', flush=True)      # Debug
    #sys.stdout.write('\r' + ' ' * 50)  # Clears the line by overwriting with spaces
    #sys.stdout.flush()

    # Split dataset
    train_data, test_data = train_test_split(data, train_size=train_size/100, shuffle=True, random_state=42)
    print("Train set length:", len(train_data), "\nTest set length:", len(test_data))
    
    # Plot to check the distribution of classes
    #train_data['Sentiment'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])

    #plt.title('Class Distribution')
    #plt.xlabel('Class')
    #plt.ylabel('Frequency')
    #plt.show()

    # Example: Assuming you have data X and y
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(np.array(train_data['vector'].tolist()))

    X_train = np.array(np.array(train_data['vector'].tolist()))
    Y_train = train_data['Sentiment']
    x_test = np.array(np.array(test_data['vector'].tolist()))
    y_test = test_data['Sentiment']
    
    if algo == 0:
        # Naive Bayes
        nb = NaiveBayes(vocab)
        print("\nTraining classifier…")
        nb.train(X_train, Y_train)
        
        # Evaluate and interact with user
        print("Testing classifier…")
        y_pred = [nb.predict_class(sentence) for sentence in x_test]
        
        #print(y_pred)
    elif algo == 1:
        # Logistic Regression
        lr = LogisticRegression(solver='saga', max_iter=1000)
        print("\nTraining classifier…")
        lr.fit(X_train, Y_train)
        # Evaluate and interact with user
        print("Testing classifier…")
        y_pred = lr.predict(x_test)
        
        #print(y_pred)      #Debug
    else:
        # Logistic Regression
        lr = LogisticRegression(solver='saga', max_iter=1000)
        nb = NaiveBayes(vocab)
        
        print("\nTraining LR classifier…")
        nb.train(X_train, Y_train)
        
        # Evaluate and interact with user
        print("Testing NB classifier…")
        y_pred_nb = [nb.predict_class(sentence) for sentence in x_test]
        y_proba_nb = [nb.predict(sentence) for sentence in x_test]
        
        print("\n")
        calculate_metrics(y_test, y_pred_nb)
        print("\n")
        
        print("Training LR classifier…")
        lr.fit(X_train, Y_train)
        # Evaluate and interact with user
        print("Testing LR classifier…")
        y_pred_lr = lr.predict(x_test)
        y_proba_lr = lr.predict_proba(x_test)
        
        y_proba_nb = np.array([probabilities[1] for probabilities in y_proba_nb])
        
        # ROC curves and AUC
        print(type(y_proba_nb), type(y_proba_lr))
        print(y_proba_nb[:5], y_proba_lr[:5])  # Check the first few entries


        plot_confusion_matrices(y_test, y_pred_nb, y_pred_lr)
        plot_roc_curves(y_test, y_proba_nb, y_proba_lr)
        
        # Plot Bar charts for data distribution
        # Calculate label breakdown
        train_label_counts = Y_train.value_counts()
        test_label_counts = y_test.value_counts()
        total_label_counts = data['Sentiment'].value_counts()

        # Combine label breakdowns into a DataFrame
        label_breakdown = pd.DataFrame({
            'Total Dataset': total_label_counts,
            'Training Set': train_label_counts,
            'Test Set': test_label_counts
        }).fillna(0)  # Fill missing values with 0

        # Plotting side-by-side bar charts
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot for total dataset
        label_breakdown['Total Dataset'].plot(kind='bar', ax=ax[0], color='skyblue', alpha=0.7)
        ax[0].set_title('Total Dataset Label Breakdown')
        ax[0].set_ylabel('Number of Samples')

        # Plot for training and test set together (stacked)
        label_breakdown[['Training Set', 'Test Set']].plot(kind='bar', ax=ax[1], color=['green', 'orange'], alpha=0.7)
        ax[1].set_title('Training and Test Set Label Breakdown')
        ax[1].set_ylabel('Number of Samples')

        plt.tight_layout()
        plt.show()
        
        print("\n")
        calculate_metrics(y_test, y_pred_lr)
        print("\n")
        
    print("\n")
    calculate_metrics(y_test, y_pred)
    print("\n")

    # Enter user input loop
    while True:
        sentence = input("Enter your sentence/document: ")
        temp_df = pd.DataFrame({'Text': [sentence]})
        temp_df = tokenizer(temp_df)
        doc_vector = nb_bag_of_words_vectorization(temp_df, vocab)
        temp_test = np.array(doc_vector['vector'])
                             
        if algo == 0:
            pred_class_prob = nb.predict(temp_test[0])
            predicted_class = max(pred_class_prob, key=pred_class_prob.get)
            class_name = 'Positive' if int(predicted_class) == 1 else 'Negative'
            print(f"Sentence/document S: {sentence}")
            print(f"was classified as {class_name}.")
            for label, prob in pred_class_prob.items():
                class_name = 'Positive' if int(label) == 1 else 'Negative'
                print(f"P({class_name} | S) = {prob:.4f}")
        else:
            temp_test = np.array(temp_df['vector'].values[0]).reshape(1, -1)
            predicted_class = lr.predict(temp_test)[0]
            class_name = 'Positive' if int(predicted_class) == 1 else 'Negative'
            print(f"Sentence/document S: {sentence}")
            print(f"was classified as {class_name}.")

        # Ask if the user wants to classify another sentence
        cont = input("\nDo you want to enter another sentence [Y/N]? ")
        if cont.lower() != 'y':
            break
if __name__ == '__main__':
    main()