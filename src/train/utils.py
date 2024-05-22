from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display_functions import display
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.colors as mcolors
import numpy as np
from bokeh.models import Label, LegendItem, Legend

# Define helper functions
def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)

def plot_bar_byldaout(lda_output):
    lda_keys = get_keys(lda_output)
    lda_categories, lda_counts = keys_to_counts(lda_keys)

    topics_df_lda = pd.DataFrame({'topic' : lda_categories, 'count' : lda_counts})
    sns.barplot(x=topics_df_lda['topic'], y = topics_df_lda['count'])
    plt.show()
    
def plot_tsnescatter_byldaout(lda_output):
    # Array of topic weights    
    arr = pd.DataFrame(lda_output).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)
    n_topics = len(np.unique(topic_num))

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_notebook()
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
                  width=900, height=700)

    # Create a legend
    legend_items = []
    renderers = []
    for topic in range(n_topics):
        # Filter the data points for the current topic
        topic_points = tsne_lda[topic_num == topic]

        # Create a scatter renderer for the current topic
        renderer = plot.scatter(x=topic_points[:,0], y=topic_points[:,1], color=mycolors[topic], legend_label=str(topic))
        renderers.append(renderer)

        # Add a legend item for the current topic
        legend_items.append(LegendItem(label=str(topic), renderers=[renderer]))

    legend = Legend(items=legend_items, location='top_right')
    plot.add_layout(legend, 'right')
    show(plot)
    
def displ_topnw_byldamodel(lda_model, keywords, n_words=15, lambda_val=0.5):
    # Function to compute relevance scores
    def compute_relevance(log_t_w, log_w, lambda_):
        return lambda_ * log_t_w + (1 - lambda_) * (log_t_w - log_w)

    topic_keywords = []
    
    # Compute term-topic distribution and log probabilities
    term_topic_dist = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
    log_t_w = np.log(term_topic_dist)
    log_w = np.log(lda_model.components_.sum(axis=0))
    for topic_idx in range(0, lda_model.n_components):
        term_scores = compute_relevance(log_t_w[topic_idx], log_w, lambda_val)
        top_keyword_locs = (-term_scores).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))

    # Create a DataFrame from the LDA model's top keywords
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = [f'Word {i}' for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = [f'Topic {i}' for i in range(df_topic_keywords.shape[0])]

    # Display the DataFrame
    display(df_topic_keywords)
    
def get_most_reprdocs(df, doc_topic_dist, orig_data_path, lda_model, n_docs=1):
# Function to get the most representative documents for each topic and return a DataFrame

    # Create a dictionary to hold the representative documents
    most_representative_docs = {}
        # Calculate the topic prevalence
    n_topics = lda_model.n_components
    
    for i, topic_idx in enumerate(range(n_topics)):
        # Get the indices of the documents with the highest probability for the current topic
        top_doc_indices = doc_topic_dist[:, topic_idx].argsort()[-n_docs:][::-1]
        # Store the documents in the dictionary
        most_representative_docs[f'Topic {i}'] = df.iloc[top_doc_indices].values.tolist()
    
    # Find the maximum number of documents across all topics
    max_n_docs = max(len(docs) for docs in most_representative_docs.values())
    
    # Ensure each topic column has the same number of documents by padding with None
    most_representative_docs_padded = {
        topic: docs + [None] * (max_n_docs - len(docs))
        for topic, docs in most_representative_docs.items()
    }
    
    # Convert the dictionary to a DataFrame
    df_most_representative_docs = pd.DataFrame(most_representative_docs_padded)
    
    # Rename the index to reflect document numbers
    df_most_representative_docs.index = [f'Doc {i+1}' for i in range(df_most_representative_docs.shape[0])]
    

    df_orig = pd.read_csv(orig_data_path)
    for i in range(0, n_topics):
        df_most_representative_docs[f'Topic {i}'] = df_orig[df_orig['id'].isin(df_most_representative_docs[f'Topic {i}'])]['text_preproc1'].values
        
    return df_most_representative_docs

