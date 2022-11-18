# Importing dependencies
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import pandas as pd
import sys
sys.path.append("E:\\ML\\FAQ-Bot\\data")
from utils import clean


def get_new_df_augmented(df):
    """
    It takes a dataframe with a column called 'Question' and returns a new dataframe with a column
    called 'Answer' and 'Index'
    
    The 'Answer' column is a list of sentences from the 'Question' column
    
    Args:
      df: the dataframe that you want to augment
    
    Returns:
      A dataframe with the questions split into sentences and the index of the original question.
    """
    
    new_df = pd.DataFrame(columns=['Answer', 'Index'])
    texts = []
    indexes = []
    for index in range(len(df)):
        text_list = sent_tokenize(df['Question'][index])
        for text in text_list:
            texts.append(text)
            indexes.append(index)

    new_df['Answer'] = pd.Series(texts)
    new_df['Index'] = pd.Series(indexes)
    return new_df


def replace_synonymns(df_aug,test_df):
    """
    It takes in the augmented dataframe and the test dataframe, and replaces the synonyms in the
    augmented dataframe's answers and the test dataframe's questions
    
    Args:
      df_aug: the augmented dataframe
      test_df: The test dataframe
    
    Returns:
      the augmented dataframe and the test dataframe.
    """
    
    df_aug['Answer'] = df_aug['Answer'].map(clean.get_synonyms)
    test_df['Question'] = test_df['Question'].map(clean.get_synonyms)
    return df_aug,test_df


def get_embeddiings(train_df, test_df):
    """
    It takes the answers and questions from the training and test dataframes, and returns the embeddings
    for each of them
    
    Args:
      train_df: the training dataframe
      test_df: the dataframe containing the questions
    
    Returns:
      The embeddings of the answers and questions
    """
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    answers = train_df['Answer']
    answer_embedding = model.encode(answers)

    targets = test_df['Question']
    target_embedding = model.encode(targets)

    return answer_embedding, target_embedding


def check_similarity(answer_embed,target_embed,df_aug,test_df,df):
    """
    It takes the embeddings of the target question and the answer embeddings and finds the most similar
    answer embedding to the target embedding. 
    
    The index of the most similar answer embedding is then used to find the answer in the original
    dataframe
    
    Args:
      answer_embed: The embeddings of the answers in the training set
      target_embed: The embeddings of the questions in the test set
      df_aug: The augmented dataframe
      test_df: The dataframe containing the questions to be answered
      df: The original dataframe
    """
    
    for i in range(len(target_embed)):
        index = cosine_similarity(
            [target_embed[i]],
            answer_embed[:]
        ).argmax()

        df_aug['Index'][index]

        print(test_df['Question'][i])
        print(df['Answer'][df_aug['Index'][index]])
        print("\n")



if __name__ == "__main__":
    df = pd.read_csv("./data/FAQs.csv")
    test_df = pd.read_csv("./data/FAQs_test.csv")

    df_aug = get_new_df_augmented(df)
    df_aug,test_df = replace_synonymns(df_aug,test_df)
    answer_embeddings, target_embeddings = get_embeddiings(df_aug, test_df)
    check_similarity(answer_embeddings, target_embeddings,df_aug,test_df,df)