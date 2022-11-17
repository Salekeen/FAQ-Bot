# Importing dependencies
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import pandas as pd
import sys
sys.path.append("E:\\ML\\FAQ-Bot\\data")


def get_new_df_augmented(df):
    new_df = pd.DataFrame(columns=['Answer', 'Index'])
    texts = []
    indexes = []
    for index in range(len(df)):
        text_list = sent_tokenize(df['Answer'][index])
        for text in text_list:
            texts.append(text)
            indexes.append(index)

    new_df['Answer'] = pd.Series(texts)
    new_df['Index'] = pd.Series(indexes)
    return new_df


def get_embeddiings(train_df, test_df):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    answers = train_df['Answer']
    answer_embedding = model.encode(answers)

    targets = test_df['Question']
    target_embedding = model.encode(targets)

    return answer_embedding, target_embedding


def check_similarity(answer_embed,target_embed,df_aug,test_df,df):
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
    answer_embeddings, target_embeddings = get_embeddiings(df_aug, test_df)
    check_similarity(answer_embeddings, target_embeddings,df_aug,test_df,df)