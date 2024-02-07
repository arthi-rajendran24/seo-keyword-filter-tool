import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
#import zcatalyst_sdk
#from zcatalyst_sdk.catalyst_app import CatalystApp

# Streamlit app layout
st.title("Keyword Filter Tool")

# Upload file
uploaded_file = st.file_uploader("Choose a file with two columns: Column 1 with existing keywords & Column 2 with keywords from Ahrefs", type=['xlsx'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write(df)  # Display the uploaded DataFrame

    if st.button("Filter"):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        row1 = df.iloc[:, 0].dropna()

        # Encode both lists
        embedding_1 = model.encode(row1.tolist(), convert_to_tensor=True)
        embedding_2 = model.encode(df.iloc[:, 1].tolist(), convert_to_tensor=True)

        # Compute cosine similarities
        similarity = util.pytorch_cos_sim(embedding_1, embedding_2)

        # Find matches
        matches = []
        for i, entry1 in enumerate(row1):
            top_k_indices = torch.topk(similarity[i], k=3).indices.tolist()
            matching_values = [df.iloc[j, 1] for j in top_k_indices]
            matches.append({"row1": entry1, "matching_values": matching_values})

        df_matches = pd.DataFrame(matches)
        df_list = df_matches['matching_values'].to_list()

        def convert_list_of_lists_to_list(list_of_lists):
            return [element for sublist in list_of_lists for element in sublist]

        new_list = convert_list_of_lists_to_list(df_list)

        def remove_duplicates_using_for_loop(list1):
            list2 = []
            for element in list1:
                if element not in list2:
                    list2.append(element)
            return list2

        row2 = remove_duplicates_using_for_loop(new_list)

        def compare_and_remove_duplicates(list1, list2):
            list1_set = set(list1)
            return [element for element in list2 if element not in list1_set]

        unique_list2 = compare_and_remove_duplicates(row1.tolist(), row2)

        # Convert the list to a DataFrame
        selected_keywords = pd.DataFrame(unique_list2, columns=['Selected Keywords'])

        # Display the final DataFrame
        st.write(selected_keywords)

        # Download the final DataFrame
        @st.cache
        def convert_df_to_csv(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df_to_csv(selected_keywords)
        st.download_button(
            label="Download selected keywords as CSV",
            data=csv,
            file_name='selected_keywords.csv',
            mime='text/csv',
        )
