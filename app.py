import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load your data here
df = pd.read_csv(r"C:\Users\hidayat\Desktop\Association_project\GroceryStoreDataSet.csv", names=['products'], sep=',')
data = list(df["products"].apply(lambda x: x.split(",")))

# Streamlit UI
st.title("Association Rule Mining App")

# Display the original data (optional)
st.write("Original Data:")
st.write(df)

# Data Preprocessing
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df_encoded = pd.DataFrame(a_data, columns=a.columns_).replace(False, 0)

# Calculate support values
min_support = st.slider("Select Minimum Support", 0.0, 1.0, 0.2, 0.05)
df_apriori = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Display the support values (optional)
st.write("Support Values:")
st.write(df_apriori)

# Association Rules
min_confidence = st.slider("Select Minimum Confidence", 0.0, 1.0, 0.6, 0.05)
df_association_rules = association_rules(df_apriori, metric="confidence", min_threshold=min_confidence)

# Display association rules
st.write("Association Rules:")
st.write(df_association_rules)

