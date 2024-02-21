import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
from io import StringIO

def get_user_data() -> pd.DataFrame:
    user_data = {}

    user_data['Age'] = st.slider(
        label='Age:',
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

    user_data['Fare'] = st.slider(
        label='How much did your ticket cost you?:',
        min_value=0,
        max_value=300,
        value=80,
        step=1
    )

    user_data['SibSp'] = st.slider(
        label='Number of siblings and spouses aboard:',
        min_value=0,
        max_value=15,
        value=3,
        step=1
    )

    user_data['Parch'] = st.slider(
        label='Number of parents and children aboard:',
        min_value=0,
        max_value=15,
        value=3,
        step=1
    )

    col1, col2, col3 = st.columns(3)

    user_data['Pclass'] = col1.radio(
        label='Ticket class:',
        options=['1st', '2nd', '3rd'],
        horizontal=False
    )

    user_data['Sex'] = col2.radio(
        label='Sex:',
        options=['Man', 'Woman'],
        horizontal=False
    )

    user_data['Embarked'] = col3.radio(
        label='Port of Embarkation:',
        options=['Cherbourg', 'Queenstown', 'Southampton'],
        index=1
    )

    for k in user_data.keys():
        user_data[k] = [user_data[k]]
    df = pd.DataFrame(data=user_data)

    df['Sex'] = df['Sex'].map({'Man': 'male', 'Woman': 'female'})
    df['Pclass'] = df['Pclass'].map({'1st': 1, '2nd': 2, '3rd': 3})
    df['Embarked'] = df['Embarked'].map(
        {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}
    )
    df['Num_relatives'] = df['SibSp'] + df['Parch']

    return df

def load_data():
    url = "https://github.com/vipul097/mlopstt2/main/data/train.csv"
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text))
    print("Columns in DataFrame:")
    print(df.columns)
    # Preprocess data
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'].astype(str))
    return df

def main():
    this_file_path = os.path.abspath(__file__)
    project_path = '/'.join(this_file_path.split('/')[:-2])

    st.header(body='Lets check whether a passenger survived or not')

    df_user_data = get_user_data()

    df = load_data()

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Split data
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make prediction
    prediction = model.predict(df_user_data)  # Get the class prediction

    if prediction[0] == 0:  # 0 indicates did not survive
        st.title("Sorry, it seems you didn't survive.")
        st.error("Bad news my friend, you will be food for sharks! 🦈")
    else:  # 1 indicates survived
        st.title("Congratulations! It seems you survived!")
        st.success("Congratulations! You can rest assured, you will be fine! 🤩")

    # display an image of the Titanic
    st.image(project_path + '/images/RMS_Titanic.jpg')


if __name__ == '__main__':
    main()

