import altair as alt
import streamlit as st


def set_polarity(df, positive=None, negative=None):
    _df = df.copy()
    _df['POLARITY'] = 'unk'
    if positive:
        _df['POLARITY'].loc[_df['WORD'].isin(positive)] = '+'
    if negative:
        _df['POLARITY'].loc[_df['WORD'].isin(negative)] = '-'
    return _df

def plot_chart(df):
    wv_chart = alt.Chart(df).mark_circle().encode(
        x='X',
        y='Y',
        color=alt.Color(
            'POLARITY', scale=alt.Scale(
                domain=['unk', '+', '-'],
                range=['#d6d6d6', '#1f77b4', '#ff7f0e']

            )
        ),
        tooltip=['WORD', 'POLARITY']
    ).interactive()

    st.write(wv_chart)