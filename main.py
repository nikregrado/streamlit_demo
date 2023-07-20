import streamlit as st
from backend_functions import vectordbqa

with st.form(key="qa_form"):
    query = st.text_area("Ask your questions to indexed document")
    submit = st.form_submit_button("Submit")

if submit:
    

    # Output Columns
    answer_col, sources_col = st.columns(2)

    result = vectordbqa(query)

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result['result'])

    with sources_col:
        st.markdown("#### Sources")
        for source in result['source_documents']:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
