import nltk
import re
import pickle
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

#model loading
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

#cleaning 
def cleanResume(resume_txt):
    cleanTxt = re.sub('http\S+\s', '', resume_txt)
    cleanTxt = re.sub('RT|cc', ' ', cleanTxt)
    cleanTxt = re.sub('#\S+', '', cleanTxt)
    cleanTxt = re.sub('@\S+', '', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)
    cleanTxt = re.sub('â¢', ' ', cleanTxt)
    return cleanTxt.strip()

# Extract section, handles multi-page resumes by normalizing text
def extract_section(text, section_name):
    normalized_text = text.replace('\f', ' ').replace('\n', ' ')
    pattern = re.compile(
        rf'{section_name}\s*[:\-]?\s*(.*?)(?=\s+[A-Z][a-zA-Z\s]*[:\-]|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(normalized_text)
    if match:
        return match.group(1).strip()
    return None

def main():
    st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="centered")

    st.title('Resume Screening App')
    st.write("Upload your resume (txt or pdf). The app will predict the category and extract Skills & Experience sections if available.")

    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        X = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(X)[0]

        category_mapping = {
            15: 'Java Developer',
            23: 'Testing',
            8: 'Advocate', 
            20: 'Python Developer',
            24: 'Web Designing',
            12: 'HR',
            13: 'Hadoop',
            3: 'Blockchain',
            10: 'ETL Developer',
            18: 'Operations Manager',
            6: 'Data Science',
            22: 'Sales',
            16: 'Mechanical Engineer',
            1: 'Arts',
            7: 'Database',
            11: 'Electrical Engineering',
            14: 'Health and Fitness',
            19: 'PMO',
            4: 'Business Analyst',
            9: 'DotNet Developer',
            2: 'Automation Testing',
            17: 'SAP Developer',
            5: 'Civil Engineering'
        }
        category_name = category_mapping.get(prediction_id, 'Unknown')

        st.markdown("### Predicted Category")
        st.success(category_name)

        skills = extract_section(resume_text, 'Skills') or "Work in progress"
        experience = extract_section(resume_text, 'Experience') or "Work in progress"

        with st.expander("Skills Section"):
            if skills != "Not found":
                st.write(skills)
            else:
                st.info("No Skills section found in the resume.")

        with st.expander("Experience Section"):
            if experience != "Not found":
                st.write(experience)
            else:
                st.info("No Experience section found in the resume.")

if __name__ == '__main__':
    main()
