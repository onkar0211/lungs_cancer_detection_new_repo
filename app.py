
import pickle
import os
import sys
import logging
from pathlib import Path
import streamlit as st
import numpy as np
import pyparsing
import pandas as pd
import matplotlib as plt
import seaborn as sns
from PIL import Image 
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# Windows MAX_PATH: use a full TF install under %USERPROFILE%\tf when the venv copy is missing/incomplete
_tf_bundle = os.path.join(os.path.expanduser("~"), "tf")
if os.path.isfile(os.path.join(_tf_bundle, "tensorflow", "__init__.py")):
    sys.path.append(_tf_bundle)
import tensorflow
from tensorflow.keras.models import load_model
import tensorflow as tf
from io import BytesIO
from streamlit_option_menu import option_menu
st.set_page_config(page_title='Lung Cancer Detection')
# hello
LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
TABULAR_MODEL_PATH = BASE_DIR / "models" / "final_model.sav"


@st.cache_resource
def load_tabular_model():
    """Load and cache the tabular model once per app process."""
    if not TABULAR_MODEL_PATH.exists():
        st.error(f"Tabular model not found at `{TABULAR_MODEL_PATH}`.")
        LOGGER.error("Missing tabular model file: %s", TABULAR_MODEL_PATH)
        return None
    try:
        with TABULAR_MODEL_PATH.open("rb") as model_file:
            return pickle.load(model_file)
    except Exception as err:
        st.error("Failed to load tabular model.")
        st.code(str(err))
        LOGGER.exception("Tabular model load failed")
        return None


def parse_numeric_fields(fields):
    """Parse user input strings to ints and return (values, invalid_fields)."""
    parsed = []
    invalid = []
    for name, raw_value in fields:
        try:
            parsed.append(int(str(raw_value).strip()))
        except (TypeError, ValueError):
            invalid.append(name)
    return parsed, invalid


# Loading model once with defensive error handling
cancer_model = load_tabular_model()


with st.sidebar:
    selection = option_menu('Lung Cancer Detection System',
    ['Introduction',
    'About the Dataset @@@@@@',
    'Lung Cancer Prediction',
    'CNN Based disease Prediction'],
    icons = ['activity','heart','person', 'heart'],
    default_index = 0)
    

#Introduction page
if (selection == 'Introduction'):

    from PIL import Image

    gg = Image.open("images/lung-cancer.jpg")

    st.image(gg, caption='Introduction to Lung Cancer',width=600)
    #page title
    st.title('How common is lung cancer?')

    st.write("Lung cancer (both small cell and non-small cell) is the second most common cancer in both men and women in the United States (not counting skin cancer). In men, prostate cancer is more common, while in women breast cancer is more common.")
    st.markdown(
    """
    The American Cancer Society’s estimates for lung cancer in the US for 2023 are:
    - About 238,340 new cases of lung cancer (117,550 in men and 120,790 in women)
    - About 127,070 deaths from lung cancer (67,160 in men and 59,910 in women)

    """
    )

    st.write("")
    st.title("Is Smoking the only cause ?")
    mawen = Image.open("images/menwa.png")

    st.image(mawen, caption='Smoking is not the major cause',width=650)
    #page title
    
    st.write("The association between air pollution and lung cancer has been well established for decades. The International Agency for Research on Cancer (IARC), the specialised cancer agency of the World Health Organization, classified outdoor air pollution as carcinogenic to humans in 2013, citing an increased risk of lung cancer from greater exposure to particulate matter and air pollution.")




    st.markdown(
    """
    The following list won't indent no matter what I try:
    - A 2012 study by Mumbai’s Tata Memorial Hospital found that 52.1 per cent of lung cancer patients had no history of smoking. 
    - The study contrasted this with a Singapore study that put the number of non-smoking lung cancer patients at 32.5 per cent, and another in the US that found the number to be about 10 per cent.
    - The Tata Memorial study found that 88 per cent of female lung cancer patients were non-smokers, compared with 41.8 per cent of males. It concluded that in the case of non-smokers, environmental and genetic factors were implicated.
    """
    )

    st.title("Not just a Delhi phenomenon ")
    stove = Image.open("images/stove.png")

    st.image(stove, caption='Smoking is not the major cause',width=650)
    #page title
    st.markdown(
    """
    The following list won't indent no matter what I try:
    - In January 2017, researchers at AIIMS, Bhubaneswar, published a demographic profile of lung cancer in eastern India, which found that 48 per cent of patients had not been exposed to active or passive smoking
    - 89 per cent of women patients had never smoked, while the figure for men was 28 per cent.
    - From available research, very little is understood about lung cancer among non-smokers in India. “We need more robust data to identify how strong is the risk and link,” Guleria of AIIMS says.
    """
    )

if (selection == 'About the Dataset'):
    tab1, tab2, tab3 , tab4 ,tab5= st.tabs(["Dataset analysis", "Training Data", "Test Data","Algorithms Used",'CNN Based Indentification'])

    with tab1:
        
        st.header("Lung Cancer Dataset")
        data=pd.read_csv("datasets/data.csv")
        st.write(data.head(10))
        code = '''
        Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
       'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
       'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
       'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
       'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
       'Frequent Cold', 'Dry Cough', 'Snoring', 'Level'],
      dtype='object')'''
        st.code(code, language='python')

        
        st.header("Pearson Correlation Matrix")
        coors = Image.open("images/coors.png")

        st.image(coors, caption='Pearson Correlation Matrix',width=800)
        st.write("From the above co-relation matrix we did apply a function which picks out values based on their high correlation with a particular attribute which could be dropped to improve Machine Learning Models Performance")
        st.markdown( """
            
            - The Following Attributed are as follows :-
            """)
        code = '''{'Chest Pain',
 'Coughing of Blood',
 'Dust Allergy',
 'Genetic Risk',
 'OccuPational Hazards',
 'chronic Lung Disease'}'''
        st.code(code, language='python')

    with tab2:
        st.header("Lung Cancer Training Dataset")

        st.subheader("X_Train Data")
        data=pd.read_csv("datasets/train.csv", index_col=0)
        st.write(data)
        code = ''' Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
       'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
       'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
       'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')'''
        st.code(code, language='python')
        data=pd.read_csv("datasets/trainy.csv", index_col=0)
        st.subheader("Y_Train Data")
        st.dataframe(data, use_container_width=True)


       

    with tab3:
        st.header("Lung Cancer Training Dataset")

        st.subheader("X_Test Data")
        data=pd.read_csv("datasets/testx.csv", index_col=0)
        st.write(data)
        code = ''' Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
       'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
       'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
       'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')'''
        st.code(code, language='python')
        data=pd.read_csv("datasets/testy.csv", index_col=0)
        st.subheader("Y_Test Data")
        st.dataframe(data, use_container_width=True)
        
    with tab4:
        st.header("List of Algorithms Used")
        algo = Image.open("images/algo.png")

        st.image(algo, caption='ML Algorithms',width=500)

        st.write("Since this is a Mutlti-Class Classification we have used Algorithms which are maily used for Supervised Learning for the following Problem Statement ")

        st.markdown(
            """
            Supervised Learning Algorithms:
            - Linear Regression
            - Support Vector Machine
            - K-Nearest Neighbours (KNN)
            - Decision Tree Classifier
            """
            )
        
        st.write("The accuracy of all the above algorithms is as follows:- ")
        code = '''The accuracy of the SVM is: 95 %
        The accuracy of the SVM is: 100 %
        The accuracy of Decision Tree is: 100 %
        The accuracy of KNN is: 100 %'''
        st.code(code, language='python')

        st.header("Confusion Matrix")

        col1, col2 = st.columns(2)

        with col1:
            algo = Image.open("images/lg.png")

            st.image(algo, caption='LG Confusion Matrix',width=350)

        with col2:
            algo = Image.open("images/svm.png")

            st.image(algo, caption='SVM Confusion Matrix',width=390)


    with tab5:
        st.header("Convolutional Neural Network Model")
        st.write("Apart from detecting cancer using various parameters in the dataset we can also make out predictions using CT Scan Images by using Convolutional Neural Networks. Link to the image dataset is given below :- ")
        url = "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images"
        st.write("Check out this [Images Dataset](%s)" % url)

        st.subheader("Approach Followed :- ")
        st.markdown(
            """
            - For training our model we have used the Keras API.
            - We have used 2D Convolution Layer along with consecutive MaxPooling Layers to improve the models performance.
            - Because we are facing a two-class classification problem, i.e. a binary classification problem, we will end the network with a sigmoid activation. The output of the network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).
            """
            )
        st.subheader("Model Summary")
        summ = Image.open("images/summary.png")

        st.image(summ, caption='Model Summary',width=700)
        st.subheader("Model Compile ")
        st.write(" You will train our model with the binary_crossentropy loss, because it's a binary classification problem and your final activation is a sigmoid. We will use the rmsprop optimizer with a learning rate of 0.001. During training, you will want to monitor classification accuracy.")
        code = '''from tensorflow.keras.optimizers import RMSprop

        model.compile(optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics = ['accuracy'])'''
        st.code(code, language='python')

        st.subheader("Fitting Data to the Model")
        st.write(" You will train our model with the binary_crossentropy loss, because it's a binary classification problem and your final activation is a sigmoid. We will use the rmsprop optimizer with a learning rate of 0.001. During training, you will want to monitor classification accuracy.")
        code = '''model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        verbose=2
            )'''
        st.code(code, language='python')

        epoc = Image.open("images/epoc.png")

        st.image(epoc, caption='Number of Epocs',width=700)

        st.subheader("Plotting the Traning vs Validation (Accuracy and Loss)")
        col1, col2 = st.columns(2)

        with col1:
            acc = Image.open("images/acc.png")

            st.image(acc, caption='Traning vs Validation Accuracy',width=350)

        with col2:
            loss = Image.open("images/loss.png")

            st.image(loss, caption='Traning vs Validation Loss',width=350)

        st.write("As we can see from the above diagram that our Models performs well on the Training as well as Validation Data")


        
    

# Lung Cancer disease Prediction pages
if (selection == 'Lung Cancer Prediction'):
    #page title
    
    ##this code is reduce effort of manuallu entering values
    testx=pd.read_csv("datasets/testx.csv",index_col=0)
    testy=pd.read_csv("datasets/testy.csv",index_col=0)
    testx.reset_index(drop=True, inplace=True)
    testy.reset_index(drop=True, inplace=True)
    
    concate_data = pd.concat([testx,testy],axis=1)

    
    

    st.title('Lung Cancer Prediction using ML')

    idn = st.slider('Select any index from Testing Data', 0, 200, 25)
    a=concate_data.iloc[idn]
    st.write("Displaying vales of index ", idn)
    aa=list(concate_data.iloc[idn])
    if st.button('Show me this value'):
        st.write(aa)

    ##values will come directly from here no manual needed
    a=concate_data.iloc[idn][0]
    b=concate_data.iloc[idn][1]
    c=concate_data.iloc[idn][2]
    d=concate_data.iloc[idn][3]
    e=concate_data.iloc[idn][4]
    f=concate_data.iloc[idn][5]
    g=concate_data.iloc[idn][6]
    h=concate_data.iloc[idn][7]
    i=concate_data.iloc[idn][8]
    j=concate_data.iloc[idn][9]
    k=concate_data.iloc[idn][10]
    l=concate_data.iloc[idn][11]
    m=concate_data.iloc[idn][12]
    n=concate_data.iloc[idn][13]
    o=concate_data.iloc[idn][14]
    p=concate_data.iloc[idn][15]
    q=concate_data.iloc[idn][16]



    
    col1, col2, col3 = st.columns(3)
    with col1:
        
        Age = st.text_input('Age', key="1",value=a)
        
    with col2:
        Gender = st.text_input('Gender', key="2",value=b)
        
    with col3:
        AirPollution = st.text_input('Air Pollution', key="3",value=c)

    with col1:
        Alcoholuse = st.text_input('Alcohol Use', key="4",value=d)  

    with col2:
        BalancedDiet = st.text_input('Balanced Diet', key="5",value=e)
        
    with col3:
        Obesity = st.text_input('Obesity', key="6",value=f)
        
    with col1:
        Smoking = st.text_input('Smoking', key="7",value=g)
        
    with col2:
        PassiveSmoker = st.text_input('Passive Smoker', key="8",value=h)
        
    with col3:
        Fatigue = st.text_input('Fatigue', key="9",value=i)
        
    with col1:
        WeightLoss = st.text_input('Weight Loss', key="10",value=j)
        
    with col2:
        ShortnessofBreath = st.text_input('Shortness of Breath', key="11",value=k)
        
    with col3:
        Wheezing = st.text_input('Wheezing', key="12",value=l)
        
    with col1:
        SwallowingDifficulty = st.text_input('Swallowing Difficulty', key="13",value=m)
        
    with col2:
        ClubbingofFingerNails = st.text_input('Clubbing of Finger Nails', key="14",value=n)

    with col3:
        FrequentCold = st.text_input('Frequent Cold', key="15",value=o)
        
    with col1:
        DryCough = st.text_input('Dry Cough', key="16",value=p)    
     
    with col2:
        Snoring = st.text_input('Snoring  ', key="17",value=q)
 
    # code for Prediction
    lung_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Predict Lung Cancer Risk'):
        if cancer_model is None:
            st.error("Prediction model is unavailable. Please check deployment artifacts.")
            st.stop()

        feature_fields = [
            ("Age", Age),
            ("Gender", Gender),
            ("Air Pollution", AirPollution),
            ("Alcohol Use", Alcoholuse),
            ("Balanced Diet", BalancedDiet),
            ("Obesity", Obesity),
            ("Smoking", Smoking),
            ("Passive Smoker", PassiveSmoker),
            ("Fatigue", Fatigue),
            ("Weight Loss", WeightLoss),
            ("Shortness of Breath", ShortnessofBreath),
            ("Wheezing", Wheezing),
            ("Swallowing Difficulty", SwallowingDifficulty),
            ("Clubbing of Finger Nails", ClubbingofFingerNails),
            ("Frequent Cold", FrequentCold),
            ("Dry Cough", DryCough),
            ("Snoring", Snoring),
        ]
        parsed_features, invalid_fields = parse_numeric_fields(feature_fields)
        if invalid_fields:
            st.error(
                "Please enter valid numeric values for: "
                + ", ".join(invalid_fields)
            )
            st.stop()

        try:
            prediction = cancer_model.predict([parsed_features])
        except Exception as err:
            st.error("Prediction failed. Please try again.")
            st.code(str(err))
            LOGGER.exception("Tabular model prediction failed")
            st.stop()

        if prediction[0] == 'High':
            lung_diagnosis = 'High risk detected. Please consult a medical professional.'
            st.error(lung_diagnosis)
        elif prediction[0] == 'Medium':
            lung_diagnosis = 'Moderate risk detected. Clinical review is recommended.'
            st.warning(lung_diagnosis)
        else:
            lung_diagnosis = 'Low risk prediction based on provided inputs.'
            st.balloons()
            st.success(lung_diagnosis)
        
        

    expander = st.expander("Here are some more random values from Test Set")
    
    expander.write(concate_data.head(5))
    

    
        
   

if (selection == 'CNN Based disease Prediction'):
  @st.cache_resource
  def loading_model():
    # Prefer the repository CNN model that is compatible with current Keras.
    fp = "cnn model/lungcancer_model_cnn.h5"
    try:
      model_loader = load_model(fp, compile=False)
      return model_loader
    except Exception as err:
      st.error("CNN model could not be loaded in this TensorFlow/Keras version.")
      st.info("Model loading failed. Check TensorFlow/Keras installation and model file integrity.")
      st.code(str(err))
      return None

  cnn = loading_model()
  st.write("""
  # Lung Cancer Detection using CNN and CT-Scan Images
  """)



  temp = st.file_uploader("Upload CT-Scan Image", type=["png", "jpeg", "jpg"])
  if temp is not None:
      file_details = {"FileName": temp.name, "FileType": temp.type, "FileSize": temp.size}
      st.write(file_details)

  buffer = temp

  if cnn is None:
    st.stop()

  if buffer is None:
    st.text("Oops! that doesn't look like an image. Try again.")

  else:
    data = buffer.getvalue()
    if not data:
      st.error("The file is empty. Please choose a valid PNG or JPEG.")
      st.stop()

    try:
      pil_img = Image.open(BytesIO(data))
      pil_img.load()
    except Image.UnidentifiedImageError:
      st.error(
        "This file is not a valid image (or is corrupted). "
        "Use a real PNG or JPEG from a CT viewer or camera export."
      )
      st.stop()

    pil_img = pil_img.convert("RGB")
    # Match the loaded model input shape dynamically.
    _, h, w, _ = cnn.input_shape
    try:
      resample = Image.Resampling.LANCZOS
    except AttributeError:
      resample = Image.LANCZOS
    pil_img = pil_img.resize((w, h), resample)

    pp_ved_img = np.asarray(pil_img, dtype=np.float32) / 255.0
    pp_ved_img = np.expand_dims(pp_ved_img, axis=0)

    hardik_preds = cnn.predict(pp_ved_img, verbose=0)

    if hardik_preds[0][0] >= 0.5:
      out = "I am {:.2%} percent confirmed that this is a Normal Case".format(hardik_preds[0][0])
      st.balloons()
      st.success(out)
    else:
      out = "I am {:.2%} percent confirmed that this is a Lung Cancer Case".format(
        1 - hardik_preds[0][0]
      )
      st.error(out)

    st.image(pil_img, width="stretch")
            
              

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)  

    
