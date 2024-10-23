import streamlit as st
import pickle
import base64

  

 

   




def set_bg_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def set_bg_from_url123(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: 100%; 
            background-repeat: no-repeat; 
            background-position: center; 
            background-attachment: fixed; 
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def home_page():
    st.markdown(
        """
        <style>
               button:not(:disabled) 
       {
         cursor: pointer;
         /* color: transparent; */
         background: transparent;
         /* border: none; */
         border-radius: 50px;
       }
         .css-1cypcdb
        {
        position: relative;
        top: 2px;
        /* background-color: rgb(38, 39, 48); */
        z-index: 999991;
        min-width: 244px;
        max-width: 550px;
        transform: none;
        transition: transform 300ms, min-width 300ms, max-width 300ms;
        background-image: url(https://i.pinimg.com/originals/2e/f6/eb/2ef6eb6eb1d6c234c50da4fd065cf3c8.gif);
        background-blend-mode: hard-light;
        background-position-x: center;

        }

        .css-1txtu6g {
          position: absolute;
          background: rgb(0, 0, 0);
          color: rgb(250, 250, 250);
          inset: 0px;
          overflow: hidden;
          background-blend-mode: hard-light;
    
}




        </style>""", unsafe_allow_html=True
    )




    gif_url = 'https://cdn.dribbble.com/users/6084/screenshots/1267647/gifforever.gif'
    set_bg_from_url(gif_url)
    set_bg_from_url("C:\\Users\\Siva G Nair\\PycharmProjects\\pythonProject\\Project\\project3\\assets\\images\\ZOHV.gif")

    navbar_height = '100px'
    st.markdown(
        f"""
        
        
        
          <div style='position: fixed; top: {navbar_height}; left: 0; width: 100%; height: calc(100% - {navbar_height}); display: flex; flex-direction: column; justify-content: center; align-items: flex-start; padding: 20px; z-index: 1;'>
              <h1 style='font-size: 72px; margin: 0; text-align: left; padding-top:420px;padding-left: 475px;color:cadetblue;'>Trash Classification for Recycling</h1>
              <p style='font-size: 24px; text-align: left; color: cadetblue;margin-top: 20px; max-width: 800px; padding: 0; padding-left: 482px; white-space: nowrap;'><b>Welcome to the TrashCam web app. Navigate through the tabs to explore the features.</b></p>
          </div>
          """,
        unsafe_allow_html=True
    )



import cv2
import numpy as np
from keras.models import load_model
from PIL import Image







import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time
# Load your trained model
model = load_model('assets/models/model.h5')


def preprocess_image(img):
    # Convert image to RGB and resize to 150x150 as expected by the model
    img = img.convert('RGB')
    img_array = np.array(img)
    img_resized = cv2.resize(img_array, (150, 150))  # Resize to (150, 150)
    
    # Normalize pixel values to [0, 1]
    img_resized = img_resized / 255.0
    
    # Add batch dimension: shape (1, 150, 150, 3)
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized


def predict_class(img, model, threshold=0.5):
    # Preprocess the image
    img_preprocessed = preprocess_image(img)
    
    # Make a prediction
    prediction = model.predict(img_preprocessed)
    
    # Interpret the result based on the threshold
    if prediction[0][0] > threshold:
        return "Recyclable", prediction[0][0]
    else:
        return "Organic", prediction[0][0]


def prediction_page():
    st.markdown(
        """
        <style>
               button:not(:disabled) 
       {
         cursor: pointer;
         /* color: transparent; */
         background: transparent;
         /* border: none; */
         border-radius: 50px;
       }
         .css-1cypcdb
        {
        position: relative;
        top: 2px;
        /* background-color: rgb(38, 39, 48); */
        z-index: 999991;
        min-width: 244px;
        max-width: 550px;
        transform: none;
        transition: transform 300ms, min-width 300ms, max-width 300ms;
        background-image: url(https://i.pinimg.com/originals/2e/f6/eb/2ef6eb6eb1d6c234c50da4fd065cf3c8.gif);
        background-blend-mode: hard-light;
        background-position-x: center;

        }

        .css-1txtu6g {
          position: absolute;
          background: rgb(0, 0, 0);
          color: rgb(250, 250, 250);
          inset: 0px;
          overflow: hidden;
          background-blend-mode: hard-light;
    
}




        </style>""", unsafe_allow_html=True
    )
    st.title("Waste Detection System")

    
    feature_options = ["♻️Upload", "♻️Real", "♻️Capture"]
    
   
    selected_feature = st.radio(
        "Select a feature:",
        options=feature_options,
        key="feature_selection"  
    )

   
    if selected_feature == "♻️Upload":
        upload_image_prediction(model)
    elif selected_feature == "♻️Real":
        real_time_detection(model)
    # elif selected_feature == "♻️Capture":
    #     capture_and_predict(model)
    else:
        st.write("Please select a feature.")






def upload_image_prediction(model):
    st.markdown("<h3 style='text-align: center;'>Upload an Image</h3>", unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image)
            st.image(img, caption='Uploaded Image.', use_column_width=True)

          
            label, confidence = predict_class(img, model)
            st.write(f"**Prediction: {label}** with **Confidence: {confidence:.2f}**")

            if label == "Organic":
                st.write("**Disposal Instructions:** Use the **Green Bin** for organic waste.")
            else:
                st.write("**Disposal Instructions:** Use the **Blue Bin** for recyclables.")

        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")



def real_time_detection(model):
    st.markdown('<h3 class="title-gap" style="text-align: center;">Real-time Detection from Webcam</h3>', unsafe_allow_html=True)

    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'tracking' not in st.session_state:
        st.session_state.tracking = False

    
    if start_button:
        try:
            if st.session_state.cap is None or not st.session_state.cap.isOpened():
                st.session_state.cap = cv2.VideoCapture(0)  
                st.session_state.tracking = True
                st.success("Webcam started.")
        except Exception as e:
            st.error(f"Failed to start webcam: {e}")

 
    if stop_button:
        try:
            if st.session_state.cap is not None and st.session_state.cap.isOpened():
                st.session_state.tracking = False
                st.session_state.cap.release() 
                st.session_state.cap = None
                cv2.destroyAllWindows()  
                st.success("Webcam stopped.")
        except Exception as e:
            st.error(f"Error while stopping the webcam: {e}")

   
    frame_placeholder = st.empty()

  
    if st.session_state.tracking and st.session_state.cap is not None and st.session_state.cap.isOpened():
        try:
            while st.session_state.tracking:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

               
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                roi = frame_rgb[100:400, 100:400] 
                
                try:
                    label, confidence = predict_class(Image.fromarray(roi), model)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    break

                
                cv2.rectangle(frame_rgb, (100, 100), (400, 400), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"{label} (Confidence: {confidence:.2f})", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

               
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

               
                if not st.session_state.tracking:
                    break

        except Exception as e:
            st.error(f"Error during webcam processing: {e}")
        finally:
            
            try:
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                cv2.destroyAllWindows()
            except Exception as e:
                st.error(f"Error while releasing resources: {e}")


# Feature 3: Capture Image from Webcam and Predict
# def capture_and_predict(model):
#     st.title("Capture Image from Webcam and Predict")

#     start_button = st.button("Start Webcam")
#     capture_button = st.button("Capture Image")
#     stop_button = st.button("Stop Webcam")

#     if 'cap' not in st.session_state:
#         st.session_state.cap = None
#     if 'tracking' not in st.session_state:
#         st.session_state.tracking = False

#     if start_button and not st.session_state.tracking:
#         st.session_state.cap = cv2.VideoCapture(0)
#         if st.session_state.cap.isOpened():
#             st.session_state.tracking = True
#             st.success("Webcam started.")
#         else:
#             st.error("Failed to open webcam.")

#     if stop_button and st.session_state.tracking:
#         st.session_state.tracking = False
#         if st.session_state.cap is not None and st.session_state.cap.isOpened():
#             st.session_state.cap.release()
#             st.session_state.cap = None
#             st.success("Webcam stopped.")

#     frame_placeholder = st.empty()

#     if st.session_state.tracking and st.session_state.cap is not None and st.session_state.cap.isOpened():
#         ret, frame = st.session_state.cap.read()
#         if not ret:
#             st.error("Failed to read from webcam.")
#             return

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

#         if capture_button:
#             captured_image = Image.fromarray(frame_rgb)
#             label, confidence = predict_class(captured_image, model)  # Pass the model
#             st.write(f"**Prediction for Captured Image: {label} (Confidence: {confidence:.2f})**")



  
    





def set_active_page():
    page = st.session_state.get("current_page", "Home")
    if page == "🏠Home":
        home_page()
    elif page == "🔎Prediction":
        prediction_page()
    elif page == "🗑️ Reporting":
        reporting_page()    
    elif page == "👨‍💻code":
        code_page()
    elif page == "📃Documentation":
        documentation_page()
    elif page == "♻️About":
        about_page()

import streamlit as st
from twilio.rest import Client
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim


# Twilio credentials
account_sid =${{ secrets.sid }}
auth_token = ${{ secrets.auth }}
client = Client(account_sid, auth_token)

# Function to send WhatsApp message with location link
def send_whatsapp_message(waste_type, lat, lon, additional_message):
    try:
        # Create a Google Maps link for the location
        location_link = f"https://www.google.com/maps?q={lat},{lon}"
        content_message = (
            f"Hey there, a {waste_type} has been placed at Latitude: {lat}, Longitude: {lon}.\n"
            f"Click to view the location: {location_link}\n\n"
            f"Additional message: {additional_message}"
        )
        
        # Send WhatsApp message with location link
        message = client.messages.create(
            from_='whatsapp:+14155238886',
            body=content_message,
            to='whatsapp:+916282684814'
        )
        return message.sid
    except Exception as e:
        st.error(f"Error sending WhatsApp message: {e}")
        return None

# Initialize reward counter in Streamlit session state
if 'reward_count' not in st.session_state:
    st.session_state['reward_count'] = 0

# Reporting page function
def reporting_page():
    st.markdown(
        """
        <style>
               button:not(:disabled) 
       {
         cursor: pointer;
         /* color: transparent; */
         background: transparent;
         /* border: none; */
         border-radius: 50px;
       }
         .css-1cypcdb
        {
        position: relative;
        top: 2px;
        /* background-color: rgb(38, 39, 48); */
        z-index: 999991;
        min-width: 244px;
        max-width: 550px;
        transform: none;
        transition: transform 300ms, min-width 300ms, max-width 300ms;
        background-image: url(https://i.pinimg.com/originals/2e/f6/eb/2ef6eb6eb1d6c234c50da4fd065cf3c8.gif);
        background-blend-mode: hard-light;
        background-position-x: center;

        }
        </style>""", unsafe_allow_html=True
    )

    


    st.title("Waste Type Selection and Location Reporting")

    # Display reward count with custom icon
    st.markdown(f"### 🌟 Reward Points: **{st.session_state['reward_count']}**")

    # Dropdown for waste type selection
    waste_type = st.selectbox("Choose Waste Type", ["Hen Waste", "Buffalo Waste"])

    # Input to search for a place (geolocation by name)
    place_name = st.text_input("Search for a place")

    # Initialize map centered on India
    lat, lon = 20.5937, 78.9629
    m = folium.Map(location=[lat, lon], zoom_start=5)

    # Geolocate based on place name (if provided)
    geolocator = Nominatim(user_agent="waste_locator")
    if place_name:
        location = geolocator.geocode(place_name)
        if location:
            lat, lon = location.latitude, location.longitude
            m = folium.Map(location=[lat, lon], zoom_start=12)
            st.success(f"Location found: {place_name} (Lat: {lat}, Lon: {lon})")
        else:
            st.error("Location not found. Try a different name.")

    # Add map marker based on clicked position
    map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])

    if map_data and map_data['last_clicked']:
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        st.write(f"Selected Location: Latitude: {lat}, Longitude: {lon}")

    # Additional message box for user input
    additional_message = st.text_area("Add any additional message or details")

    # Button to send message
    if st.button("Send WhatsApp Message"):
        message_sid = send_whatsapp_message(waste_type, lat, lon, additional_message)
        if message_sid:
            st.success(f"WhatsApp message sent successfully! Message SID: {message_sid}")
            st.balloons()
            st.write("🎉 Congratulations! You've successfully reported the waste.")
            # Increment reward counter
            st.session_state['reward_count'] += 5  # Add 5 points as a reward
        else:
            st.error("Failed to send the WhatsApp message.")
        


def page_navigation():
    
    pass


def main():
    page_navigation()


main()


def code_page():
    st.markdown(
        """
        <style>
               button:not(:disabled) 
       {
         cursor: pointer;
         /* color: transparent; */
         background: transparent;
         /* border: none; */
         border-radius: 50px;
       }
         .css-1cypcdb
        {
        position: relative;
        top: 2px;
        /* background-color: rgb(38, 39, 48); */
        z-index: 999991;
        min-width: 244px;
        max-width: 550px;
        transform: none;
        transition: transform 300ms, min-width 300ms, max-width 300ms;
        background-image: url(https://i.pinimg.com/originals/2e/f6/eb/2ef6eb6eb1d6c234c50da4fd065cf3c8.gif);
        background-blend-mode: hard-light;
        background-position-x: center;

        }
        </style>""", unsafe_allow_html=True
    )

    set_bg_from_url("https://wallpaperaccess.com/full/1595911.jpg")

    st.title("Colab Code ⭐")

    st.markdown(
        "<p> <br><br></p>",
        unsafe_allow_html=True)




    import nbformat
    from nbconvert import HTMLExporter




    notebook_path = 'C:\\Users\\Siva G Nair\\OneDrive\\Documents\\dl  projects\\Trash-Detection\\assets\\images\\1.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)
        html_exporter = HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(notebook_content)




    st.components.v1.html(body, height=800, scrolling=True)





def page_navigation():

    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠Home"

        sidebar_css = """
        <style>

       button:not(:disabled) 
       {
         cursor: pointer;
         /* color: transparent; */
         background: transparent;
         /* border: none; */
         border-radius: 50px;
       }
   
        
        .css-1cypcdb
        {
        position: relative;
        top: 2px;
        /* background-color: rgb(38, 39, 48); */
        z-index: 999991;
        min-width: 244px;
        max-width: 550px;
        transform: none;
        transition: transform 300ms, min-width 300ms, max-width 300ms;
        background-image: url(https://i.pinimg.com/originals/2e/f6/eb/2ef6eb6eb1d6c234c50da4fd065cf3c8.gif);
        background-blend-mode: hard-light;
        background-position-x: center;
    
        } 
        
        
        
        


          
            .sidebar h1, .sidebar h2, .sidebar h3, .sidebar p {
                color: #333; 
            }

          
            .sidebar-menu button {
                font-size: 18px;
                padding: 10px;
                width: 100%;
                text-align: left;
                background-color: transparent;
                color: #4B4B4B; 
                border: none;
                cursor: pointer;
                display: block;
                transition: background-color 0.3s, color 0.3s;
            }

            .sidebar-menu button:hover {
                background-color: #f1f1f1; 
                color: #007BFF; 
            }

            .sidebar-menu button.active {
                font-weight: bold;
                color: #007BFF; 
                background-color: #e0e0e0; 
                border-left: 4px solid #007BFF; 
            }
        </style>
        """


        st.markdown(sidebar_css, unsafe_allow_html=True)


        st.sidebar.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)

    if st.sidebar.button("🏠Home", key="🏠Home", help="Go to Home page"):
        st.session_state.current_page = "🏠Home"
    if st.sidebar.button("🔎Prediction", key="🔎Prediction", help="Go to Prediction page"):
        st.session_state.current_page = "🔎Prediction"
    if st.sidebar.button("🗑️ Reporting", key="🗑️ Reporting", help="Go to Prediction page"):
        st.session_state.current_page = "🗑️ Reporting"        
    if st.sidebar.button("👨‍💻code", key="👨‍💻code", help="Codings"):
        st.session_state.current_page = "👨‍💻code"
    if st.sidebar.button("📃Documentation", key="📃Documentation", help="Go to Documentation page"):
        st.session_state.current_page = "📃Documentation"
    if st.sidebar.button("♻️About", key="♻️About", help="Go to About page"):
        st.session_state.current_page = "♻️About"

    st.sidebar.markdown('</div>', unsafe_allow_html=True)


    set_active_page()


import pandas as pd

def documentation_page():
    st.markdown(
        """
        <style>
               button:not(:disabled) 
       {
         cursor: pointer;
         /* color: transparent; */
         background: transparent;
         /* border: none; */
         border-radius: 50px;
       }
         .css-1cypcdb
        {
         position: relative;
         top: 2px;
         /* background-color: rgb(38, 39, 48); */
         z-index: 999991;
         min-width: 244px;
         max-width: 550px;
         transform: none;
         transition: transform 300ms, min-width 300ms, max-width 300ms;
         background-image: url(https://i.pinimg.com/originals/2e/f6/eb/2ef6eb6eb1d6c234c50da4fd065cf3c8.gif);
         background-blend-mode: hard-light;
         background-position-x: center;

        }
        </style>""", unsafe_allow_html=True
    )


    set_bg_from_url123("https://c4.wallpaperflare.com/wallpaper/149/547/831/violet-gradient-abstract-wallpaper-preview.jpg")


    st.markdown(
        """
        <style>
        
        .ap {
    position: relative;
    left: 5px;
}
        
        .css-zybl48 {
    /* display: inline-flex; */
    -webkit-box-align: center;
    /* align-items: center; */
    -webkit-box-pack: center;
    /* justify-content: center; */
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: 19%;
    user-select: none;
    background-color: rgb(8, 8, 8);
    border: 1px solid rgba(250, 250, 250, 0.2);
    padding-left: 0px;
    margin-left: 50px;
    position: absolute;
    /* align-items: center; */
    left: 1px;
    /* position: absolute; */
}
   
        
        
        
        
        
        
        
        
        
            /* Create a translucent background rectangle with frosted glass effect */
            .translucent-background {
                position: absolute;
                top: 126px; 
                left: 0px; 
                width: 600px; 
                height: 400px; 
                background: rgba(255, 255, 255, 0.3); 
                backdrop-filter: blur(10px); 
                border-radius: 15px; 
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1); 
                z-index: 0; 
            }

            
            .main-content {
                position: relative;
                z-index: 1; 
            }

            
 

            .doc-title {
                font-size: 36px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 55px;
                font-family: 'Arial', sans-serif;
            }

            .doc-text {
                font-size: 18px;
                line-height: 1.6;
                margin: 10px 0;
                padding: 0 20px;
                text-align: left;
                font-family: 'Georgia', serif;
            }

            .doc-link {
                color: #1E90FF;
                text-decoration: underline;
                transition: color 0.3s;
            }

            .doc-link:hover {
                color: #FF4500;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown('<div class="translucent-background"></div>', unsafe_allow_html=True)


    st.markdown('<div class="main-content">', unsafe_allow_html=True)


    st.markdown("<h1 class='doc-title'>DOCUMENTATIONS</h1>", unsafe_allow_html=True)



    st.markdown("<p class='doc-text'>Here you can find the documentation and relevant links:</p>", unsafe_allow_html=True)


    st.markdown(
        "<p> <br>&nbsp &nbsp &nbsp <li>The dataset used for training model is given below.</li>.</p>",
        unsafe_allow_html=True)


    df = pd.read_csv('C:\\Users\\Siva G Nair\\PycharmProjects\\pythonProject\\Project\\project3\\assets\\models\\eng_dataset.csv')
    csv = df.to_csv(index=False)


    st.download_button(
        label="  Download",
        data=csv,
        file_name="downloaded_data.csv",
        mime="text/csv"
    )


    st.markdown(
        "<p class='doc-text'> <br><a class='doc-link' href='https://github.com/SivaG2002/Forest-Species-Prediction'><li>🤖</a> To get the Ui link which will redirect to my git.</li><br><br></p>",
        unsafe_allow_html=True)


    


def about_page():
    st.markdown(
        """
        <style>
               button:not(:disabled) 
       {
         cursor: pointer;
         /* color: transparent; */
         background: transparent;
         /* border: none; */
         border-radius: 50px;
       }
         .css-1cypcdb
        {
        position: relative;
        top: 2px;
        /* background-color: rgb(38, 39, 48); */
        z-index: 999991;
        min-width: 244px;
        max-width: 550px;
        transform: none;
        transition: transform 300ms, min-width 300ms, max-width 300ms;
        background-image: url(https://i.pinimg.com/originals/2e/f6/eb/2ef6eb6eb1d6c234c50da4fd065cf3c8.gif);
        background-blend-mode: hard-light;
        background-position-x: center;

        }
        </style>""", unsafe_allow_html=True
    )

    set_bg_from_url123("https://images.saymedia-content.com/.image/t_share/MTkzOTUzODU0MDkyODc5MzY1/particlesjs-examples.gif")


    st.markdown(
        """
        <style>
        
        

       
        .translucent-box {
    position: absolute;
    top: 1px;
    padding-left: 1px;
    width: 730px;
    height: 400px;
    background: rgba(255, 255, 255, 0.3);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    /* z-index: 0; */
    /* padding-left: 10px; */
    height: 1000px;
}
      
        .custom-header {
            padding-left:50px;
            font-size: 28px;  
            margin-bottom: 10px;  
            font-family: 'Arial', sans-serif; 
        }

        .custom-text {
            padding-left:50px;
            font-size: 18px;  
            line-height: 1.6;  
            font-family: 'Arial', sans-serif; 
        }

        .custom-list {
            font-size: 18px;  
            line-height: 1.6;  
            font-family: 'Arial', sans-serif; 
            padding-left: 20px; 
        }

        </style>
        """,
        unsafe_allow_html=True
    )


    st.title("About")

    
    st.markdown('<div class="translucent-box">', unsafe_allow_html=True)

    
    st.markdown('<div class="custom-header"><b><u>Technologies Used<u></b><br><br></div>', unsafe_allow_html=True)
    st.markdown(
    """
    <p class="custom-text">This project is built using Python and the Streamlit framework. It leverages machine learning models, specifically Convolutional Neural Networks (CNN), to classify trash into different categories such as plastic, paper, metal, and glass. The following libraries and techniques are used in this project:<br><br></p>

    <ul>
        <li><strong>Python:</strong> The main programming language used to develop and run the machine learning model.</li>
        <li><strong>Streamlit:</strong> An open-source framework used to build and deploy the web application, making it interactive and user-friendly.</li>
        <li><strong>Convolutional Neural Networks (CNNs):</strong> A deep learning model architecture used for image classification, identifying different types of trash from images.</li>
        <li><strong>TensorFlow/Keras (or PyTorch):</strong> Deep learning libraries used to build, train, and deploy the CNN model for accurate trash detection.</li>
        <li><strong>Image Preprocessing:</strong> Techniques such as resizing, normalization, and augmentation to prepare images for training and improve model performance.</li>
        <li><strong>OpenCV:</strong> A library used to handle real-time webcam input and capture images for processing by the CNN model.</li>
        <li><strong>Streamlit Interface:</strong> Provides an interactive UI for uploading images or using live webcam input to display real-time trash classification results.</li>
    </ul>
    """,
    unsafe_allow_html=True
)
    st.markdown(
        """
        <ul class="custom-list">
            <li>Scikit-learn for building and training machine learning models.</li>
            <li>Streamlit for building an interactive web interface.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )


    st.markdown('<div class="custom-header"><b>Features:</b></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="custom-text">Key features include:</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <ul class="custom-list">
            <li>User-friendly text input for emotion detection.</li>
            <li>Instant emotion predictions with a description of each emotion.</li>
            <li>Visual representation of emotion distribution in the provided text.</li>
            <li>Real-time feedback as the user inputs different texts.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )


    st.markdown('</div>', unsafe_allow_html=True)


def main():

    page_navigation()



main()



