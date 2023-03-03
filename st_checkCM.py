import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image

def plot_chart(df):
    """
    Parameters
    ----------
    df: pd.DataFrame
        The source DataFrame containing the data to be plotted
    Returns
    -------
    streamlit.altair_chart
        Bar chart with text above each bar denoting the conversion rate
    """
    chart = (
        alt.Chart(df)
        .mark_bar(color="#61b33b")
        .encode(
            x=alt.X("Type:O", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Number:Q", title="Number of images"),
            opacity="Type:O",
        )
        .properties(width=500, height=500)
    )

    # Place conversion rate as text above each bar
    chart_text = chart.mark_text(
        align="center", baseline="middle", dy=-10, color="black"
    ).encode(text=alt.Text("Number:Q", format=",.0g"))

    return st.altair_chart((chart + chart_text).interactive())

st.set_page_config(
    page_title="Checking label", page_icon="🧐", initial_sidebar_state="expanded"
)

st.title('Checking Confusion Matrix')
st.text("Author: Phan Vy Hao. \nWe're going to check the data that were wrongly predicted by the model.")

# upload the csv file
uploaded_file = st.file_uploader("Upload CSV of Confusion matrix", type=".csv")

# Button to identify that the Dataframe contains 'pred' column
compare_labels_button = st.checkbox(
    "Compare predicted and original labels?", False, help="If checked, will compare 2 labels."
)

# default values of dataframe
true_col_default = 0 #'true'
pred_col_default = 0 # 'pred'
path_col_default = 0 # 'path'
if compare_labels_button:
    true_col_default = 0 #'true'
    pred_col_default = 1 # 'pred'
    path_col_default = 3 # 'path'

# path of resulting variables
imgs_to_remove = ''
imgs_to_relabel = ''
imgs_to_keep_the_same = ''


if not uploaded_file:
    st.error("Please upload the file that contains image paths and its labels.")
else:
    df = pd.read_csv(uploaded_file)

    st.markdown("### Data preview")
    st.dataframe(df.head())

    with st.form(key="choose_columns"):
        true_col = st.selectbox(
            label = 'Column: Original class',
            options=df.columns,
            help="Select which column that is the original label of the image in raw dataset.",
            index=true_col_default,
        )
        path_col = st.selectbox(
            label = 'Column: Path to image',
            options=df.columns,
            help="Select which column that is the path to images.",
            index=path_col_default,
        )
        if true_col == path_col:
            st.error("Duplicated input.")
        if compare_labels_button:
            pred_col = st.selectbox(
                label = 'Column: predicted class',
                options=df.columns,
                help="Select which column that is the predicted label.",
                index=pred_col_default,
            )
            if pred_col == path_col or pred_col == true_col:
                st.error("Duplicated input.")
        

        # where to save images that will be removed?
        imgs_to_remove = st.text_input(
            label = "Path to save images that will be removed",
            value='/DataCentric/remove.csv',
            key = 'text_imgs_to_remove'
        )
        if len(imgs_to_remove) < 2:
            st.error("The path cannot be none. Please fill the path.")
        # where to save images that will be re-label?
        imgs_to_relabel = st.text_input(
            label = "Path to save images that will be relabel?",
            value='/DataCentric/relabel.csv',
            key = 'text_imgs_to_relabel'
        )
        if len(imgs_to_relabel) < 2:
            st.error("Path cannot be none. Please fill the path.")

        # where to save images that will be kept unchanged?
        imgs_to_keep_the_same = st.text_input(
            label = "The path to save images that its original label should be kept unchanged?",
            value='/DataCentric/keep.csv',
            key = 'text_imgs_to_keep_the_same'
        )
        if len(imgs_to_keep_the_same) < 2:
            st.error("The path cannot be none. Please fill the path.")
        
        # with st.expander("Adjust the Interface"):
        #     st.markdown("### Interface setting")
        #     st.radio(
        #         "Show processed images",
        #         options=["Preview", "No preview"],
        #         index=0,
        #         key="show_processed",
        #     )
            
        submit_button = st.form_submit_button(label="Submit")
    
    if not true_col or not path_col or len(imgs_to_remove) < 2 or len(imgs_to_relabel)<2 or len(imgs_to_keep_the_same)<2:
        st.warning("Please check columns and paths above.")
        st.stop()
    if compare_labels_button and not pred_col:
        st.warning("Please check columns and paths above.")
        st.stop()

    # st.balloons()

    name = uploaded_file.name

    st.write("")
    st.write("## Load images for re-checking label from ", name)
    st.write("")

    df_removed = pd.read_csv(imgs_to_remove)
    df_relabel = pd.read_csv(imgs_to_relabel)
    df_keep = pd.read_csv(imgs_to_keep_the_same)    
    number_of_processed_imgs = len(df_removed) + len(df_relabel) + len(df_keep)
    processed_img = list(df_removed['path']) + list(df_relabel['path']) + list(df_keep['path'])

    mcol1, mcol2 = st.columns(2)

    with mcol1:
        st.metric(
            "Uploaded:",
            value=f"{len(df)} images",
            # delta=f"{(st.session_state.crb - st.session_state.cra):.3g}%",
        )
    with mcol2:        
        st.metric(
            "Processed:",
            value=f"{number_of_processed_imgs} images",
            # delta=f"{(st.session_state.crb - st.session_state.cra):.3g}%",
        )
    
    if number_of_processed_imgs > 0:
        result_df = pd.DataFrame(
            {
                "Type": ["Remove", "Relabel", "Keep"],
                "Number": [len(df_removed), len(df_relabel), len(df_keep)],
            }
        )
        st.write("")
        # Plot bar chart of conversion rates
        plot_chart(result_df)
        st.write(result_df)

    # with st.form("form_show_image_label"):
        # output columns: label, path, type, checker
        # remove.csv
        # relabel.csv
        # keep.csv
    st.write("### Checking label")    

    label = ''
    path = ''
    check_type = ''
    checker = ''

    # lay so luong anh lam index hien tai, CHU Y NHA
    current_index = number_of_processed_imgs
    if number_of_processed_imgs > 0:
        current_index = number_of_processed_imgs - 1
    # kiem tra, neu trung thi +1
    while df.loc[current_index][path_col] in processed_img:
        current_index += 1               
    
    left, right = st.columns(2)

    rform = right.form("form_show_img")
    rform.write("Here's the IMAGE:")
    rform.image(Image.open(df.loc[current_index][path_col]), width=250)

    df_after_checking = pd.DataFrame(
        data = {
            'label': [label],
            'path': [path],
            'type': [check_type], 
            'checker': [checker]
        }
    )
    # result:
    if rform.form_submit_button('Load next image.'):                
        st.dataframe(df_after_checking.head())

    left.write("Checking the image with label:")
    form = left.form("form_check_label")
    checker = form.text_input("Checker name")    

    if compare_labels_button:
        form.write("### Predicted label:")
        form.write(str(df.loc[current_index][pred_col]))
        # if left.button(str(df.loc[current_index][pred_col])):
        #     label = str(df.loc[current_index][pred_col])
    form.write("### Original label:")
    form.write(str(df.loc[current_index][true_col]))
    # if left.button(str(df.loc[current_index][true_col])):
    #     label = str(df.loc[current_index][true_col])
    label = form.text_input("What's true label?")
    label = label.replace('1','i')
    label = label.replace('2', 'ii')
    label = label.replace('3', 'iii')
    label = label.replace('4', 'iv')
    label = label.replace('5', 'v')
    label = label.replace('6', 'vi')
    label = label.replace('7', 'vii')
    label = label.replace('8', 'viii')
    label = label.replace('9', 'ix')
    label = label.replace('10', 'x')
        
    path = df.loc[current_index][path_col]  
    check_type_categories = ['remove', "relabel", 'keep']      
    check_type = form.selectbox(
        "Checking type",
        options=check_type_categories,
        help="Select the best labels.",
        index=0,
    )
    if check_type == check_type_categories[-1]:
        label = str(df.loc[current_index][true_col])
        form.write("The original label will be kept:"+ label)
    # output columns: label, path, type, checker
    df_after_checking = pd.DataFrame(
        data = {
            'label': [label],
            'path': [path],
            'type': [check_type], 
            'checker': [checker]
        }
    )
    # result:
    # if rform.form_submit_button('Done.'):                
    #     st.dataframe(df_after_checking.head())

    # Every form must have a submit button. 
    check_submitted = form.form_submit_button("Save & Next")
    if check_submitted:
        if check_type == check_type_categories[0]:
            # remove
            df_result = pd.concat([df_removed, df_after_checking])
            df_result.to_csv(imgs_to_remove, index=False)
        elif check_type == check_type_categories[1]:
            # relabel
            df_result = pd.concat([df_relabel, df_after_checking])
            df_result.to_csv(imgs_to_relabel, index=False)
        elif check_type == check_type_categories[2]:
            # keep
            df_result = pd.concat([df_keep, df_after_checking])
            df_result.to_csv(imgs_to_keep_the_same, index=False)
        st.balloons()
        
