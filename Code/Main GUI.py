# Need Add Size for Each Class into Excel

# Libraries
import ast
import cv2
import sys
import os
import glob2
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QWidget, QTabWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Own Libraries
import general_calculator as gcal
import feature_calculator as fcal

# Initialize GUI
class GUI(QMainWindow):
    # Take stuff from QMainWindow
    def __init__(self):
        super().__init__()
        self.initUI()
    
    # Setting basic UI
    def initUI(self):
        # Setting the global font and size
        font = QFont("Arial", 9)
        app.setFont(font)

        self.title = 'Object Detection GUI'
        self.left = 550
        self.top = 300
        self.width = 850
        self.height = 600
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.tab_widget = ObjectDetection(self)
        self.setCentralWidget(self.tab_widget)

# Object Detection Main Program
class ObjectDetection(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        # Setting the global font and size again
        font = QFont("Arial", 9)
        app.setFont(font)

        # Initialize Settings
        self.settings_update()
        self.save_delete()

        # Initialize Tab Screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.resize(800, 500)

        # Add Tabs
        self.tabs.addTab(self.tab1, "Cognition")
        self.tabs.addTab(self.tab2, "Localization")
        self.tabs.addTab(self.tab3, "Live Feed")

        # Cognition Tab
        # Making Variables
        self.class_features = {}
        self.saved = False

        # Load Training Set Image button
        self.load_train_btn = QPushButton('Add a training set', self.tab1) # Make a push button with that name
        self.load_train_btn.resize(180, 30)
        self.load_train_btn.move(25, 50)
        self.load_train_btn.clicked.connect(self.cog_load_training_set) # Connect button to load_training_set fn

        # Add a new button to save data to Excel
        self.save_excel_btn = QPushButton('Save to Excel', self.tab1)
        self.save_excel_btn.resize(180, 30)
        self.save_excel_btn.move(25, 90)
        self.save_excel_btn.clicked.connect(self.cog_save_to_excel)

        # Display Results
        self.result_label1 = QTextEdit(self.tab1)
        self.result_label1.move(240, 50)
        self.result_label1.setFixedSize(550, 350)
        self.result_label1.setAlignment(Qt.AlignTop)
        self.result_label1.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 2px solid #1e90ff;
                border-radius: 10px;
                color: #000000;
                padding: 5px;
            }
        """)



        # Localization
        # Load Cognitive Results Button
        self.load_cognition_results_btn = QPushButton('Change Cognition File', self.tab2)
        self.load_cognition_results_btn.resize(180, 30)
        self.load_cognition_results_btn.move(25, 50)
        self.load_cognition_results_btn.clicked.connect(self.recog_change_cognition_results)

        # Select New Image button
        self.select_new_images_btn = QPushButton('Select new images', self.tab2)
        self.select_new_images_btn.resize(180, 30)
        self.select_new_images_btn.move(25, 90)
        self.select_new_images_btn.clicked.connect(self.recog_select_new_images)

        # Results Display Label
        self.result_label2 = QTextEdit(self.tab2)
        self.result_label2.move(240, 50)
        self.result_label2.setFixedSize(550, 350)
        self.result_label2.setAlignment(Qt.AlignTop)
        self.result_label2.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 2px solid #1e90ff;
                border-radius: 10px;
                color: #000000;
                padding: 5px;
            }
        """)

        # Add Tabs to Widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
    
    def settings_update(self):
        # sk_list = self.settings['sk_list'][0] is example to access setting
        self.settings_addr = "Settings.xlsx" # Important
        settings =  pd.read_excel(self.settings_addr, header = None) 
        # Converting 1st column to header
        settings = settings.transpose(copy = False) 
        settings.columns = settings.iloc[0]
        settings = settings[1:]
        settings = settings.reset_index(drop = True)
        # Convert strings that are list to list
        for setting, value in settings.items():
            try:
                temp = ast.literal_eval(value[0])
            except (ValueError, SyntaxError):
                temp = value[0]
                if str(temp).lower() == "true":
                    temp = True
                if str(temp).lower() == "false":
                    temp = False
            settings[setting][0] = temp
        self.settings = settings
        # print(self.settings['resize']) # Testing

    def save_delete(self):
        files_in_folder = glob2.glob(self.settings['save_addr'][0] + "/*") # Look at all files inside selected folder
        #print(files_in_folder)
        for file in files_in_folder:
            list_of_items = glob2.glob(file + "/*")
            for item in list_of_items:
                # print(item + "\n")
                os.remove(item)

    def cog_load_training_set(self):
        folder_address = QFileDialog.getExistingDirectory(self, 'Add a training set') # Get user to select folder
        if folder_address:
            # Resetting if user saved to excel but wants to save more dataset
            if self.saved == True:
                current_displayed_text = ""
                self.class_features = {}
                self.saved = False
            else:
                current_displayed_text = self.result_label1.toHtml() # Fetch current displayed text from GUI
            
            # Empty list to store new text to update
            new_data_list = []

            # Extract class features
            files_in_folder = glob2.glob(folder_address + "/*") # Look at all files inside selected folder
            folder_check = False # Initially assume selected folder only contains training images
            folders = {} # Dictionary to store only folders
            class_features = {} # Library to store class features of each image

            # Finds all folders and extracts into 'Folder Name': 'Absolute Address of Folder'
            for address in files_in_folder:
                name = os.path.basename(address) # Finds names of files
                split_name = name.split('.') # Splits names using .
                if len(split_name) == 1: # Only folders have length of 1 since no extension
                    folders[name] = address.replace("\\", "/")
                    folder_check = True
            # print(folders)
            
            # If the folder selected is the folder containing training image itself
            if folder_check == False:
                name = os.path.basename(folder_address) # Finds names of files
                folders[name] = folder_address.replace("\\", "/")
            
            # Calculate Feature Vectors for Each Class
            for class_name, folder_address in folders.items():
                # print("Name:", class_name, "Folder Address:", folder_address)
                
                # Extract Images Addresses
                image_files = [] # List containing address of images
                image_files = glob2.glob(folder_address + "/*.jpg")
                image_files.extend(glob2.glob(folder_address + "/*.jpeg"))
                image_files.extend(glob2.glob(folder_address + "/*.png"))
                image_files.extend(glob2.glob(folder_address + "/*.jfif"))
                # print(image_files)
                
                if self.settings['resize'][0] == True and self.settings['size_auto'][0] == True:
                     # Find out average height and width of training images in each class
                    raw_image_list, height_list, width_list = [], [], []
                    for image_file in image_files:
                        image = cv2.imread(image_file) # Gets a pixel list like with np.array
                        height, width, _ = image.shape # Get height and width
                        raw_image_list.append(image)
                        height_list.append(height)
                        width_list.append(width)
                    
                    height_avg = sum(height_list) / len(image_files)
                    width_avg = sum(width_list) / len(image_files)

                    # Make dimensions to resize to be half of the average
                    dimensions = [int(height_avg/2), int(width_avg/2)] # Testing
                elif self.settings['resize'][0] == True and self.settings['size_auto'][0] == False:
                    dimensions = self.settings['size'][0]
                else:
                    dimensions = [-1,-1]

                # Extract and resize images
                training_images = gcal.resizer(self.settings['save'][0], raw_image_list, dimensions, class_name)

                # Extract feature vectors
                time_feature_vectors_list, freq_feature_vectors_list = fcal.calculate_feature_vectors(training_images) # Gets feature vectors and address of each image
                time_mean_feature_vector, freq_mean_feature_vector, time_variance, freq_variance = fcal.get_mean_and_variance(time_feature_vectors_list, freq_feature_vectors_list) # Gets array of mean of feature vectors and one value of variance
                mean_vector = np.append(time_mean_feature_vector, freq_mean_feature_vector)
                variance_val = np.append(time_variance, freq_variance)
                # print(mean_vector, '\n', variance_val)
                class_features[class_name] = {"dimensions": dimensions, "mean": mean_vector, "variance": variance_val}

            # Display feature vectors to GUI
            for class_name, vectors in class_features.items():                
                # Format the new data
                header = f"<font size='5' color='red'><b>{class_name}:</b></font><br>"
                mean_str = ', '.join([f"{val:.4f}" for val in vectors['mean'].flatten()])
                new_data_strings = [
                header,
                f"Overall Mean of Features: [{mean_str}]<br>",
                f"Time Variance: {vectors['variance'][0]:.4f}, Freq Variance: {vectors['variance'][1]:.4f}<br>"
                "<hr>"  # Add a line for visual separation
                ]
                new_data_list.append("<br>".join(new_data_strings))
            
            self.class_features.update(class_features) # Update Dictionary

            # Add new data to current displayed text and set the text
            self.result_label1.setText(current_displayed_text + "<br>".join(new_data_list))

    def cog_save_to_excel(self):
        # "Save to Excel" is title of popup, "" is starting directory, "Excel Files (*.xlsx)" is possible extensions to save as
        save_path, _ = QFileDialog.getSaveFileName(self, "Save to Excel", "", "Excel Files (*.xlsx)")
        if save_path:
            if not save_path.endswith(".xlsx"):
                save_path += ".xlsx" # Force add .xlsx
            
            # Check if file exist
            try:
                # Copy Existing Content in Excel to Update
                existing_df = pd.read_excel(save_path)
            except FileNotFoundError:
                existing_df = pd.DataFrame()
            
            # Uses file name selected by user to save file
            data_to_save = [] # Set up list
            for classname, data in self.class_features.items(): # {"ClassName": {"mean": ..., "variance": ...}, ...}, classname direct access, data need name of library as key
                row_data = {"Class Name": classname, "Dimensions": data['dimensions'], "Time Variance": data['variance'][0], "Freq Variance": data['variance'][1]} # Store as new data
                
                # (r_time_features, g_time_features, b_time_features, r_freq_features, g_freq_features, b_freq_features)
                # [mean, std, dispersion_x, dispersion_y]
                # Header in Excel
                vector_names = ["Red 0th Time Mean", "Red 0th Time Std", "Red 0th Time Distribution (H)", "Red 0th Time Distribution (V)",
                                "Red 1st Time Mean", "Red 1st Time Std", "Red 1st Time Distribution (H)", "Red 1st Time Distribution (V)",
                                "Green 0th Time Mean", "Green 0th Time Std", "Green 0th Time Distribution (H)", "Green 0th Time Distribution (V)",
                                "Green 1st Time Mean", "Green 1st Time Std", "Green 1st Time Distribution (H)", "Green 1st Time Distribution (V)",
                                "Blue 0th Time Mean", "Blue 0th Time Std", "Blue 0th Time Distribution (H)", "Blue 0th Time Distribution (V)",
                                "Blue 1st Time Mean", "Blue 1st Time Std", "Blue 1st Time Distribution (H)", "Blue 1st Time Distribution (V)",
                                "Red Real Freq Mean", "Red Real Freq Std", "Red Real Freq Distribution (H)", "Red Real Freq Distribution (V)",
                                "Red Img Freq Mean", "Red Img Freq Std", "Red Img Freq Distribution (H)", "Red Img Freq Distribution (V)",
                                "Green Real Freq Mean", "Green Real Freq Std", "Green Real Freq Distribution (H)", "Green Real Freq Distribution (V)",
                                "Green Img Freq Mean", "Green Img Freq Std", "Green Img Freq Distribution (H)", "Green Img Freq Distribution (V)",
                                "Blue Real Freq Mean", "Blue Real Freq Std", "Blue Real Freq Distribution (H)", "Blue Real Freq Distribution (V)",
                                "Blue Img Freq Mean", "Blue Img Freq Std", "Blue Img Freq Distribution (H)", "Blue Img Freq Distribution (V)"]
                for i, val in enumerate(data['mean']): # Enumerate adds a counter for each object, eg. [(0, 'Python'), (1, 'Java'), (2, 'JavaScript')]
                    row_data[vector_names[i]] = val # Make new header with name to store val
                data_to_save.append(row_data) # Add data to save into list
            df = pd.DataFrame(data_to_save) # Convert to dataframe, each header is a column header and each item is below it
            updated_df = pd.concat([existing_df, df], ignore_index = True, sort = False)
        
        # Save data to Excel, saving is done here, below is to open book to adjust col size, still have issue
        save_path = save_path.replace("/", "\\")
        updated_df.to_excel(save_path, index=False)
        self.result_label1.setText(f"Data has been saved to {save_path}")

        # Reset incase user wants to repeat the cognition process
        self.saved = True

    def recog_change_cognition_results(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Cognition Results", "", "Excel Files (*.xlsx)") # Select file
        if file_path:
            settings =  pd.read_excel(self.settings_addr, header = None)
            for i, value in enumerate(settings[0]):
                if value == "cog_addr":
                    settings[1][i] = file_path
            settings.to_excel(self.settings_addr, header = None, index=False)
            self.settings_update()
            self.result_label2.setText("Cognition Results Address Has Been Updated in Settings")

    def recog_select_new_images(self):
        # Modify: To connect to video feed in the future
        # Retrieve Classes in Excel
        self.recog_load_cognition_results()

        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "") # Get image address
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):
            self.original_image = cv2.imread(file_name)

            # Find out what is the max split for height and width so images will always downsize
            height, width, _ = self.original_image.shape # Get height and width

            # Split image and run
            split_images_list, self.image_data_list = [], []
            k = 0
            for split in self.settings['sk_list'][0]:
                # Run Split Image Sub-Function, split_images returned with information
                split_images, image_data = gcal.image_splitter(split, self.original_image)
                if self.settings['save'][0] == True:
                    for img in split_images:
                        filename = f"Saved Files/Testing/Image[{k}], Split[{split[0]}x{split[1]}].jpg" # Testing
                        cv2.imwrite(filename, img)
                        k += 1
                split_images_list.extend(split_images)
                self.image_data_list.extend(image_data)
            gcal.k_reset() # To reset global variable k to reuse for image splitting

            processed_split_images_list = []
            if self.settings['resize'][0] == True:
                for _, data in self.class_features.items(): # .items() makes variables classname = label, data = data for that classname
                    dimensions = data['dimensions']
                    for split_image in split_images_list:
                        resized_image = cv2.resize(split_image, (dimensions[1], dimensions[0]), interpolation=cv2.INTER_AREA)
                        processed_split_images_list.append(resized_image)
            else:
                processed_split_images_list = split_images_list

            # Current system designed to search for only one specific class, else need to recalculate feature vectors for every different resize in the classes list   
            self.time_new_images_features, self.freq_new_images_features = fcal.calculate_feature_vectors(processed_split_images_list) # Returns 48 feature vectors for each split image, split between time and frequency
        
        self.recog_calculate_possibility()
    
    def recog_load_cognition_results(self):
        # Modify to have location in a setting instead
        class_names_html = [] # Make empty list
        self.class_features = {} # Make empty dictionary

        try:
            df = pd.read_excel(self.settings['cog_addr'][0]) # Extract Dataframe from Excel
            addr_valid = True
        except FileNotFoundError:
            addr_valid = False

        if addr_valid == True:
            for _, row in df.iterrows(): # iterrows() extracts the item of each header for each class in dataframe
                class_name = row['Class Name'] # Extract item under label 'Class Name'
                dimensions = ast.literal_eval(row['Dimensions']) # Need convert from string to list format
                time_variance = row['Time Variance'] # Extract 'Variance'
                freq_variance = row['Freq Variance']
                mean_vector = row[4:].values # Skip 'Class Name', "Dimensions" & 'Variances
                self.class_features[class_name] = {"dimensions": dimensions, "mean": mean_vector, "time_variance": time_variance, "freq_variance": freq_variance} # Makes a dictionary in a dictionary
            
            for class_name, _ in self.class_features.items():
                # Add the class name to the HTML list to display it later in a text editor
                class_names_html.append(f"<font size='5' color='red'><b>{class_name}</b></font><br>")
            
            # Concatenate all class names into a single string and set it to the HTML of the result tag
            self.result_label2.setHtml("Loaded class names:<br>" + "".join(class_names_html))
        else:
            # Print Error
            self.result_label2.setHtml("Error: Invalid Cognition Results File Address")

    def recog_calculate_possibility(self):
        # Checker if class_features exist
        if not self.class_features:
            self.result_label2.setText("Please load cognition results first.")
            return
        
        # Checker if new_images loaded
        if not hasattr(self, 'time_new_images_features') or not self.time_new_images_features or not hasattr(self, 'freq_new_images_features') or not self.freq_new_images_features:
            self.result_label2.setText("Please select new images first.")
            return
        
        highest_similarity, best_match = gcal.calculate_possibility(self.settings['threshold'][0], self.settings['weights'][0], self.settings['save'][0], self.class_features, self.time_new_images_features, self.freq_new_images_features)
        results, prob_str, probs_strings = [], [], []
        # Displaying highest similarity for each class
        for val in highest_similarity:
            probs_strings.append(f"{val[0]}: {val[1]:.4f}")
        
        # Combine the class possibilities into a string
        prob_str = ', '.join(probs_strings)
        # print(prob_str)

        # Display possibilities and best match in separate lines
        results.append(
            f"Image Possibilities: {prob_str}.<br>"
            f"<span style='color: red; font-size: 18px;'><b>Best match: {best_match[0]}</b></span>"
        )

        # Combine all image results
        self.result_label2.setText("<br><br>".join(results))

        # Display Localized Image
        if best_match[0] != "None":
            _, height_start, height_end, width_start, width_end = self.image_data_list[best_match[2]]
            self.original_image = cv2.rectangle(self.original_image, (width_start,height_start), (width_end,height_end), (0,0,255), 2)
            self.original_image = cv2.rectangle(self.original_image, (width_start,height_start), (width_start + 6*20 + len(best_match[0] + str(round(best_match[1],4)) + str(best_match[2]))*20,height_start+30), (0,0,0), -1)
            self.original_image = cv2.putText(self.original_image, f"{best_match[0]}:{round(best_match[1],4)}, Image[{best_match[2]}]", (width_start,height_start+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            if self.settings['save'][0] == True:
                filename = f"Saved Files/Results/Result.jpg" # Testing
                cv2.imwrite(filename, self.original_image)
            
            cv2.imshow("Image", self.original_image)
            # Waits for user to press any key 
            # (this is necessary to avoid Python kernel form crashing) 
            cv2.waitKey(0)

            # Closing all open windows 
            cv2.destroyAllWindows() 
        
        gcal.k_reset() # To reset global variable k

if __name__ == "__main__": # This ensures this part will only run as main program
    app = QApplication(sys.argv) # Create pyqt5 app
    window = GUI() # Make window a Class object (impt)
    window.show() # Show GUI
    sys.exit(app.exec_()) # Clean Exit when closing GUI