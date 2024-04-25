import cv2
import numpy as np
import math
import pandas as pd

# Get images and resize
def resizer(save, raw_image_list, dimensions, classname):
    images_list = [] # List to store processed images
    if dimensions[0] == -1 and dimensions[1] == -1:
        images_list = raw_image_list
    else:
        for i, image in enumerate(raw_image_list):
            resized_img = cv2.resize(image, (dimensions[1], dimensions[0]), interpolation=cv2.INTER_AREA) # Resize, this best for downsizing
            images_list.append(resized_img)
    if save == True:
        for i, img in enumerate(images_list):
            filename = f"Saved Files/Training/{classname} Image[{i}].jpg" # Testing
            cv2.imwrite(filename, img)
    return images_list

k = 0 # To keep count even when function resets

def image_splitter(split, image):
    # Variables
    split_images, image_data = [], []

    # Amount of pixels to split into
    o_height, o_width, _ = image.shape # Get height and width
    pix_height = o_height/split[0]
    pix_width = o_width/split[1]

    # Split Image
    for i in range(0, split[0]):
        for j in range(0, split[1]):
            height_start = int(i*pix_height)
            height_end = int((i+1)*pix_height-1)
            width_start = int(j*pix_width)
            width_end = int((j+1)*pix_width-1)
            split_image = image[height_start:height_end, width_start:width_end]

            # Obtain list of images and data in sequence
            split_images.append(split_image)
            image_data.append([split, height_start, height_end, width_start, width_end])
    
    return split_images, image_data

def k_reset():
    global k
    k = 0

def calculate_possibility(threshold, weights, save, class_features, time_new_images_features, freq_new_images_features):
    # Dictionary to store possibilities
    new_images_possibilities = {}
    global k

    for classname, data in class_features.items(): # .items() makes variables classname = label, data = data for that classname
        # This part gets data for each classname stored in Excel
        mu_f = data['mean'] # Extract 'mean' from dictionary data, 48 values
        time_variance_f = data['time_variance'] # Same here
        freq_variance_f = data['freq_variance']
        possibilities_time, possibilities_freq= [], [] # Empty possibility list

        # Splitting Time from Frequency for each class in Excel
        mu_split = np.array_split(mu_f,2)

        # f_new is the 48 values of each split image
        for f_new in time_new_images_features:
            # Time possibility
            error_time = f_new - mu_split[0] # f_new 48 values minus 48 values from Excel
            exponent_time = -0.5 * np.dot(error_time, error_time) / time_variance_f
            possibility_time = math.exp(exponent_time)
            possibilities_time.append(possibility_time)

        for f_new in freq_new_images_features:
            # Frequency possibility
            error_freq = f_new - mu_split[1] # f_new 48 values minus 48 values from Excel
            exponent_freq = -0.5 * np.dot(error_freq, error_freq) / freq_variance_f
            possibility_freq = math.exp(exponent_freq)
            possibilities_freq.append(possibility_freq)

        new_images_possibilities[classname] = {"possibilities_time": possibilities_time, "possibilities_freq": possibilities_freq} # Stored all possiblities for each class as dictionary separated by time and frequency

    # Recognition Initialization
    #print(new_images_possibilities) # Testing
    highest_similarity = []
    highest = 0.0 # Set similarity as -ve infinity
    key = -1

    # Extract classname and probilities for each image and find the highest possibility for each class and sub image
    for classname, probs in new_images_possibilities.items():
        save_possibilities = {}

        # Extract possibilities for each class
        prob_time = np.array(probs['possibilities_time'])
        prob_freq = np.array(probs['possibilities_freq'])
        if save == True:
            decimal = 4
            list_prob_time = list(np.around(prob_time,decimal))
            list_prob_freq = list(np.around(prob_freq,decimal))
        # print(classname + ": Time" + str(prob_time) + ", Freq" + str(prob_freq) + "\n") # Testing

        # Continue: Make Filter to remove all values below threshold and their respective key, refer to Image Splitting Test 2
        # Look for indexes that are below threshold and set to zero
        # This applies to index so both probability must pass threshold to remain
        index_time = np.argwhere(prob_time < threshold)
        index_freq = np.argwhere(prob_freq < threshold)
        index_time = index_time.flatten()
        index_freq = index_freq.flatten()
        index = np.concatenate((index_time,index_freq))
        index = np.unique(index)
        prob_time[index] = 0
        prob_freq[index] = 0

        combined_probs = weights[0]*prob_time + weights[1]*prob_freq  # Combining both probabilities with weights on each side, 0.5 for average
        if save == True:
            if k == 0:
                w_mode = 'w'
            else:
                w_mode = 'a'
            list_combined_probs = list(np.around(combined_probs,decimal))
            save_possibilities[classname] = {"possibilities_time": list_prob_time, "possibilities_freq": list_prob_freq, "possibilities_combined": list_combined_probs}
            with pd.ExcelWriter("Saved Files/Results/Possibilities.xlsx", mode=w_mode) as writer:
                temp_df = pd.DataFrame(save_possibilities[classname])
                temp_df.to_excel(writer, sheet_name=classname)
            #print(save_possibilities) # Testing
            k += 1

        highest_similarity_index = combined_probs.argmax(axis=0) # Find highest probability with position
        highest_similarity.append([classname, combined_probs[highest_similarity_index], highest_similarity_index]) # Append into list
   
    # Find the highest similarity to be the final class with relavant position in the list
    for i, value in enumerate(highest_similarity):
        if (value[1] > highest) and (value[2] != -1):
            highest = value[1]
            key = i
    
    if key != -1:
        best_match = highest_similarity[key]
    else:
        best_match = ['None', 0, -1]

    return highest_similarity, best_match