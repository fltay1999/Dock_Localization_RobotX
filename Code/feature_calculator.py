import cv2
import numpy as np

def calculate_feature_vectors(images_list):
    f = 0
    time_feature_vectors_list, freq_feature_vectors_list= [], []
    for image in images_list:
        # Separate the RGB channels
        # Convert the image to float for more precise calculations
        image = image.astype(np.float64) # Convert to float64 format
        b_channel, g_channel, r_channel  = cv2.split(image) # Split to RGB, CV2 is BGR

        # Calculate Time features for each channel
        r_time_features, r_0th, r_1st = calculate_time_features(r_channel)
        g_time_features, g_0th, g_1st = calculate_time_features(g_channel)
        b_time_features, b_0th, b_1st = calculate_time_features(b_channel)

        # Calculate Frequency features for each channel
        r_freq_features, r_real, r_img = calculate_freq_features(r_channel)
        g_freq_features, g_real, g_img = calculate_freq_features(g_channel)
        b_freq_features, b_real, b_img = calculate_freq_features(b_channel)

        # Testing
        if(f == 2):
            cv2.imwrite("Processed Image/0th_Red.jpg", r_0th)
            cv2.imwrite("Processed Image/1th_Red.jpg", r_1st)
            cv2.imwrite("Processed Image/0th_Green.jpg", g_0th)
            cv2.imwrite("Processed Image/1st_Green.jpg", g_1st)
            cv2.imwrite("Processed Image/0th_Blue.jpg", b_0th)
            cv2.imwrite("Processed Image/1st_Blue.jpg", b_1st)
            cv2.imwrite("Processed Image/Real_Red.jpg", r_real)
            cv2.imwrite("Processed Image/Img_Red.jpg", r_img)
            cv2.imwrite("Processed Image/Real_Green.jpg", g_real)
            cv2.imwrite("Processed Image/Img_Green.jpg", g_img)
            cv2.imwrite("Processed Image/Real_Blue.jpg", b_real)
            cv2.imwrite("Processed Image/Img_Blue.jpg", b_img)
        f += 1

        time_feature_vectors = np.concatenate([r_time_features, g_time_features, b_time_features])
        freq_feature_vectors = np.concatenate([r_freq_features, g_freq_features, b_freq_features])
        time_feature_vectors_list.append(time_feature_vectors)
        freq_feature_vectors_list.append(freq_feature_vectors)
    return time_feature_vectors_list, freq_feature_vectors_list

def calculate_time_features(image):
    # Initialization
    time_feature_vectors = []
    gaussian_filter = np.array([[1/16,2/16,1/16],
                                [2/16,4/16,2/16],
                                [1/16,2/16,1/16]])
    sobel_h_filter = np.array([[1,0,-1],
                               [2,0,-2],
                               [1,0,-1]])
    sobel_v_filter = np.array([[1,2,1],
                               [0,0,0],
                               [-1,-2,-1]])
    
    # Gaussian filtered image
    zero_order_image = cv2.filter2D(src=image, ddepth=-1, kernel=gaussian_filter, borderType=0)

    # Need convert from uint8 to int to increase cap of integer to prevent overflow
    # All values above 255 after calculations used ratio method to get value btwn 0 to 255 and rounded 
    # Return image back to uint8 afterwards
    first_order_h = cv2.filter2D(src=zero_order_image, ddepth=-1, kernel=sobel_h_filter, borderType=0)
    first_order_v = cv2.filter2D(src=zero_order_image, ddepth=-1, kernel=sobel_v_filter, borderType=0)
    first_order_h = first_order_h.astype(int)
    first_order_v = first_order_v.astype(int)
    combined_sobel = np.round(np.sqrt(np.square(first_order_h) + np.square(first_order_v))*(255/(np.sqrt(np.square(255) + np.square(255)))))    
    combined_sobel = combined_sobel.astype(np.uint8)

    # Calculate Feature Vectors
    feature_vector_zero = statistics(zero_order_image)
    feature_vector_first = statistics(combined_sobel)
    time_feature_vectors.extend(feature_vector_zero)
    time_feature_vectors.extend(feature_vector_first)

    return time_feature_vectors, zero_order_image, combined_sobel

def calculate_freq_features(image):
    # Initialize the feature vector
    freq_feature_vectors = []

    # Compute the FFT of the channel
    f = np.fft.fft2(image) # Compute the 2-dimensional discrete Fourier Transform of channel(an array)

    # Split the FFT output into its real and imaginary parts
    real_part = f.real
    imaginary_part = f.imag

    # Set a threshold
    threshold = 10
    # Apply the threshold to the real and imaginary parts, anything below treshold to zero
    real_part[real_part < threshold] = 0
    imaginary_part[imaginary_part < threshold] = 0

    # Calculate Feature Vectors
    feature_vector_real = statistics(real_part)
    feature_vector_img = statistics(imaginary_part)
    freq_feature_vectors.extend(feature_vector_real)
    freq_feature_vectors.extend(feature_vector_img)
    return freq_feature_vectors, real_part, imaginary_part

def statistics(input):
    mean = np.mean(input)
    std = np.std(input)
    h, w = input.shape # height and width using shape
    # meshgrid makes an 2D array using x axis and y axis same as pixels in images
    # arange make a range 0 to number stated, x makes np.arange(w) as row, y makes it as col
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    epsilon = 1e-8  # small positive value in case divide by zero
    total_weight = np.sum(input) + epsilon
    # Sum with divide total weight
    centroid_x = np.sum(x*input) / total_weight
    centroid_y = np.sum(y*input) / total_weight
    # If below epsilon, make it epsilon
    variance_x = np.sum(((x - centroid_x) ** 2) * input) / total_weight
    variance_y = np.sum(((y - centroid_y) ** 2) * input) / total_weight
    # Find dispersion
    dispersion_x = np.sqrt(variance_x) if variance_x >= epsilon else 0
    dispersion_y = np.sqrt(variance_y) if variance_y >= epsilon else 0
    return [mean, std, dispersion_x, dispersion_y]

def get_mean_and_variance(time_feature_vectors_list, freq_feature_vectors_list):
    # print(feature_vectors.shape) # Testing
    time_mean_feature_vector = np.mean(time_feature_vectors_list, axis=0) # Find mean along column, meaning the mean of each first element, second, etc.
    freq_mean_feature_vector = np.mean(freq_feature_vectors_list, axis=0)
    time_error_vectors = time_feature_vectors_list - time_mean_feature_vector
    freq_error_vectors = freq_feature_vectors_list - freq_mean_feature_vector
    time_squared_distances = np.sum(time_error_vectors**2, axis=1) # Find sum along row
    freq_squared_distances = np.sum(freq_error_vectors**2, axis=1) # Find sum along row
    time_variance = np.mean(time_squared_distances)
    freq_variance = np.mean(freq_squared_distances)
    return time_mean_feature_vector, freq_mean_feature_vector, time_variance, freq_variance