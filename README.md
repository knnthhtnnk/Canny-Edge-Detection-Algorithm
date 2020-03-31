# Canny Edge Detection Algorithm

## **Using OpenCV v4.2.0 with Visual Studio 2019**

### Main Function
Firstly, the main function prompts the user to choose which image they wish to perform Canny Edge Detection on. The chosen image will then be converted from its original .raw format into an OpenCV Mat object by the function described in **§1** below.

Next, the user is prompted to input a standard deviation of their choice. This standard deviation will be used to generate the Gaussian kernel in **§2** below which is used to smooth out any noise present in the image. The generated Gaussian kernel is then displayed in the command window along with its dimensions and weighted sum.

Before the image was convoluted with the Gaussian kernel, it was padded (reflection) in order to avoid a dimension reduction at the borders. This was implemented with the function described in **§3** below.

The function in **§4** initialises new Mat objects with the similar dimensions as the source but populated with zeros.

After padding the image, a 2-dimensional convolution of the image with the generated Gaussian kernel was then performed with the use of the function described in **§5** below, resulting image in a smoothed image.

Once smoothing is complete, the edge gradients in the x-direction and y-direction were then computed with the function described in **§6** below, which outputs the image’s edge map.

Then, given that the edges in the edge map were thick and fine details of the original image were lost, it was necessary to perform non-maximum suppression onto the edge map. This involves scanning through every pixel in the edge map and computing the local maximum. If a particular pixel is not the local maximum, it is suppressed to 0, but if it is a local maximum, its original value is preserved. This was implemented with the function described in **§7** below. After non-maximum suppression was performed, the resulting image now has thinned edges, allowing the fine details of the original image to be studied.

However, it was observed that the thinned image still contained a substantial amount of unwanted noise. This issue was addressed by subjecting the thinned image to hysteresis thresholding. In hysteresis thresholding, a minimum and maximum threshold is pre-set, and any pixel with pixel value less than the minimum threshold is considered to be a weak-edge and suppressed to zero while those with pixel values more than the maximum threshold are considered to be strong-edges and are raised to 255 (The maximum pixel value is 255 as OpenCV Mat objects of type CV_8UC1 are 8-bit unsigned, single channel containers). This process was implemented with the function described in **§8** below.

Finally, after hysteresis thresholding, the resulting image was then displayed in a new window, along with the original and thinned images.


### §1. Converting .RAW files into OpenCV Mat objects
The images provided possessed a .raw file extension which indicated that none of the image data were compressed, which makes the images higher in quality. However, this means that the images are in binary format, which OpenCV is unable to display using its in-built imshow() function. Therefore, it is necessary to convert the binary data of the images into a Mat object which OpenCV can then parse and display.

The pseudocode of this function is as follows:

	FUNCTION Convert Binary Image to Mat Object
		Declare the chosen image as an input for the file stream, indicating that it is in binary
	 	Declare a vector of unsigned characters to contain the image data
	 	Declare a vector of unsigned characters to contain the dimensions of the image
	 	Declare a Mat object which is of the CV_8UC1 (8-bit unsigned, single channel) type
	 	Copy the image data from the vector into the Mat object
	 	Return Mat image
	END FUNCTION

### §2. Generating the Gaussian Smoothing Kernel
Before doing any further processing of the image, smoothing of the image using a Gaussian Kernel should be carried out to remove the noise.

The pseudocode of this function is as follows:

	FUNCTION Generating Gaussian Smoothing Kernel
		Declare temporary sigma variable with type double with initial value of 0.0
		Declare the kernel row and index variables with type double with initial values of 0.0
		IF Standard Deviation (SD) is greater than or equal to 1
			Assign to temporary sigma variable the ceiling value of SD
			IF ceiling(SD) modulo 2 is equal to 0 *SD is an even number*
				Assign kernel row to be 5 times of SD plus 1
			END IF
			IF ceiling(SD) modulo 2 is equal to 1 *SD is an odd number*		
				Assign kernel row to be 5 times of SD
			END IF
		END IF
		IF SD is less than 1
			Assign the kernel row to 3, which is the smallest kernel size
		END IF
		Assign the kernel index to be equal to the kernel row minus 1 then divided by 2
		Call the Gaussian Formula with x and y both equal to the complement of the kernel index
		FOR every row of the kernel with post-incrementation
			Declare a temporary vector with type double to hold kernel values
			FOR every column of the kernel with post-incrementation
				Compute value of each pixel with the Gaussian Formula
				Assign the computed pixel value to the temporary kernel
				Compute the weighted sum of the kernel
			END FOR
			Assign the computed pixel value to the actual Gaussian kernel
		END FOR
		Return weighted sum of the kernel
	END FUNCTION

### §3. Reflection Padding
The pseudocode of this function is as follows:

	FUNCTION Reflection Padding of Image
		Declare the number of pixels to pad the image, which is dependent on the kernel size
		Declare a Mat object with dimensions of original image with the added padding pixels
		Call the in-built copyMakeBorder() function to generate the padding pixels
		Return the resulting padded image
	END FUNCTION

### §4. Initialisation of a New Image
The pseudocode of this function is as follows:

	FUNCTION Initialising a New Mat Object with Zeros
		Declare output image as a Mat object with same dimensions as the source image
		FOR every row of the output image with post-incrementation
			FOR every column of the output image with post-incrementation
				IF the source image is of type CV_8UC1 (8-bit unsigned single channel)
					Assign the pixel of output image to 0.0 of type unsigned char
				END IF
				IF the source image is of type CV_32SC1 (32-bit signed single channel)
					Assign the pixel of output image to 0.0 of type integer
				END IF
			END FOR
		END FOR
		Return output image
	END FUNCTION

### §5. Convolution with Kernel
The pseudocode of this function is as follows:

	FUNCTION Convolution of Image with Kernel
		Declare output image
		Declare centre of kernel w.r.t x direction as the floor of the kernel width divided by 2
		Declare centre of kernel w.r.t y direction as the floor of the kernel height divided by 2
		IF the signs of the pixel values want to be preserved
			Convert source image into type CV_32SC1 (32-bit signed single channel)
		ELSE *signs of pixel values do not need to be preserved*
			Convert source image into type CV_8C1 (8-bit unsigned single channel)
		END IF
		FOR every row of the source image with post-incrementation
			FOR every column of the source image with post-incrementation
				Declare variable of type double to contain temporary pixel value
				Declare variable of type double to contain weighted sum of kernel
				FOR every row of the kernel with post-incrementation
					FOR every column of the kernel with post-incrementation
						Assign to weighted sum the summation of kernel values
						Assign to temporary pixel value the summation of kernel
						values multiplied by the pixel value of source image
					END FOR
				END FOR
				IF the signs of the pixel values want to be preserved
					Assign to output image the temporary pixel value divided by the weighted sum with type integer
				ELSE *signs of pixel values do not need to be preserved*
					Assign to output image the temporary pixel value divided by the weighted sum with type unsigned char
				END IF
			END FOR
		END FOR
		Return output image
	END FUNCTION

### §6. Obtaining the Edge Map
The pseudocode of this function is as follows:

	FUNCTION Obtaining Edge Map
		Declare output image
		Compute the absolute values of the horizontal edge map
		Compute the absolute values of the vertical edge map
		Assign to the output image the summation of the horizontal & vertical edge maps
		Return output image
	END FUNCTION

### §7. Non-Maximum Suppression
After obtaining the edge map, it can be observed that the edges are thick, and one is unable to make out the fine details of the image. Therefore, non-maximum suppression was utilised in order to thin the edges so that the edge map can be studied in greater detail. This was achieved by scanning through every pixel within the edge map and computing the local maximum. If a particular pixel is the local maximum, it will retain its pixel value, however, if it is not the local maximum, its pixel value will be suppressed, i.e. assigned a zero value. Only the four orthogonal directions, horizontal, vertical, top left to bottom right, and bottom left to top right, were taken into consideration.

The pseudocode for this function is as follows:

	FUNCTION Non-Maximum Suppression to Thin Edges of Edge Map
		Declare 4 Mat objects for the horizontal, vertical, and 2 oblique (top left to bottom right & bottom left to top right) components
		Initialise them with same dimensions as the edge map but populated with zeros
		Convert them to type CV_8UC1 (8-bit unsigned single channel)
		*Horizontal Component*
		FOR every row of the edge map with post-incrementation
	  		FOR every column of the edge map with post-incrementation
		  		IF pixel value (x, y) is greater than pixel value at (x-1, y) & (x+1, y)
					Assign pixel value of (x, y) to that pixel in horizontal Mat
		  		ELSE
					Assign 0 to that pixel in the horizontal Mat
		  		END IF	
	  		END FOR
  		END FOR
  		*Vertical Component*
  		FOR every row of the edge map with post-incrementation
	  		FOR every column of the edge map with post-incrementation
		  		IF pixel value (x, y) is greater than pixel value at (x, y-1) & (x, y+1)
			  		Assign pixel value of (x, y) to that pixel in vertical Mat
		  		ELSE
			  		Assign 0 to that pixel in the vertical Mat
		  		END IF
	  		END FOR
  		END FOR
  		*Oblique1 (Top Left to Bottom Right) Component*
  		FOR every row of the edge map with post-incrementation
	  		FOR every column of the edge map with post-incrementation
		  		IF pixel value (x, y) is greater than pixel value at (x+1, y-1) & (x-1, y+1)
			  		Assign pixel value of (x, y) to that pixel in oblique1 Mat
		  		ELSE
			  		Assign 0 to that pixel in the oblique Mat
		  		END IF
	  		END FOR
  		END FOR
  		*Oblique2 (Bottom Left to Top Right) Component*
  		FOR every row of the edge map with post-incrementation
	  		FOR every column of the edge map with post-incrementation
		  		IF pixel value (x, y) is greater than pixel value at (x-1, y-1) & (x+1, y+1)
			  		Assign pixel value of (x, y) to that pixel in oblique2 Mat
		  		ELSE
			  		Assign 0 to that pixel in the oblique Mat
		  		END IF
	  		END FOR
  		END FOR
  		Assign to the output image the summation of all 4 components
  		Return output image
	END FUNCTON













