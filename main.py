# Importing libraries
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from tkinter import messagebox

class AIStitchEdgeApp:
	def __init__(self, master):
		self.master = master
		self.master.title("AI-StitchEdge")

		self.images = [] # list of uploaded images
		self.stitched_image = None

		# Buttons for all operations

		self.upload_button = tk.Button(master, text="Upload Images", command=self.imageUploader)
		self.upload_button.pack(pady=10)

		self.stitch_button = tk.Button(master, text="Stitch Images", command=self.imageStitcher)
		self.stitch_button.pack(pady=5)

		self.stitch_button = tk.Button(master, text="Apply Canny", command=self.cannyApply)
		self.stitch_button.pack(pady=5)

		self.stitch_button = tk.Button(master, text="Apply DoG", command=self.DoGApply)
		self.stitch_button.pack(pady=5)

		self.stitch_button = tk.Button(master, text="Human Detection", command=self.humanDetection)
		self.stitch_button.pack(pady=5)

		self.morphOp_label = tk.Label(master, text="Enter kernel size: ")
		self.morphOp_box = tk.Spinbox(master, from_=1, to=10)

		# Create canvas to display images
		self.canvas = tk.Canvas(master, width=600, height=400)
		self.canvas.pack()

	def imageUploader(self):
		"""
		Upload images to be stitching
		:return: uploaded images in one figure
		"""

		# Get rade of previous stitched image when loading new images to be stitching
		self.stitched_image = None

		# Specify all the file types of images we accept
		fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]

		# Call the function from filedialog module to accept the image
		images_path = tk.filedialog.askopenfilenames(filetypes=fileTypes)

		# If file is selected
		if len(images_path):

			s = len(images_path)
			plt.figure(figsize=(s*5,5))
			for idx, img in enumerate(images_path):

				# Create figure with one raw and columns equal to number of images (s)
				# then plot every image in their index
				plt.subplot(1, s, idx + 1)

				# Read image by using cv2, convert it to RGB scale and resize it
				img = cv2.imread(img)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = cv2.resize(img, (200, 200))

				# Add image to images list
				self.images.append(img)

				# show image in their plot
				plt.imshow(img)

			# Show the full plot of all images in one figure
			plt.show()

		else:

			# Show warning message if no file is selected
			messagebox.showwarning('Warning', "No file is choosen !! Please choose a file.")
	def imageStitcher(self):
		"""
		Apply stitching operation on input images by using stitcher to get
		one stitched image as a result
		:return: stitched image
		"""

		self.stitched_image = None

		# Confirm that the stitching operation need 2 images at least
		if len(self.images) < 2:
			messagebox.showwarning('Warning', "Please upload at least 2 images.")
			return

		# Define and use the stitcher
		stitcher = cv2.Stitcher_create()
		status, self.stitched_image = stitcher.stitch(self.images)

		# Check if the stitching operation complete successfully which return 0 for success
		if status == 0:

			# Show stitched image
			plt.imshow(self.stitched_image)
			plt.show()

		else:

			# Show warning message if stitching faild
			messagebox.showerror("Error", 'Stitching Faild.')

	def cannyApply(self):

		"""
		Apply Canny detection on stitching image
		:return: Canny result on stitching image
		"""

		if self.stitched_image is not None:

			# Convert stitched image to gray
			gray = cv2.cvtColor(self.stitched_image, cv2.COLOR_RGB2GRAY)

			# Calculate the two thresholds of gray image
			median = np.median(gray)
			lower = int(0.68 * median)
			upper = int(1.32 * median)

			# Apply Canny edge detection using the computed thresholds
			canny = cv2.Canny(gray, lower, upper)

			# Show the Canny edge detection on stitched image in gray scale
			plt.imshow(canny, cmap='gray')
			plt.show()

		else:

			# Show warning message if there is no stitched image
			messagebox.showwarning('Warning', 'No stitched Image!')
	def DoGApply(self):

		"""
		Apply different of gaussians on stitched image for blob detection in image
		:return: DoG result in stitched image
		"""
		if self.stitched_image is not None:

			# Convert stitched image to gray
			gray = cv2.cvtColor(self.stitched_image, cv2.COLOR_RGB2GRAY)

			# Convolve the gray image with two gaussians (one with a scale of 1 and another with a scale of 3)
			s = 31
			gus_1sc_img = cv2.GaussianBlur(gray, (s, s), 1)
			gus_3sc_img = cv2.GaussianBlur(gray, (s, s), 3)

			# Apply difference of gaussians (DoG)
			DoG = gus_1sc_img - gus_3sc_img

			'''plt.imshow(DoG, cmap='gray')
			plt.show()'''

			# Calling morphOp to apply morphology operation to enhance the result
			self.morphOp(DoG)

		else:

			# Show warning message if there is no stitched image
			messagebox.showwarning('Warning', 'No stitched Image!')
	def morphOp(self,DoG):
		"""
		enhance the DoG results using opening or closing morphology operation
		:return: enhanced DoG result on stitched image
		"""
		# Appear the label text and the box to entry kernel size by user
		self.morphOp_label.pack(pady=5)
		self.morphOp_box.pack(pady=5)

		# Get the kernel size entered in morphOp_box as int
		k = int(self.morphOp_box.get())

		# Create a mask with suitable size choosen by the user
		kernel = np.ones((k,k))

		# Shape then use opening then closing
		op_img = cv2.morphologyEx(DoG, cv2.MORPH_OPEN, kernel)
		op_img = cv2.morphologyEx(DoG, cv2.MORPH_CLOSE, kernel)

		# Show the enhanced DoG after choosen
		plt.imshow(op_img, cmap='gray')
		plt.show()
	def humanDetection(self):

		"""
		AI-based object detection model capable of
		identifying human figures within the stitched image.
		:return: stitched image with human detection
		"""

		if self.stitched_image is not None:

			# Load and pretrained yolov8m model
			model = YOLO("yolov8m.pt")

			# Run the model to predict person(0 class) with confidence level above 50% in stitched image
			results = model.predict(self.stitched_image, show=True, imgsz=320, conf=0.5, classes=0)
			result = results[0]

			# Confirm the user that there is no person in image if there is not
			if len(result) == 0:
				messagebox.showinfo('Info', 'No person found in image, try another one')

			else:
				for box in result.boxes:
					class_id = result.names[box.cls[0].item()]
					cords = box.xyxy[0].tolist()
					cords = [round(x) for x in cords]
					conf = round(box.conf[0].item(), 2)
					print("Object type:", class_id)
					print("Coordinates:", cords)
					print("Probability:", conf)
					print("---")
		else:

			# Show warning message if there is no stitched image
			messagebox.showwarning('Warning', 'No stitched Image!')



# Main method
if __name__ == "__main__":

	# Create instance of Tkinter class to upload images
	root = tk.Tk()
	app = AIStitchEdgeApp(root)
	root.mainloop()


