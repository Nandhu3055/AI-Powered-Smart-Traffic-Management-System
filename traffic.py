# Smart Traffic Management System using YOLOv8

import customtkinter
import cv2
import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog,Canvas,Scrollbar
from ultralytics import YOLO
from PIL import Image
import customtkinter as ctk
import time
from database import UserClass
import sqlite3

db = UserClass()
class App(customtkinter.CTk):
    video_paths = []
    caps = []
    model = YOLO("models/yolov8n.pt")
    carTime, bikeTime, rickshawTime, busTime, truckTime = 3, 2, 3.25, 3.5, 3.5
    defaultMinimum, defaultMaximum = 10, 60

    # Store results to avoid re-processing the same video
    cached_results = {}

    # Store drawn region
    roi_points = []
    roi_defined = False

    def __init__(self):
        super().__init__()

        self.title("Smart Traffic Management System.py")
        self.geometry("700x450")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1) 
        self.grid_columnconfigure(1, weight=1)

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="  Traffic Management  ",
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Set Lane",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                    anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.result_button = customtkinter.CTkButton(
            self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="View Results",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
            anchor="w", command=self.frame_2_button_event
        )
        self.result_button.grid(row=2, column=0, sticky="ew")

        self.about_us_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="About Us",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.frame_3_button_event)
        self.about_us_button.grid(row=3, column=0, sticky="ew")


        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        self.home_label = customtkinter.CTkLabel(self.home_frame, text="Select a video file to process",
                                                             compound="center", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.home_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_frame_large_image_label = customtkinter.CTkLabel(self.home_frame, text="")
        self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=10)

        self.home_frame_button_1 = customtkinter.CTkButton(self.home_frame, text="Select Videos",command= self.select_file)
        self.home_frame_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.home_frame_button_2 = customtkinter.CTkButton(self.home_frame, text="Select ROI",command= self.select_roi)
        self.home_frame_button_2.grid(row=2, column=0, padx=20, pady=10)

        # create second frame
        self.second_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.second_frame.grid_columnconfigure(0, weight=1)

        # create third frame
        self.third_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.third_frame.grid_columnconfigure(1, weight=1)
        # select default frame
        self.select_frame_by_name("home")

    def select_roi(self): 
        self.render_frame(self.video_paths,self.caps)

    def select_file(self):
        for i in range(4):
            file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
            if file_path:
                self.video_paths.append(file_path)
                self.caps.append(cv2.VideoCapture(file_path)) 

    rois = []
    results = []
    first_frames = []
    def render_frame(self,file_paths,caps):
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Cannot read frame from {file_paths[i]}")
                exit()
            else:    
                self.first_frames.append(frame) 
                roi = cv2.selectROI(f"Select ROI for Video {i+1}", frame, showCrosshair=True)
                self.rois.append(roi)
                print(self.rois)
                cv2.destroyWindow(f"Select ROI for Video {i+1}")    

        for i in range(4):
            db.insert_data(file_paths[i],i,self.rois[i][0],self.rois[i][1],self.rois[i][2],self.rois[i][3])
            frame,total_vehicles, greenTime, noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws = self.process_frame(self.first_frames[i], self.rois[i])
            self.results.append((self.first_frames[i],total_vehicles, greenTime, noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws)) 

        self.home_frame_image_label = customtkinter.CTkLabel(self.home_frame, text="You can view the results in View results tab !")
        self.home_frame_image_label.grid(row=2, column=0, padx=20, pady=10)

        self.display(self.video_paths)  


    def process_frame(self,frame,roi):
        # global roi_defined, roi_points

        # Initialize vehicle counters
        noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws = 0, 0, 0, 0, 0

        # Get bounding rectangle of ROI and crop it
        x, y, w, h = roi
        cropped_frame = frame[y:y+h, x:x+w]

        # Display the processed frame

        # Run YOLO on cropped frame
        results = self.model(cropped_frame)
        df = pd.DataFrame(results[0].boxes.data.cpu().numpy(), columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"])
        # print(df)
        for _, row in df.iterrows():
            if int(row["class"]) in [0,1,2,3,5,7,9]: 
            # Adjust bounding box coordinates to original frame if ROI was used
                x1, y1, x2, y2 = int(row["xmin"]) + x, int(row["ymin"]) + y, int(row["xmax"]) + x, int(row["ymax"]) + y

                # Get vehicle class
                vehicle_class = self.model.names[int(row["class"])]
                flag = 0
                if vehicle_class in ["car", "suv"]:
                    noOfCars += 1
                    flag = 1
                elif vehicle_class in ["motorbike", "bicycle"]:
                    noOfBikes += 1
                    flag = 1
                elif vehicle_class in ["bus"]:
                    noOfBuses += 1
                    flag = 1
                elif vehicle_class in ["truck"]:
                    noOfTrucks += 1
                    flag = 1
                elif vehicle_class in ["rickshaw", "auto"]:
                    noOfRickshaws += 1
                    flag = 1


                if flag == 1:
                    # Draw bounding box on original frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 4)
                    label = f"{vehicle_class} {row['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,255), 4)

        # Update total vehicle count
        total_vehicles = noOfCars + noOfBikes + noOfBuses + noOfTrucks + noOfRickshaws

        # Compute green light timing
        greenTime = math.ceil((noOfCars * self.carTime) + (noOfRickshaws * self.rickshawTime) + 
                            (noOfBuses * self.busTime) + (noOfTrucks * self.truckTime) + (noOfBikes * self.bikeTime))
        if (greenTime < self.defaultMinimum):
            greenTime = self.defaultMinimum
        elif (greenTime > self.defaultMaximum):
            greenTime = self.defaultMaximum

        # Update UI
        return frame,total_vehicles, greenTime, noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws




    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.result_button.configure(fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.about_us_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "frame_2":
            self.second_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()
        if name == "frame_3":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def frame_2_button_event(self):
        self.select_frame_by_name("frame_2")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def convert_cv_to_tk(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        img = img.resize((400, 300))  # Resize to fit UI
        return customtkinter.CTkImage(light_image=img, size=(400, 300))
     
    def display(self,video_paths):
    # Clear previous content (if necessary)
        for widget in self.second_frame.winfo_children():
            widget.destroy()

        num_images = len(self.results)
        
        # Configure columns dynamically for centering
        for i in range(num_images + 2):  # Adding extra columns for centering
            self.second_frame.columnconfigure(i, weight=1)

        for i, (frame, total_vehicles, greenTime, noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws) in enumerate(self.results):
            col_position = i + 1  # Shift position by 1 to leave space on the left for centering
            # greenTime *= 2
            count_text = f"Cars: {noOfCars}\nBikes: {noOfBikes}\nBuses: {noOfBuses}\nTrucks: {noOfTrucks}\nRickshaws: {noOfRickshaws}"
            count_label = customtkinter.CTkLabel(self.second_frame, text=count_text, font=("Arial", 12))
            count_label.grid(row=1, column=col_position)
            
            cap = cv2.VideoCapture(video_paths[i])
            frame_count = 0
            while greenTime >= 0:
                ret, frame = cap.read()
                frame_count += 1

                if frame_count >= 15:
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                        frame = cv2.resize(frame, (640, 360))  # Resize to fit Canvas
                        pil_image = Image.fromarray(frame)
                        img = customtkinter.CTkImage(light_image=pil_image, size=(640, 360))  

                        second_frame_image = customtkinter.CTkLabel(self.second_frame, image=img, text="")
                        # second_frame_image.image = img
                        second_frame_image.grid(row=0, column=col_position, padx=10, pady=10)
                        green_time_label = customtkinter.CTkLabel(self.second_frame, text=f"Green Light Time: {greenTime} sec", font=("Arial", 12, "bold"))
                        green_time_label.grid(row=2, column=col_position)
                        app.update()  # Force UI update
                        time.sleep(1)
                        greenTime -= 1
                    else:
                        cap.release()



    def show_about_us(self):
        # Clear previous content
        for widget in self.third_frame.winfo_children():
            widget.destroy()

        bg_color = "#2E2E2E"  # Dark gray (same as the text box background)

        # Scrollable frame for long content
        canvas = tk.Canvas(self.third_frame, bg=bg_color, highlightthickness=0)  # Set canvas background
        scrollbar = tk.Scrollbar(self.third_frame, orient="vertical", command=canvas.yview)
        # Scrollable frame for long content
        scrollable_frame = ctk.CTkFrame(canvas)

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add the content to the scrollable frame
        title_label = ctk.CTkLabel(scrollable_frame, text="🚦 AI-Powered Traffic Signal Simulation", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # Problem Statement
        problem_label = ctk.CTkLabel(scrollable_frame, text="Problem Statement", font=("Arial", 14, "bold"))
        problem_label.pack(pady=5)
        problem_text = tk.Text(scrollable_frame, wrap="word", height=5, width=80, bg="#2E2E2E", fg="white", font=("Arial", 12))
        problem_text.insert("1.0", "In urban environments, inefficient traffic signals contribute to congestion, increased fuel consumption, and pollution. This project develops an AI-driven approach to optimize traffic light durations based on real-time vehicle detection.")
        problem_text.config(state="disabled")
        problem_text.pack(pady=5)

        # Approach
        approach_label = ctk.CTkLabel(scrollable_frame, text="Approach to Solve the Problem", font=("Arial", 14, "bold"))
        approach_label.pack(pady=5)
        approach_text = tk.Text(scrollable_frame, wrap="word", height=5, width=80, bg="#2E2E2E", fg="white", font=("Arial", 12))
        approach_text.insert("1.0", "The system uses YOLOv8 for vehicle detection and an LSTM model for traffic prediction. The AI dynamically adjusts signal timings based on real-time traffic density and predicted congestion levels.")
        approach_text.config(state="disabled")
        approach_text.pack(pady=5)

        # How It Works
        how_it_works_label = ctk.CTkLabel(scrollable_frame, text="How It Works", font=("Arial", 14, "bold"))
        how_it_works_label.pack(pady=5)

        steps = [
            "1. Vehicle Detection: Uses YOLOv8 to detect vehicles from live video feeds.",
            "2. Data Collection: Analyzes vehicle count and type for congestion monitoring.",
            "3. Dynamic Signal Adjustment: AI calculates optimal green light duration.",
            "4. Continuous Learning: LSTM model predicts future congestion."
        ]

        for step in steps:
            step_label = ctk.CTkLabel(scrollable_frame, text=step, font=("Arial", 12), wraplength=600)
            step_label.pack(anchor="w", padx=20)

        # Video Section (Placeholder - Tkinter does not support video directly)

        # Pack Canvas and Scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # show_about_us()

if __name__ == "__main__":
    app = App()
    app.show_about_us()
    app.mainloop()