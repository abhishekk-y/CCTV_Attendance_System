import tkinter as tk
from tkinter import ttk, font, simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
import face_recognition
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import os
import csv

class AttendanceSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Attendance System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f5f5f5")
        
        # Load face encodings
        try:
            with open("encodings.pickle", "rb") as file:
                data = pickle.load(file)
            self.known_face_encodings = np.array(data["encodings"], dtype=np.float64)
            self.known_face_names = np.array(data["names"])
        except:
            self.known_face_encodings = np.array([])
            self.known_face_names = np.array([])
        
        # Initialize camera with optimized settings
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Attendance tracking
        self.attendance_file = "attendance1.csv"
        self.attendance_log = set()
        self.recent_attendances = []
        
        # Initialize CSV
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=["Name", "Date", "Time", "Confidence"])
            df.to_csv(self.attendance_file, index=False)
        else:
            # Load existing attendance for today
            df = pd.read_csv(self.attendance_file)
            today = datetime.now().strftime("%Y-%m-%d")
            today_df = df[df["Date"] == today]
            for _, row in today_df.iterrows():
                self.attendance_log.add((row["Name"], row["Date"]))
                self.recent_attendances.append({
                    "name": row["Name"],
                    "time": row["Time"],
                    "date": row["Date"],
                    "confidence": row.get("Confidence", "95.0")
                })
        
        # Statistics
        self.total_registered = len(self.known_face_names)
        self.total_present = len(self.attendance_log)
        self.fps = 0
        self.faces_detected = 0
        
        # GUI State
        self.is_running = True
        self.current_frame = None
        
        # Performance optimization - Balanced speed & accuracy
        self.process_every_n_frames = 4  # Process every 4th frame for balanced performance
        self.frame_counter = 0
        self.last_face_locations = []
        self.last_face_names = []
        self.skip_frames = 0  # Skip frames for smoother display
        
        # Build GUI
        self.create_gui()
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.process_camera, daemon=True)
        self.camera_thread.start()
        
        # Update GUI
        self.update_gui()
        
    def create_gui(self):
        # Top Statistics Panel
        stats_frame = tk.Frame(self.root, bg="#f5f5f5")
        stats_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Statistics Cards
        self.create_stat_card(stats_frame, "Total Present", "0", "80% attendance rate", "#4caf50", "✓", 0)
        self.create_stat_card(stats_frame, "Total Absent", "0", "20% absent rate", "#f44336", "✕", 1)
        self.create_stat_card(stats_frame, "Registered Users", str(self.total_registered), "Active in database", "#2196f3", "👥", 2)
        self.create_stat_card(stats_frame, "Active Feeds", "1", "Live camera feeds", "#9c27b0", "📹", 3)
        
        # Main Content Frame
        content_frame = tk.Frame(self.root, bg="#f5f5f5")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Left Panel - Camera Feed
        left_frame = tk.Frame(content_frame, bg="white", relief=tk.RAISED, borderwidth=1)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera Header
        cam_header = tk.Frame(left_frame, bg="white")
        cam_header.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(cam_header, text="Live Camera Feed", font=("Segoe UI", 14, "bold"), 
                bg="white", fg="#333").pack(side=tk.LEFT)
        tk.Label(cam_header, text="Real-time face detection and recognition", 
                font=("Segoe UI", 9), bg="white", fg="#999").pack(side=tk.LEFT, padx=10)
        
        # Add Face Button
        add_face_btn = tk.Button(cam_header, text="➕ Add New Face", 
                                font=("Segoe UI", 10, "bold"), bg="#2196f3", fg="white",
                                padx=15, pady=5, cursor="hand2", relief=tk.FLAT,
                                command=self.open_add_face_dialog)
        add_face_btn.pack(side=tk.RIGHT, padx=5)
        
        # Reset Attendance Button
        reset_btn = tk.Button(cam_header, text="🗑️ Reset Attendance", 
                             font=("Segoe UI", 10, "bold"), bg="#f44336", fg="white",
                             padx=15, pady=5, cursor="hand2", relief=tk.FLAT,
                             command=self.reset_attendance)
        reset_btn.pack(side=tk.RIGHT, padx=5)
        
        # Live Badge
        live_badge = tk.Frame(cam_header, bg="#4caf50", padx=10, pady=2)
        live_badge.pack(side=tk.RIGHT)
        tk.Label(live_badge, text="● LIVE", font=("Segoe UI", 9, "bold"), 
                bg="#4caf50", fg="white").pack()
        
        # Camera Canvas
        cam_container = tk.Frame(left_frame, bg="#1a1a2e")
        cam_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        self.camera_canvas = tk.Canvas(cam_container, bg="#1a1a2e", highlightthickness=0)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Camera Status Bar
        status_bar = tk.Frame(left_frame, bg="#0f0f1e", height=40)
        status_bar.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        self.rec_label = tk.Label(status_bar, text="● REC", font=("Segoe UI", 9), 
                                  bg="#0f0f1e", fg="#f44336")
        self.rec_label.pack(side=tk.LEFT, padx=10)
        
        self.time_label = tk.Label(status_bar, text="00:00:00", font=("Segoe UI", 9), 
                                   bg="#0f0f1e", fg="white")
        self.time_label.pack(side=tk.LEFT, padx=10)
        
        tk.Label(status_bar, text="CAM-01", font=("Segoe UI", 9), 
                bg="#0f0f1e", fg="white").pack(side=tk.RIGHT, padx=10)
        
        self.fps_label = tk.Label(status_bar, text="FPS: 30", font=("Segoe UI", 9), 
                                 bg="#0f0f1e", fg="#4caf50")
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        self.faces_label = tk.Label(status_bar, text="Faces: 0", font=("Segoe UI", 9), 
                                   bg="#0f0f1e", fg="#2196f3")
        self.faces_label.pack(side=tk.RIGHT, padx=10)
        
        tk.Label(status_bar, text="🤖 AI Modes Active", font=("Segoe UI", 9), 
                bg="#0f0f1e", fg="#00bcd4").pack(side=tk.RIGHT, padx=10)
        
        # Right Panel - Recent Attendance
        right_frame = tk.Frame(content_frame, bg="white", relief=tk.RAISED, borderwidth=1, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Attendance Header
        att_header = tk.Frame(right_frame, bg="white")
        att_header.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(att_header, text="Recent Attendance", font=("Segoe UI", 14, "bold"), 
                bg="white", fg="#333").pack(anchor=tk.W)
        tk.Label(att_header, text="Latest face recognition logs", 
                font=("Segoe UI", 9), bg="white", fg="#999").pack(anchor=tk.W)
        
        # Scrollable Attendance List
        att_container = tk.Frame(right_frame, bg="white")
        att_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.att_canvas = tk.Canvas(att_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(att_container, orient="vertical", command=self.att_canvas.yview)
        self.att_scrollable_frame = tk.Frame(self.att_canvas, bg="white")
        
        self.att_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.att_canvas.configure(scrollregion=self.att_canvas.bbox("all"))
        )
        
        self.att_canvas.create_window((0, 0), window=self.att_scrollable_frame, anchor="nw")
        self.att_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.att_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_stat_card(self, parent, title, value, subtitle, color, icon, col):
        card = tk.Frame(parent, bg="white", relief=tk.RAISED, borderwidth=1)
        card.grid(row=0, column=col, padx=10, sticky="ew")
        parent.columnconfigure(col, weight=1)
        
        # Header with icon
        header = tk.Frame(card, bg="white")
        header.pack(fill=tk.X, padx=15, pady=(15, 5))
        
        tk.Label(header, text=title, font=("Segoe UI", 10), 
                bg="white", fg="#666").pack(side=tk.LEFT)
        
        icon_label = tk.Label(header, text=icon, font=("Segoe UI", 16), bg="white", fg=color)
        icon_label.pack(side=tk.RIGHT)
        
        # Value
        value_label = tk.Label(card, text=value, font=("Segoe UI", 24, "bold"), 
                              bg="white", fg="#333")
        value_label.pack(anchor=tk.W, padx=15)
        
        if col == 0:
            self.present_label = value_label
        elif col == 1:
            self.absent_label = value_label
        
        # Subtitle
        tk.Label(card, text=subtitle, font=("Segoe UI", 9), 
                bg="white", fg="#999").pack(anchor=tk.W, padx=15, pady=(0, 15))
    
    def add_attendance_entry(self, name, time, date, confidence):
        entry_frame = tk.Frame(self.att_scrollable_frame, bg="#f8f9fa", 
                              relief=tk.RAISED, borderwidth=1)
        entry_frame.pack(fill=tk.X, pady=5)
        
        # Left side - Avatar and Info
        left = tk.Frame(entry_frame, bg="#f8f9fa")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Avatar
        avatar = tk.Label(left, text="👤", font=("Segoe UI", 24), 
                         bg="#e3f2fd", fg="#2196f3", width=2)
        avatar.pack(side=tk.LEFT, padx=(0, 10))
        
        # Info
        info = tk.Frame(left, bg="#f8f9fa")
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(info, text=name, font=("Segoe UI", 11, "bold"), 
                bg="#f8f9fa", fg="#333").pack(anchor=tk.W)
        tk.Label(info, text=f"⏰ {time}", font=("Segoe UI", 9), 
                bg="#f8f9fa", fg="#666").pack(anchor=tk.W)
        tk.Label(info, text=f"{confidence}% confidence", font=("Segoe UI", 8), 
                bg="#f8f9fa", fg="#4caf50").pack(anchor=tk.W)
        
        # Right side - Badge
        badge = tk.Frame(entry_frame, bg="#000", padx=12, pady=4)
        badge.pack(side=tk.RIGHT, padx=15)
        
        tk.Label(badge, text="present", font=("Segoe UI", 9), 
                bg="#000", fg="white").pack()
    
    def process_camera(self):
        frame_count = 0
        start_time = datetime.now()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                self.fps = int(frame_count / elapsed) if elapsed > 0 else 0
            
            # Only process every Nth frame for face recognition (reduces CPU load)
            self.frame_counter += 1
            process_this_frame = (self.frame_counter % self.process_every_n_frames == 0)
            
            if process_this_frame:
                # Resize to 1/3 size for balanced speed and accuracy
                small_frame = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33, interpolation=cv2.INTER_LINEAR)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces with HOG model with 1 upsample for better detection
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog", number_of_times_to_upsample=1)
                
                self.faces_detected = len(face_locations)
                
                if len(face_locations) > 0:
                    # Only encode faces if we found any
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    names = []
                    # Process each face
                    for face_encoding in face_encodings:
                        if self.known_face_encodings.size == 0:
                            names.append(("Unknown", 0))
                            continue

                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                        name = "Unknown"
                        confidence = 0

                        if True in matches:
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            
                            if matches[best_match_index] and face_distances[best_match_index] < 0.55:
                                name = self.known_face_names[best_match_index]
                                confidence = round((1 - face_distances[best_match_index]) * 100, 1)
                                
                                # Mark attendance in separate thread to avoid blocking
                                if name != "Unknown":
                                    threading.Thread(target=self.mark_attendance, args=(name, confidence), daemon=True).start()

                        names.append((name, confidence))
                    
                    # Cache results
                    self.last_face_locations = face_locations
                    self.last_face_names = names
                else:
                    # No faces found, clear cache
                    self.last_face_locations = []
                    self.last_face_names = []
            
            # Draw boxes using cached locations (draw every frame for smooth display)
            for face_location, (name, confidence) in zip(self.last_face_locations, self.last_face_names):
                # Scale back up coordinates (3x since we used 0.33)
                top, right, bottom, left = face_location
                top = int(top * 3.03)
                right = int(right * 3.03)
                bottom = int(bottom * 3.03)
                left = int(left * 3.03)
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Name label with background
                label = f"{name} {confidence}%" if name != "Unknown" else "Unknown"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (left, top - 30), (left + label_size[0] + 10, top), color, -1)
                cv2.putText(frame, label, (left + 5, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self.current_frame = frame
    
    def mark_attendance(self, name, confidence):
        now = datetime.now()
        date_today = now.strftime("%Y-%m-%d")
        time_now = now.strftime("%H:%M:%S")
        
        if (name, date_today) not in self.attendance_log:
            self.attendance_log.add((name, date_today))
            
            # Save to CSV
            df = pd.DataFrame([[name, date_today, time_now, confidence]], 
                            columns=["Name", "Date", "Time", "Confidence"])
            df.to_csv(self.attendance_file, mode="a", header=False, index=False)
            
            # Add to recent list
            self.recent_attendances.insert(0, {
                "name": name,
                "time": time_now,
                "date": date_today,
                "confidence": confidence
            })
            
            # Keep only last 20
            if len(self.recent_attendances) > 20:
                self.recent_attendances = self.recent_attendances[:20]
            
            # Update stats
            self.total_present = len(self.attendance_log)
            
            print(f"✅ Attendance marked for {name} at {time_now}")
    
    def update_gui(self):
        if not self.is_running:
            return
        
        # Update camera feed (skip some frames for speed)
        self.skip_frames += 1
        if self.current_frame is not None and self.skip_frames % 2 == 0:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize to fit canvas
            canvas_width = self.camera_canvas.winfo_width()
            canvas_height = self.camera_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Use faster BILINEAR instead of LANCZOS
                img = img.resize((canvas_width, canvas_height), Image.Resampling.BILINEAR)
            
            photo = ImageTk.PhotoImage(image=img)
            self.camera_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.camera_canvas.image = photo
        
        # Update status bar
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=current_time)
        self.fps_label.config(text=f"FPS: {self.fps}")
        self.faces_label.config(text=f"Faces: {self.faces_detected}")
        
        # Update statistics
        self.present_label.config(text=str(self.total_present))
        absent = self.total_registered - self.total_present
        self.absent_label.config(text=str(absent))
        
        # Update attendance list (only if changed)
        if hasattr(self, '_last_attendance_count'):
            if self._last_attendance_count != len(self.recent_attendances):
                self.update_attendance_list()
        else:
            self.update_attendance_list()
        
        self._last_attendance_count = len(self.recent_attendances)
        
        # Schedule next update (reduced to 20 FPS for maximum performance)
        self.root.after(50, self.update_gui)  # 20 FPS
    
    def update_attendance_list(self):
        # Clear existing entries
        for widget in self.att_scrollable_frame.winfo_children():
            widget.destroy()
        
        # Add new entries
        for entry in self.recent_attendances:
            self.add_attendance_entry(
                entry["name"],
                entry["time"],
                entry["date"],
                entry["confidence"]
            )
    
    def open_add_face_dialog(self):
        """Open dialog to capture new face and add to database"""
        # Create new window
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Face")
        dialog.geometry("720x780")
        dialog.configure(bg="#f5f5f5")
        dialog.resizable(False, False)
        
        # Variables
        capture_cap = None
        try:
            capture_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            # Set lower resolution for faster preview
            capture_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture_cap.set(cv2.CAP_PROP_FPS, 30)
            capture_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture_cap.isOpened():
                messagebox.showerror("Camera Error", "❌ Could not open camera!")
                dialog.destroy()
                return
        except Exception as e:
            messagebox.showerror("Camera Error", f"❌ Error: {str(e)}")
            dialog.destroy()
            return
        
        captured_frame = [None]
        is_capturing = [True]
        frame_skip = [0]  # Skip frames for performance
        
        # Header
        header = tk.Frame(dialog, bg="#2196f3", height=60)
        header.pack(fill=tk.X)
        tk.Label(header, text="📸 Capture New Face", font=("Segoe UI", 16, "bold"),
                bg="#2196f3", fg="white").pack(pady=15)
        
        # Camera frame
        cam_frame = tk.Frame(dialog, bg="#1a1a2e", relief=tk.SUNKEN, borderwidth=2)
        cam_frame.pack(fill=tk.X, padx=20, pady=20)
        
        canvas = tk.Canvas(cam_frame, bg="#1a1a2e", width=640, height=480, highlightthickness=0)
        canvas.pack(padx=5, pady=5)
        
        # Instructions
        instruction_frame = tk.Frame(dialog, bg="#f5f5f5")
        instruction_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        tk.Label(instruction_frame, text="📋 Position your face in the camera and click Capture",
                font=("Segoe UI", 10), bg="#f5f5f5", fg="#666").pack()
        
        # Name input and buttons
        control_frame = tk.Frame(dialog, bg="#f5f5f5")
        control_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Name input
        name_frame = tk.Frame(control_frame, bg="#f5f5f5")
        name_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(name_frame, text="Name:", font=("Segoe UI", 10), 
                bg="#f5f5f5", fg="#333").pack(side=tk.LEFT, padx=(0, 10))
        name_entry = tk.Entry(name_frame, font=("Segoe UI", 12), width=25)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Buttons
        btn_frame = tk.Frame(control_frame, bg="#f5f5f5")
        btn_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        def update_capture_feed():
            if not is_capturing[0]:
                return
            
            # Skip every other frame for performance
            frame_skip[0] += 1
            if frame_skip[0] % 2 != 0:
                dialog.after(50, update_capture_feed)  # 20 FPS
                return
            
            ret, frame = capture_cap.read()
            if ret:
                # Draw guide box
                height, width = frame.shape[:2]
                box_size = 300
                x1 = (width - box_size) // 2
                y1 = (height - box_size) // 2
                x2 = x1 + box_size
                y2 = y1 + box_size
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Position face in box", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display - use faster method
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=img)
                canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                canvas.image = photo
            
            dialog.after(50, update_capture_feed)  # 20 FPS for smooth preview
        
        def capture_face():
            if capture_cap is None:
                messagebox.showerror("Error", "❌ Camera not available!")
                return
            
            ret, frame = capture_cap.read()
            if ret:
                # Check if face is detected - use faster detection
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                
                if len(face_locations) > 0:
                    captured_frame[0] = frame.copy()
                    canvas.configure(bg="#4caf50")
                    # Flash effect
                    dialog.after(200, lambda: canvas.configure(bg="#1a1a2e"))
                    messagebox.showinfo("Success", "✅ Face captured successfully!\n\nNow enter the name and click Save.")
                else:
                    messagebox.showwarning("No Face", "⚠️ No face detected in the frame!\n\nPlease position your face in the green box and try again.")
            else:
                messagebox.showerror("Error", "❌ Failed to capture frame!")
        
        def save_face():
            if captured_frame[0] is None:
                messagebox.showwarning("No Capture", "⚠️ Please capture a face first!\n\nClick the 'Capture' button to take a photo.")
                return
            
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("No Name", "⚠️ Please enter a name!\n\nType the person's name in the field above.")
                return
            
            # Check if name already exists
            if os.path.exists(f"known_faces/{name}.jpg"):
                response = messagebox.askyesno("Name Exists", 
                    f"⚠️ '{name}' already exists in the database!\n\nDo you want to replace it?")
                if not response:
                    return
            
            # Disable buttons during processing
            save_btn.config(state=tk.DISABLED, text="Processing...")
            capture_btn.config(state=tk.DISABLED)
            cancel_btn.config(state=tk.DISABLED)
            name_entry.config(state=tk.DISABLED)
            dialog.update()
            
            def process_save():
                try:
                    # Save image to known_faces folder
                    if not os.path.exists("known_faces"):
                        os.makedirs("known_faces")
                    
                    filename = f"known_faces/{name}.jpg"
                    cv2.imwrite(filename, captured_frame[0])
                    
                    # Re-encode all faces in separate thread
                    self.encode_all_faces()
                    
                    # Reload encodings in main app
                    try:
                        with open("encodings.pickle", "rb") as file:
                            data = pickle.load(file)
                        self.known_face_encodings = np.array(data["encodings"], dtype=np.float64)
                        self.known_face_names = np.array(data["names"])
                        self.total_registered = len(self.known_face_names)
                    except Exception as e:
                        print(f"Error reloading encodings: {e}")
                    
                    # Show success and close
                    dialog.after(0, lambda: messagebox.showinfo("Success", 
                        f"✅ {name} has been added successfully!\n\nTotal registered users: {self.total_registered}"))
                    dialog.after(100, lambda: finish_save())
                    
                except Exception as e:
                    dialog.after(0, lambda: messagebox.showerror("Error", 
                        f"❌ Failed to save face!\n\nError: {str(e)}"))
                    dialog.after(100, lambda: finish_save())
            
            def finish_save():
                is_capturing[0] = False
                if capture_cap is not None:
                    capture_cap.release()
                dialog.destroy()
            
            # Run encoding in background thread
            threading.Thread(target=process_save, daemon=True).start()
        
        def cancel():
            is_capturing[0] = False
            if capture_cap is not None:
                capture_cap.release()
            dialog.destroy()
        
        # Buttons
        capture_btn = tk.Button(btn_frame, text="📷 Capture", font=("Segoe UI", 10, "bold"),
                               bg="#ff9800", fg="white", padx=20, pady=8, cursor="hand2",
                               relief=tk.FLAT, command=capture_face)
        capture_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = tk.Button(btn_frame, text="💾 Save", font=("Segoe UI", 10, "bold"),
                            bg="#4caf50", fg="white", padx=20, pady=8, cursor="hand2",
                            relief=tk.FLAT, command=save_face)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(btn_frame, text="❌ Cancel", font=("Segoe UI", 10, "bold"),
                              bg="#f44336", fg="white", padx=20, pady=8, cursor="hand2",
                              relief=tk.FLAT, command=cancel)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Start camera feed
        update_capture_feed()
        
        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Handle window close
        dialog.protocol("WM_DELETE_WINDOW", cancel)
    
    def reset_attendance(self):
        """Reset all attendance records with confirmation"""
        # Confirmation dialog
        response = messagebox.askyesno(
            "Reset Attendance",
            "⚠️ WARNING: This will permanently delete ALL attendance records!\n\n"
            "Are you sure you want to continue?",
            icon='warning'
        )
        
        if not response:
            return
        
        # Second confirmation for safety
        response2 = messagebox.askyesno(
            "Confirm Reset",
            "⚠️ FINAL CONFIRMATION\n\n"
            "This action CANNOT be undone!\n\n"
            f"Current records: {len(self.attendance_log)} entries\n\n"
            "Delete all records?",
            icon='warning'
        )
        
        if not response2:
            return
        
        try:
            # Clear the CSV file (keep header)
            with open("attendance1.csv", "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Date", "Time", "Confidence"])
            
            # Clear internal data structures
            self.attendance_log.clear()
            self.recent_attendances.clear()
            self.total_present = 0
            
            # Update GUI
            self.update_attendance_list()
            
            messagebox.showinfo(
                "Success",
                "✅ All attendance records have been cleared!\n\n"
                "The system is ready for new attendance entries."
            )
            
            print("✅ Attendance records reset successfully")
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"❌ Failed to reset attendance!\n\nError: {str(e)}"
            )
            print(f"❌ Error resetting attendance: {e}")
    
    def encode_all_faces(self):
        """Re-encode all faces in known_faces folder - OPTIMIZED VERSION"""
        KNOWN_FACES_DIR = "known_faces"
        ENCODINGS_FILE = "encodings.pickle"
        
        known_face_encodings = []
        known_face_names = []
        
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                
                # Load image with OpenCV (faster than face_recognition)
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Resize for faster processing if image is too large
                height, width = img.shape[:2]
                if width > 800:
                    scale = 800 / width
                    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                
                # Convert BGR to RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get encoding with optimized settings (num_jitters=1 for 3x speed boost)
                encoding = face_recognition.face_encodings(rgb_img, num_jitters=1)
                
                if len(encoding) > 0:
                    known_face_encodings.append(encoding[0])
                    known_face_names.append(os.path.splitext(filename)[0])
                    print(f"✅ Encoded {filename}")
                else:
                    print(f"⚠️ No face found in {filename}")
        
        # Save encodings to file
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        
        print(f"✅ Encoding complete. Saved {len(known_face_names)} faces to {ENCODINGS_FILE}")
    
    def on_closing(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = AttendanceSystemGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
