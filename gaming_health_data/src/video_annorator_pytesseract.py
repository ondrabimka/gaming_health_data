# %%
import cv2
import pytesseract
import numpy as np
import re
import os
import pandas as pd

# %%
class VideoAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        # ROI: Adjusted for OW2 health bottom-left corner
        self.health_roi = {
            'x1': 140,  # position of the health bar in the x-axis
            'y1': 595,  # position of the health bar in the y-axis
            'x2': 220,  # width of the health bar in the x-axis
            'y2': 615   # height of the health bar in the y-axis
        }

    def parse_health_string(self, text):
        """
        Parse OCR output string to extract health values.
        Handles cases like:
        - '1221250' -> '122/250'
        - '122250' -> '122/250'
        - '2501250' -> '250/250' (repeated max health case)
        """
        try:
            # Remove any non-digit characters
            digits = ''.join(filter(str.isdigit, text))
            
            if len(digits) >= 4:
                # Always take last 3 digits as max_health
                max_health = int(digits[-3:])
                
                # Check if the digits contain repeated max_health value
                if digits.startswith(str(max_health)):
                    current_health = max_health
                # Regular cases
                elif len(digits) >= 6:
                    current_health = int(digits[-6:-3])
                else:
                    current_health = int(digits[:-3])

                print(f"DEBUG: digits={digits}, current={current_health}, max={max_health}")
                
                # Validate the values
                if 0 <= current_health <= max_health and max_health <= 999:
                    return current_health, max_health
                    
            return None, None
        except Exception as e:
            print(f"Error parsing health string: {e}")
            return None, None

    def extract_health_from_roi(self, roi):
        """Extract health values from ROI using OCR and parse_health_string."""
        try:
            # Basic preprocessing for better OCR
            roi = cv2.resize(roi, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            contrast = clahe.apply(gray)
            _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Add padding for better OCR
            padded = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            
            # Single OCR pass with optimal config
            custom_config = '--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/'
            text = pytesseract.image_to_string(padded, config=custom_config).strip()
            # print(f"OCR Result: '{text}'")
            
            # Use parse_health_string to extract values
            return self.parse_health_string(text)
                
        except Exception as e:
            print(f"Error processing ROI: {e}")
            return None, None

    def process_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number}")

        roi = frame[
            self.health_roi['y1']:self.health_roi['y2'],
            self.health_roi['x1']:self.health_roi['x2']
        ]
        current_health, max_health = self.extract_health_from_roi(roi)
        return current_health, max_health

    def show_processed_roi(self, frame_number):
        """Display the preprocessed ROI to debug OCR."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number}")
        
        roi = frame[
            self.health_roi['y1']:self.health_roi['y2'],
            self.health_roi['x1']:self.health_roi['x2']
        ]
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

        cv2.imshow("Processed ROI", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_roi(self, frame_number=0):
        """Show the ROI on the original frame for visual debugging."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Could not read frame")
            return
        roi = frame[
            self.health_roi['y1']:self.health_roi['y2'],
            self.health_roi['x1']:self.health_roi['x2']
        ]
        cv2.imshow('ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def debug_ocr_pipeline(self, frame_number=0):
        """Show and save each step of the OCR pipeline for debugging."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Could not read frame")
            return

        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)

        # Extract ROI and resize
        roi = frame[
            self.health_roi['y1']:self.health_roi['y2'],
            self.health_roi['x1']:self.health_roi['x2']
        ]
        roi_resized = cv2.resize(roi, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{debug_dir}/01_original_roi.png", roi_resized)

        # Convert to grayscale
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{debug_dir}/02_grayscale.png", gray)

        # Enhanced CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        contrast = clahe.apply(gray)
        cv2.imwrite(f"{debug_dir}/03_clahe.png", contrast)

        # Gaussian blur + Otsu
        blurred = cv2.GaussianBlur(contrast, (5,5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(f"{debug_dir}/04_otsu.png", thresh)

        # Enhanced morphological operations
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(f"{debug_dir}/05_cleaned.png", cleaned)

        # Final padded version
        padded = cv2.copyMakeBorder(cleaned, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(f"{debug_dir}/06_final.png", padded)

        # Try OCR with multiple configs
        configs = [
            '--oem 1 --psm 7',
            '--oem 1 --psm 8',
            '--oem 1 --psm 6'
        ]

        print("\nOCR Results:")
        for config in configs:
            custom_config = f'{config} -c tessedit_char_whitelist=0123456789/'
            text = pytesseract.image_to_string(padded, config=custom_config)
            print(f"Config '{config}': '{text.strip()}'")
            print("Actual out:", self.parse_health_string(text))

        print("\nDebug images saved to:", debug_dir)


    def release(self):
        self.cap.release()

    def save_processed_roi(self, frame_number, filename="debug_roi.png"):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            print("Could not read frame")
            return
        roi = frame[
            self.health_roi['y1']:self.health_roi['y2'],
            self.health_roi['x1']:self.health_roi['x2']
        ]
        roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        cv2.imwrite(filename, thresh)
        print(f"Saved processed ROI to {filename}")

    
    def analyze_video(self, start_frame=0, end_frame=None, frame_skip=1):
        """
        Analyze health values across video frames and return a DataFrame.
        
        Parameters:
        -----------
        start_frame : int, optional (default=0)
            First frame to analyze
        end_frame : int, optional (default=None)
            Last frame to analyze. If None, processes until end of video
        frame_skip : int, optional (default=30)
            Number of frames to skip between readings
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: time, current_health, max_health
        """
        try:
            # Get video properties
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Validate end_frame
            if end_frame is None or end_frame > total_frames:
                end_frame = total_frames
                
            # Prepare data storage
            data = []
            
            # Process frames
            for frame_num in range(start_frame, end_frame, frame_skip):
                # Get health values
                current_health, max_health = self.process_frame(frame_num)
                
                # Calculate time in seconds
                time_sec = frame_num / fps
                
                # Store valid readings
                # if current_health is not None and max_health is not None:
                data.append({
                    'time': time_sec,
                    'current_health': current_health,
                    'max_health': max_health
                })
                    
                # Optional progress indicator
                if frame_num % 300 == 0:  # Show progress every 300 frames
                    percent = (frame_num - start_frame) / (end_frame - start_frame) * 100
                    print(f"Progress: {percent:.1f}%")
                    
            # Create DataFrame
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def show_frame_number(self, frame_number):
        """Display a specific frame number for debugging."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            print(f"Could not read frame {frame_number}")
            return
        cv2.imshow(f"Frame {frame_number}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# %%

# video_path = "gaming_health_data/recorded_data/VIDEO/VIDEOS/Overwatch 2_20250508132346.mp4"
# annotator = VideoAnnotator(video_path)
# # Visualize ROI on a specific frame
# annotator.visualize_roi(frame_number=14000)
# # Step through the OCR pipeline for a specific frame
# result = annotator.process_frame(frame_number=14000)
# print(f"Health at frame 14000: {result}")
# annotator.release()
# %%

# Analyze whole video
video_path = "gaming_health_data/recorded_data/VIDEO/VIDEOS/Overwatch 2_20250508132346.mp4"
annotator = VideoAnnotator(video_path)
# annotator.show_frame_number(frame_number=20082)  # Show frame 3600 for debugging
# annotator.visualize_roi(frame_number=20082)
# annotator.debug_ocr_pipeline(frame_number=20082)


# Analyze (assuming 30 fps)
df = annotator.analyze_video(
    start_frame=19000,
    end_frame=25000,  # 5 minutes
    
)


# Print results
# print("\nFirst few readings:")
# print(df.head())

# %%
df
# %%
