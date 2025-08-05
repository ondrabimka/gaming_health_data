# %%
import cv2
import easyocr
import numpy as np
import re
import os
import pandas as pd

class VideoAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
        
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
        """Extract health values from ROI using EasyOCR and parse_health_string."""
        try:
            # Preprocessing for better OCR
            roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            
            # EasyOCR works better with minimal preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Use EasyOCR to detect text
            results = self.reader.readtext(enhanced, allowlist='0123456789/')
            
            if results:
                # Find the result with highest confidence
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1]
                confidence = best_result[2]
                
                print(f"EasyOCR: '{text}' (confidence: {confidence:.2f})")
                
                if confidence > 0.3:  # Lower threshold since EasyOCR is more conservative
                    return self.parse_health_string(text)
                
            return None, None
            # # If no results with allowlist, try without restrictions
            # results_unrestricted = self.reader.readtext(enhanced)
            # if results_unrestricted:
            #     best_result = max(results_unrestricted, key=lambda x: x[2])
            #     text = best_result[1]
            #     confidence = best_result[2]
            #     
            #     print(f"EasyOCR (unrestricted): '{text}' (confidence: {confidence:.2f})")
            #     
            #     if confidence > 0.3:
            #         return self.parse_health_string(text)
                    
        except Exception as e:
            print(f"Error processing ROI with EasyOCR: {e}")
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
        """Display the preprocessed ROI to debug EasyOCR."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number}")
        
        roi = frame[
            self.health_roi['y1']:self.health_roi['y2'],
            self.health_roi['x1']:self.health_roi['x2']
        ]
        roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        cv2.imshow("Processed ROI for EasyOCR", enhanced)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Test EasyOCR on this processed ROI
        results = self.reader.readtext(enhanced, allowlist='0123456789/')
        print("EasyOCR results:")
        for result in results:
            text, confidence = result[1], result[2]
            print(f"  Text: '{text}', Confidence: {confidence:.2f}")
            print(f"  Parsed: {self.parse_health_string(text)}")

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
        """Show and save each step of the EasyOCR pipeline for debugging."""
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
        roi_resized = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{debug_dir}/01_original_roi.png", roi_resized)

        # Convert to grayscale
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{debug_dir}/02_grayscale.png", gray)

        # Enhanced CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        cv2.imwrite(f"{debug_dir}/03_enhanced.png", enhanced)

        # Try EasyOCR with different preprocessing
        print("\nEasyOCR Results:")
        
        # Method 1: Direct on enhanced image
        results1 = self.reader.readtext(enhanced, allowlist='0123456789/')
        print("Method 1 (enhanced, restricted):")
        for result in results1:
            text, confidence = result[1], result[2]
            print(f"  Text: '{text}', Confidence: {confidence:.2f}")
            print(f"  Parsed: {self.parse_health_string(text)}")
        
        # Method 2: Without character restrictions
        results2 = self.reader.readtext(enhanced)
        print("\nMethod 2 (enhanced, unrestricted):")
        for result in results2:
            text, confidence = result[1], result[2]
            print(f"  Text: '{text}', Confidence: {confidence:.2f}")
            print(f"  Parsed: {self.parse_health_string(text)}")
        
        # Method 3: Apply threshold and try again
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(f"{debug_dir}/04_threshold.png", thresh)
        
        results3 = self.reader.readtext(thresh, allowlist='0123456789/')
        print("\nMethod 3 (threshold, restricted):")
        for result in results3:
            text, confidence = result[1], result[2]
            print(f"  Text: '{text}', Confidence: {confidence:.2f}")
            print(f"  Parsed: {self.parse_health_string(text)}")

        print(f"\nDebug images saved to: {debug_dir}")
        print("EasyOCR is generally more accurate than Tesseract for gaming text!")


    def release(self):
        self.cap.release()

    def save_processed_roi(self, frame_number, filename="debug_roi.png"):
        """Save preprocessed ROI for EasyOCR analysis."""
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
        
        # Apply CLAHE for better contrast (EasyOCR works well with this)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        cv2.imwrite(filename, enhanced)
        print(f"Saved processed ROI to {filename}")
        
        # Test EasyOCR on saved image
        results = self.reader.readtext(enhanced, allowlist='0123456789/')
        print("EasyOCR results on saved image:")
        for result in results:
            text, confidence = result[1], result[2]
            print(f"  Text: '{text}', Confidence: {confidence:.2f}")
            print(f"  Parsed: {self.parse_health_string(text)}")

    
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
        # try:
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
        print("This is data: ", data)
        df = pd.DataFrame(data)
        return df
            
        # except Exception as e:
        #     print(f"Error analyzing video: {e}")
        #     return pd.DataFrame()  # Return empty DataFrame on error
    
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

# Example usage with EasyOCR
# video_path = "gaming_health_data/recorded_data/VIDEO/VIDEOS/Overwatch 2_20250508132346.mp4"
# annotator = VideoAnnotator(video_path)
# # Visualize ROI on a specific frame
# annotator.visualize_roi(frame_number=14000)
# # Step through the EasyOCR pipeline for a specific frame
# result = annotator.process_frame(frame_number=14000)
# print(f"Health at frame 14000: {result}")
# annotator.release()
# %%

# Analyze whole video with EasyOCR
video_path = "gaming_health_data/recorded_data/VIDEO/VIDEOS/Overwatch 2_20250508132346.mp4"
annotator = VideoAnnotator(video_path)
# annotator.show_frame_number(frame_number=20082)  # Show frame for debugging
# annotator.visualize_roi(frame_number=20082)
# annotator.debug_ocr_pipeline(frame_number=20082)  # Test EasyOCR pipeline


# Analyze video frames (assuming 30 fps)
df = annotator.analyze_video(
    start_frame=19000,
    end_frame=20000 # 25000,  # Analyze 6000 frames
    
)

# Print results
# print("\nFirst few readings:")
# print(df.head())

# %%
df
