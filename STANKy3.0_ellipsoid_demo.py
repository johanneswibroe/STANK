from pypylon import pylon
import cv2
import time
import pygame
import csv
import matplotlib.pyplot as plt
import cProfile
import pstats
import subprocess
import io
import numpy as np

# Import OpenCV's OpenCL module
import cv2.ocl

def prof_to_csv(prof: cProfile.Profile):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = 'ncalls' + result.split('ncalls')[-1]
    lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
    return '\n'.join(lines)

fly_name = "UAS_RDL_RNAi_w1118(cs)_SD_15"

rawname = fly_name + "_raw.mp4"
moviename = fly_name + ".mp4"
csv_name = fly_name + ".csv"
plotname = fly_name + ".pdf"

# connecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

camera.Width = 200
camera.OffsetX.SetValue(628)

# Enable OpenCL for OpenCV
#cv2.ocl.setUseOpenCL(True)

#binning_factor = 1  # Adjust this value as needed

# # Enable and set binning
#camera.BinningHorizontal.SetValue(binning_factor)
#camera.BinningVertical.SetValue(binning_factor)

camera.ExposureTime.SetValue(30000)
# Set the desired frame rate
frame_rate = 30

time_list = []
velocity_list = []

# Set the duration of the video directly IN SECONDS
video_duration = 30

camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(frame_rate)

# initialize background subtractor and bounding box variables
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=500)

bbox = None

# initialize variables for calculating velocity
prev_center = None
prev_time = time.time()
velocity = None

writer = cv2.VideoWriter(moviename, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (200, 1080))

raw_writer = cv2.VideoWriter(rawname, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (200, 1080))

pygame.init()
pygame.mixer.music.load("/home/joeh/Auditory_filtering_experiment/long/mating_call_normal.wav")

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

start_time = time.time()
elapsed_time = 0

# get some variables going
frame_count = 0

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

counter = 0

sound_time = []

last_motion = 0
#pygame.mixer.music.play()

sound_played = False

# Create a profiler instance
profiler = cProfile.Profile()

# Start profiling
profiler.enable()

time_since_motion = 0

while camera.IsGrabbing() and elapsed_time < video_duration:
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    elapsed_time = time.time() - start_time

    if grabResult.GrabSucceeded():
        # access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()

        frame = img
        writer.write(img)
    # apply background subtraction
    fgmask = fgbg.apply(img)

    kernel_size = (5, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    kernel = cv2.GaussianBlur(kernel, kernel_size, 0)
    fgmask = cv2.filter2D(fgmask, -1, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    cv2.dilate(fgmask, kernel, dst=fgmask, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 100
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    min_width = 50
    min_height = 50
    max_width = 50
    max_height = 50

    if max_contour is not None and len(max_contour) >=5 :
        # Fit an ellipse around the contour
        ellipse = cv2.fitEllipse(max_contour)

        # Extract ellipse parameters
        center, axes, angle = ellipse
        major_axis, minor_axis = axes

        # Convert the angle to radians for rotation
        angle_rad = np.radians(angle)

        # Draw the rotating ellipsoidal bounding box
        rect_points = cv2.boxPoints(((center[0], center[1]), (major_axis, minor_axis), angle))
        rect_points = np.int0(rect_points)
        cv2.drawContours(img, [rect_points], 0, (0, 255, 0), 2)

        # Update velocity calculation using the rotated bounding box
        if prev_center is not None:
            current_time = time.time()
            dt = current_time - prev_time
            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]

            # Rotate velocity vector back to the original image orientation
            rotated_dx = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
            rotated_dy = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)

            velocity = (rotated_dx / dt, rotated_dy / dt)
            prev_time = current_time
        prev_center = center

        if cv2.norm(velocity) > 5:
            time_since_motion = 0
            last_motion = time.time()

    else:
        time_since_motion += elapsed_time
        velocity = (0,0)

    time_without_motion = time.time() - last_motion

    if cv2.norm(velocity) > 5:
        time_without_motion = 0

    if elapsed_time >= 10 and not sound_played:
        if cv2.norm(velocity) < 100:  # for males
            sound_time.append(elapsed_time)
            pygame.mixer.music.play()
            print([sound_time[0]])
            sound_played = True
            cv2.putText(frame, "sound ON ", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, "Time: " + str(round(elapsed_time, 1)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    raw_writer.write(img)

    cv2.putText(img, f"Time without motion: {(time_without_motion)}", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 500), 2)

    if time_without_motion >= 300:
        cv2.putText(img, f"SLEEP DETECTED", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 500), 2)

    if bbox is not None:
        cv2.ellipse(img, bbox, (0, 255, 0), 2)

        cv2.putText(img, "Velocity: {:.2f} pixels/sec".format(cv2.norm(velocity)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if bbox is None:
        velocity == 0

    resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Motion Detection", resized_img)
    cv2.resizeWindow("Motion Detection", 10, 120)

    if cv2.waitKey(1) == ord('q'):
        break

    if elapsed_time > 5:  # Filter out data points where time is below 5000 milliseconds
        time_list.append(elapsed_time)
        velocity_list.append(cv2.norm(velocity))

print(elapsed_time)

grabResult.Release()

camera.Close()
cv2.destroyAllWindows()
writer.release()
raw_writer.release()

with open(csv_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    pixels_to_mm = 193 / 10  # Conversion factor from pixels to millimeters
    writer.writerow(['Time (milliseconds)', 'Locomotion (mm/sec)', 'First Sound Time'])
    for i in range(len(time_list)):
        scaled_velocity_mm = round(velocity_list[i] / pixels_to_mm, 2)
        writer.writerow([round(time_list[i] * 1000, 1), scaled_velocity_mm, (sound_time[0] * 1000)])


# Plot the data for Time above 5000 milliseconds
time_above_5000 = [t for t in time_list if t > 5]
velocity_above_5000 = [v for t, v in zip(time_list, velocity_list) if t > 5]

conversion_factor = 1 / 19.3  # Convert pixels to cm, then to mm
velocity_above_5000_mm_s = [v * conversion_factor for v in velocity_above_5000]

plt.plot(time_above_5000, velocity_above_5000_mm_s)  # Use the converted velocities
plt.xlim([5, 15])
plt.ylim([0,35])  # Set the y-axis limit based on converted velocities
plt.xlabel("Time (s)")
plt.ylabel("Movement (mm/s)")  # Adjust y-axis label
plt.axvline(x=[sound_time[0]], color='green', label='axvline - full height')
plt.title("Locomotion / Time (Above 5000 ms)")
plt.savefig(plotname)
plt.show()

# Stop profiling
profiler.disable()

csv = prof_to_csv(profiler)
with open("prof_data.csv", 'w+') as f:
    f.write(csv)

# Save the profile data to a file
profiler.dump_stats("profile_data.prof")
