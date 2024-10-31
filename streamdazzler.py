from vidgear.gears import CamGear
import cv2
import pytesseract
import re

# set desired quality as best
options = {"STREAM_RESOLUTION": "best"}

# Add any desire Video URL as input source
# for e.g https://www.twitch.tv/shroud
# and enable Stream Mode (`stream_mode = True`)
stream = CamGear(
    source="https://www.twitch.tv/kempsauce",
    stream_mode=True,
    logging=True,
    # colorspace="COLOR_BGR2GRAY",
    # backend=cv2.CAP_GSTREAMER,
    **options
).start()

# get Video's metadata as JSON object
video_metadata =  stream.ytv_metadata

# print all available keys
print(video_metadata)

video_res = video_metadata.get("resolution")
    
video_width = int(video_res.split("x")[0])

video_height = int(video_res.split("x")[1])

print(video_width, video_height)

frame_count = 0
frame_reset = 1000000

uuid_regex = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')

# loop over
while True:
    if frame_count > frame_reset:
        frame_count = 0
    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break
    
    frame_count += 1
    # get top right hand corner of the frame where match id is
    match_id = frame[0:30, video_width-300:video_width]
    
    # Check every 10 frames
    if(frame_count % 10 == 0):
        print("Checking frame", frame_count)
        
        # Convert to grayscale
        match_id_gray = cv2.cvtColor(match_id, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        match_id_thresh = cv2.adaptiveThreshold(
            match_id_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,  # Block size - adjust based on your text size
            5    # C constant - adjust based on your contrast
        )
        
        cv2.imshow("Match ID Threshold", match_id_thresh)
        
        
        match_id_thresh = cv2.bitwise_not(match_id_thresh)
        
        # Apply morphological operations to clean up the text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(match_id_thresh, cv2.MORPH_CLOSE, kernel)
        
        cv2.imshow("Match ID Morph", morph)
        
        # Apply some noise reduction
        match_id_denoised = cv2.fastNlMeansDenoising(morph, None, 10, 7, 21)
        
        cv2.imshow("Match ID Denoised", match_id_denoised)
        
        match_id_resized = cv2.resize(match_id_denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789abcdef- -c tessedit_fix_strings=1'
        
        # get text from frame
        text = pytesseract.image_to_string(match_id_resized, config=custom_config).strip().lower()

        cv2.imshow("Match ID Processed", match_id_resized)
        print("Text: ", text)
        # Check if text is not empty and is a valid uuid
        if text != "" and len(text.strip()) == 36:
            # Use regex to check if text is a valid uuid v4
            if uuid_regex.match(text):
                print("FOUND MATCH", text)
                break
    
    
    # Show match id window
    cv2.imshow("Match ID", match_id)

    # Show output window
    # cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()