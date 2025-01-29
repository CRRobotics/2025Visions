"""
GUIDE TO CODE:
This code runs objects through a color filter, fits a contour and a convexHull
to each continuous shape that passed the color filter, and then calculates and 
returns the polar distance and angle in relation to the camera to the center 
of mass of the continuous shape that takes up the most pixels, with the 
assumption that the object is on the ground.

Program Flaws:
- Often struglles to differientiate between bumpers, depending on the color (ai might be better)
- Currently assuming the center of mass is on the ground instead of the bottom of the algae.

Potential Improvements
- Make all variables must be a certain type in functions

Assumptions:
- Object is on the ground
- 
"""

import cv2
import numpy as np
import math

# CONSTANTS

MAX_HUE = 179 #The maximum value hue can be. It's 180 because hue is measured it degrees. Used for invertedHueMask

BLUR_SIZE = 4 #Used for color colorMask

UPPER = np.array([150, 255, 255]) #The upper and lower HSV bounds for what color the object is. Used for color filter
LOWER = np.array([50, 188, 50])

CAMERA_CENTER_ANGLE_DEGREES = -29 #Used in pose estimation
CAMERA_HEIGHT_IN = -8.5
DISTANCE_FROM_OBJECT_CENTER_TO_GROUND = 8

FOV_HEIGHT_DEGREES = 41 #Used in pose estimation
FOV_HEIGHT_PIX = 240
FOV_WIDTH_DEGREES = 54
FOV_WIDTH_PIX = 320

X_ERROR_M = 0.96 #Used in undoError, mostly unnecessary
X_ERROR_B = 0.69
Z_ERROR_M = 0.92
Z_ERROR_B = 8.06

# PRIMARY FUNCTIONS

def processImage(image):
    """The main image processing function, returns the image with things drawn on it, the colorMask, and the X and Z coorinates of notes found."""
    toDisplay = image
    pixelCenters, convexHulls, colorMask, contours = findAlgae(toDisplay)
    distances2, groundAngles = computeAlgaeCoordsFromPixelCenters(pixelCenters, toDisplay)
    try:
        convexHull, distances2, groundAngles, pixelCenter, contour = closestAlgae(convexHulls, distances2, groundAngles, pixelCenters, contours)
        toDisplay = cv2.drawContours(toDisplay, contour, -1, (0, 0, 255), 2)
        toDisplay = cv2.putText(toDisplay, "distance " + str(distances2), (5,435), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
        toDisplay = cv2.putText(toDisplay, "angle (radians) " + str(groundAngles), (5,460), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
        toDisplay = cv2.circle(toDisplay, (pixelCenter), 0, (0, 0, 0), 10)
        toDisplay = cv2.drawContours(toDisplay, convexHulls, -1, (0, 0, 255), 2)
        toDisplay = cv2.drawContours(toDisplay, convexHull, -1, (0, 255, 0), 5)
        #print("algae present") #used during testing
    except:
        #print("no algae present") #used during testing
        pass
    xCoords, zCoords = polarToRectangular(distances2, groundAngles)

    xCoords.append(0)
    zCoords.append(0)


    return toDisplay, colorMask, xCoords, zCoords

def findAlgae(image):
    """finds the pixelCenters of the contours of the image and returns the contours and the images that pass"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.medianBlur(image, BLUR_SIZE * 2 + 1)
    colorMask = cv2.inRange(blurred, LOWER, UPPER)

    contours, _ = cv2.findContours(colorMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return [], [], colorMask, []

    convexHulls = [cv2.convexHull(contour) for contour in contours]
    pixelCenters = []
    sufficientHulls = []
    sufficientContours = []
    isTrue = True
    for i in range(len(convexHulls)):
        try:
            moments = cv2.moments(convexHulls[i])
            pixelCenter = ((int(moments['m10']/moments['m00'])), int(moments['m01']/moments['m00']))
            pixelCenters.append(pixelCenter)
            sufficientHulls.append(convexHulls[i])
            sufficientContours.append(contours[i])
        except:
            pass
    return pixelCenters, sufficientHulls, colorMask, contours

# POS MATH

def computeAlgaeCoordsFromPixelCenters(pixelCenters, image):
    """Uses the pixelCenter point of a note to calculate its z-coordinate based on the tilt of the camera, assuming it is on the floor and the center point is below the camera."""
    distances = []
    angles = []
    for vertex in pixelCenters:
        opticalHorizontalAngle = getOpticalAngle(image, 0, vertex) # if vertex is not None else getOpticalAngle(image, 0, pixelCenters[0])
        groundHorizontalAngle = horizontalOpticalToGround(opticalHorizontalAngle)
        opticalVerticalAngle = getOpticalAngle(image, 1, vertex) # if vertex is not None else getOpticalAngle(image, 1, pixelCenters[0])
        groundVerticalAngle = verticalOpticalToGround(opticalHorizontalAngle, opticalVerticalAngle)

        horizontalDistance = getHorizontalDistance(groundVerticalAngle)
        groundHorizontalAngle = math.pi / 2 - groundHorizontalAngle # Convert from angle from pixelCenter to angle from X-axis

        distances.append(horizontalDistance)
        angles.append(groundHorizontalAngle)
    return distances, angles

def closestAlgae(convexHulls, distances, angles, pixelCenters, contours):
    """Finds the biggest contour and returns it, the polar coordinates of it in relation to the camera, and the pixel_pixelCenter of it."""
    maxNote = 0
    maxIndex = -1
    for i in range(len(convexHulls)): 
        convexHull = convexHulls[i]
        distance = distances[i] + 1
        if distance == 0: sizeDistance = 1
        else: sizeDistance = cv2.contourArea(convexHull)
        if sizeDistance > maxNote:
            maxNote = sizeDistance
            maxIndex = i
    return [convexHulls[maxIndex]], [distances[maxIndex]], [angles[maxIndex]], pixelCenters[maxIndex], contours[maxIndex]

# POS MATH HELPER FUNCTIONS

def getOpticalAngle(img, orientation:int, coordinate:tuple):
    """Gets the angle based on the fov, orientation, and coordinate of the parabola representing the target. From 2022."""
    h, w, _ = img.shape

    px = coordinate[0]
    py = coordinate[1]

    centerPixel = (int(w / 2), int(h / 2))
    if orientation == 0:
        distanceFromCenter = px - centerPixel[0]
        radiansPerPixelWidth = math.radians(FOV_WIDTH_DEGREES) / FOV_WIDTH_PIX
        angle = radiansPerPixelWidth * distanceFromCenter
    elif orientation == 1:
        distanceFromCenter = centerPixel[1] - py
        radiansPerPixelHeight = math.radians(FOV_HEIGHT_DEGREES) / FOV_HEIGHT_PIX
        angle = radiansPerPixelHeight * distanceFromCenter
    return angle

def getHorizontalDistance(angle, degrees=False, heightToTarget=CAMERA_HEIGHT_IN + DISTANCE_FROM_OBJECT_CENTER_TO_GROUND):
    """Determines the horizontal distance to the target based on the angle and height of the target relative to the robot. From 2022."""
    return heightToTarget / math.tan(math.radians(angle)) if degrees else heightToTarget / math.tan(angle)

def horizontalOpticalToGround(angle):
    """Converts the horizontal angle relative to the optical axis to the horizontal angle relative to the ground. From 2022."""
    return math.atan((1 / math.cos(math.radians(CAMERA_CENTER_ANGLE_DEGREES))) * math.tan(angle))

def verticalOpticalToGround(opticalHorizontalAngle, opticalVerticalAngle):
    """Converts the vertical angle relative to the optical axis to the vertical angle relative to the ground. From 2022."""
    return math.asin(math.cos(math.radians(CAMERA_CENTER_ANGLE_DEGREES)) * math.sin(opticalVerticalAngle)
        + math.cos(opticalHorizontalAngle) * math.cos(opticalVerticalAngle) * math.sin(math.radians(CAMERA_CENTER_ANGLE_DEGREES)))

# GENERAL HELPER FUNCTIONS

def polarToRectangular(rs, thetas):
    """Converts the given polar coordinates (r, theta) into rectangular coordinates (x, y)."""
    xs = []
    ys = []
    for i in range(len(rs)):
        r = rs[i]
        theta = thetas[i]

        x = r * math.cos(theta)
        y = r * math.sin(theta)

        xs.append(x)
        ys.append(y)
    return xs, ys

def invertedHueMask(image):
    """Returns an HSV colorMask with the hue range inverted, similar to what LimeLight's built-in color threshold can do. Likely won't be needed."""
    lower1 = np.array([0, LOWER[1], LOWER[2]])
    upper1 = np.array([LOWER[0], UPPER[1], UPPER[2]])
    lower2 = np.array([UPPER[0], LOWER[1], LOWER[2]])
    upper2 = np.array([MAX_HUE, UPPER[1], UPPER[2]])
    mask1 = cv2.inRange(image, lower1, upper1)
    mask2 = cv2.inRange(image, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)

def undoError(givenXs, givenZs):
    """Uses the given X and Z and the error of the given X and Z, adjusts for the error of the functions. We're too good for this, though ;)."""
    #Generally unnecessary, provided in case you want to use it
    adjustedXs = []
    adjustedZs = []
    for i in range(len(givenXs)):
        givenX = givenXs[i]
        givenZ = givenZs[i]

        adjustedX = (givenX - X_ERROR_B) / X_ERROR_M
        adjustedZ = (givenZ - Z_ERROR_B) / Z_ERROR_M

        adjustedXs.append(adjustedX)
        adjustedZs.append(adjustedZ)
    return adjustedXs, adjustedZs

# MAIN

def runPipeline(image, llrobot):
    #Main method, the method that the Limelight runs
    toDisplay, colorMask, xCoords, zCoords = processImage(image)
    colorMask = cv2.cvtColor(colorMask, cv2.COLOR_GRAY2BGR)

    llpython = [xCoords[0], zCoords[0], 0, 0, 0, 0, 0, 0]

    return [], toDisplay, llpython