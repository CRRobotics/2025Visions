"""
GUIDE TO CODE:
This code runs objects through a color filter, fits a contour and a convexHull
to each continuous shape that passed the color filter, fits an ellipse to the convexHull, 
and then calculates and returns the polar distance and angle in relation to the camera 
to the center of the ellipse.

Program Flaws:
- Often struglles to differientiate between bumpers, depending on the color (ai might be better)
- Gets extremely inaccurate if the angle between the center of the object and the center of the camera's view is small
- Gets slightly inaccurate the further to the edge of the FOV

Potential Improvements:
- Make all variables must be a certain type in functions
- If angle between center of object and center of camera's view is too small, find the distance to another point 
- If angle between center of object and center of camera's view is too small, use a different method to calculate distance
"""

import cv2
import numpy as np
import math


# CONSTANTS

HUE = 179 #Used for invertedHueMask

BLUR_SIZE = 4 #Used for colorMask

UPPER = np.array([19, 219, 197]) #Used for color filter
LOWER = np.array([7, 78, 61])

CAMERA_CENTER_ANGLE_DEGREES = 0 #Used in pose estimation
DISTANCE_FROM_CAMERA_CENTER_TO_OBJECT_CENTER = -.7 + 8

FOV_HEIGHT_DEGREES = 41 #Used in pose estimation
FOV_HEIGHT_PIX = 240
FOV_WIDTH_DEGREES = 54
FOV_WIDTH_PIX = 320

OBJECT_WIDTH_IN = 8 #Used to fit ellipses
OBJECT_THICKNESS_IN = 8

X_ERROR_M = 0 #Used in undoError, mostly unnecessary
X_ERROR_B = 0
Z_ERROR_M = 0
Z_ERROR_B = 0

# PRIMARY FUNCTIONS

def processImage(image):
    """The main image processing function, returns the image with things drawn on it, the mask, and the X and Z coorinates of objects found."""
    convexHull, mask = findObjectContours(image)
    ellipses, convexHull = fitEllipsesToObjects(convexHull)
    centers = [ellipse[0] for ellipse in ellipses]
    distances2, groundAngles = computeObjectCoordsFromCenters(centers, image)
    convexHull, distances2, groundAngles, ellipses = closestObject(convexHull, distances2, groundAngles, ellipses)
    xCoords, zCoords = polarToRectangular(distances2, groundAngles)

    displayText = [str(round(xCoords[i], 1)) + ", " + str(round(zCoords[i], 1)) for i in range(len(ellipses))] # X coord, Z coord
    toDisplay = drawEllipses(ellipses, displayText, image)
    toDisplay = cv2.drawContours(toDisplay, convexHull, -1, (0, 0, 255), 2)

    # If xCoords and zCoords are empty, 0 will be pushed to NetworkTables for both
    xCoords.append(0)
    zCoords.append(0)

    return toDisplay, mask, xCoords, zCoords

# IMAGE READING STUFF

def findObjectContours(image):
    """Filters the image for objects and returns a list of convexHulls where they are, plus the mask."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.medianBlur(image, BLUR_SIZE * 2 + 1)
    mask = cv2.inRange(blurred, LOWER, UPPER)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask
    convexHull = [cv2.convexHull(contour) for contour in contours]
    return convexHull, mask

def fitEllipsesToObjects(convexHull):
    """Given a list of contours, finds ellipses that represent the center circles of objects, also returns the contours that could have ellipses fit to them."""
    ellipses = []
    passedHulls = []
    for hull in convexHull:
        if len(hull) < 5: continue # fitEllipse needs at least 5 points

        ellipse = cv2.fitEllipse(hull)
        newMajor = ellipse[1][1] * (1 - OBJECT_THICKNESS_IN / OBJECT_THICKNESS_IN) # Shrink the ellipse to be at roughly the center of the torus
        newMinor = ellipse[1][0] - ellipse[1][1] * OBJECT_THICKNESS_IN / OBJECT_THICKNESS_IN # Removing the same amount from the minor axis as the major axis
        ellipse = list(ellipse) # Tuples must be converted to lists to edit their contents
        ellipse[1] = (newMinor, newMajor)
        ellipse = tuple(ellipse)

        ellipses.append(ellipse)
        passedHulls.append(hull)
    return ellipses, passedHulls

def drawEllipses(ellipses, textToDisplay, image):
    """Displays the inputted array of ellipses on image with textToDisplay (an array of the same length) at their centers."""
    toReturn = image.copy()
    for i in range(len(ellipses)):
        ellipse = ellipses[i]
        text = textToDisplay[i]

        try:
            ellipseCenter = tuple([int(coord) for coord in ellipse[0]])
            toReturn = cv2.circle(toReturn, ellipseCenter, 0, (0, 0, 0), 5)
            toReturn = cv2.ellipse(toReturn, ellipse, (255, 255, 0), 2)
            toReturn = cv2.putText(toReturn, str(text), ellipseCenter, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        except:
            # Errors this catches: ellipse has infinity in it, ellipse is zero size, center doesn't work for text
            print("Can't display ellipse")
    return toReturn


# POS MATH

def computeObjectCoordsFromCenters(centers, image):
    """Uses the center point of a object to calculate its z-coordinate based on the tilt of the camera, assuming it is on the floor and the center point is below the camera."""
    distances = []
    angles = []
    for vertex in centers:
        # Some code from 2022
        opticalHorizontalAngle = getOpticalAngle(image, 0, vertex) # if vertex is not None else getOpticalAngle(image, 0, centers[0])
        groundHorizontalAngle = horizontalOpticalToGround(opticalHorizontalAngle)
        opticalVerticalAngle = getOpticalAngle(image, 1, vertex) # if vertex is not None else getOpticalAngle(image, 1, centers[0])
        groundVerticalAngle = verticalOpticalToGround(opticalHorizontalAngle, opticalVerticalAngle)

        horizontalDistance = getHorizontalDistance(groundVerticalAngle)
        groundHorizontalAngle = math.pi / 2 - groundHorizontalAngle # Convert from angle from center to angle from X-axis

        distances.append(horizontalDistance)
        angles.append(groundHorizontalAngle)
    return distances, angles

def closestObject(contours, distances, angles, ellipses):
    """Finds the closest and biggest contour."""
    if len(ellipses) == 0:
        return [], [], [], []
    maxObject = 0
    maxIndex = -1
    for i in range(len(ellipses)):
        contour = contours[i]
        distance = distances[i]
        
        if distance == 0: sizeDistance = 639639639
        else: sizeDistance = cv2.contourArea(contour) / distance
        if sizeDistance > maxObject:
            maxObject = sizeDistance
            maxIndex = i
    return [contours[maxIndex]], [distances[maxIndex]], [angles[maxIndex]], [ellipses[maxIndex]]

# POS MATH HELPER FUNCTIONS

def getOpticalAngle(img, orientation:int, coordinate:tuple):
    """
        Gets the angle based on the fov, orientation, and coordinate of the parabola representing the target. From 2022.
        @param img The image or frame to use
        @param orientation The orientation of the angle (0 gets horizontal angle to the coordinate, 1 gets vertical angle to the tape)
        NOTE: vertical angle is only accurate if the x-coordinate of the center coordinate is in the middle of the frame
        @param coordinate The coordinate of the center of the tape or parabola
    """
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

def getHorizontalDistance(angle, degrees=False, heightToTarget=DISTANCE_FROM_CAMERA_CENTER_TO_OBJECT_CENTER):
    """Determines the horizontal distance to the target based on the angle and height of the target relative to the robot. From 2022."""
    return heightToTarget / math.tan(math.radians(angle)) if degrees else heightToTarget / math.tan(angle)

def horizontalOpticalToGround(angle):
    """Converts the horizontal angle relative to the optical axis to the horizontal angle relative to the ground. From 2022."""
    return math.atan((1 / math.cos(math.radians(CAMERA_CENTER_ANGLE_DEGREES))) * math.tan(angle))

def verticalOpticalToGround(opticalHorizontalAngle, opticalVerticalAngle):
    """Converts the vertical angle relative to the optical axis to the vertical angle relative to the ground. From 2022."""
    return math.asin(math.cos(math.radians(CAMERA_CENTER_ANGLE_DEGREES)) * math.sin(opticalVerticalAngle) + \
        math.cos(opticalHorizontalAngle) * math.cos(opticalVerticalAngle) * math.sin(math.radians(CAMERA_CENTER_ANGLE_DEGREES)))

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
    """Returns an HSV mask with the hue range inverted, similar to what LimeLight's built-in color threshold can do. Likely won't be needed."""
    lower1 = np.array([0, LOWER[1], LOWER[2]])
    upper1 = np.array([LOWER[0], UPPER[1], UPPER[2]])
    lower2 = np.array([UPPER[0], LOWER[1], LOWER[2]])
    upper2 = np.array([HUE, UPPER[1], UPPER[2]])
    mask1 = cv2.inRange(image, lower1, upper1)
    mask2 = cv2.inRange(image, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)

def undoError(givenXs, givenZs):
    """Uses the given X and Z and the error of the given X and Z, adjusts for the error of the functions. We're too good for this, though ;)."""
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
    toDisplay, mask, xCoords, zCoords = processImage(image)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    llpython = [xCoords[0], zCoords[0], 0, 0, 0, 0, 0, 0]

    return [], toDisplay, llpython