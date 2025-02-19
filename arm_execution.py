import sys
sys.path.append("../lib")
import unitree_arm_interface
import time
import numpy as np

print("Press ctrl+\\ to quit process.")

np.set_printoptions(precision=3, suppress=True)
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
armState = unitree_arm_interface.ArmFSMState
arm.loopOn()

arm.labelRun("forward")
#extract positions
def read_positions(file_path):
    positions = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  
            position = list(map(float, parts[:4])) 
            positions.append(position)
    return positions

positions_file = 'path\to\positions'
positions = read_positions(positions_file)

#For grasping multiple objects, executing the grasping action in a loop
#In each loop, (x1, y1, z1) represents the coordinates of the target object, and (x2, y2, z2) represents the placement coordinates
#(x10, y10, z10) and (x20, y20, z20) are both in the camera coordinate system and need to be converted to the robotic arm coordinate system
for i in range(len(positions) - 1):
    x10, y10, z10, width01 = positions[i][:4]
    x20, y20, z20 , width02= positions[-1][:4]
    np.set_printoptions(precision=3, suppress=True)



    gripper_pos_reach = -1.5
    gripper_pos_grasp = 1
    gripper_pos_release = -1.5
    cartesian_speed = 1
    lastPos = arm.lowstate.getQ()

    #Pose of the grasping point
    roll1 = -1.5
    pitch1 = 0.9
    yaw1 = -0.5
    x10 = round(x10 / 1000, 3)
    y10 = round(y10 / 1000, 3)
    z10 = round(z10 / 1000, 3)
    x1 = z10 #+ width01/1000
    y1 = -x10-0.135
    z1 = -y10+0.055+0.023

    #Pose of the pre-grasping point
    roll = -0.3
    pitch = 0.7
    yaw = 0
    x = x1  - 0.1
    y = y1 
    z = z1 + 0.1
    arm.MoveL(np.array([roll,pitch,yaw, x,y,z]), gripper_pos_reach, cartesian_speed)
    arm.MoveL(np.array([roll1,pitch1,yaw1, x,y,z]), gripper_pos_reach, cartesian_speed)

    #Repeat the action to allow the gripper to open and close at the same position
    arm.MoveL(np.array([roll1,pitch1,yaw1, x1,y1,z1]), gripper_pos_reach, cartesian_speed)
    arm.MoveL(np.array([roll1,pitch1,yaw1, x1,y1,z1]), gripper_pos_grasp, cartesian_speed)
    arm.MoveJ(np.array([roll1,pitch1,yaw1, x1,y1,z1]), gripper_pos_grasp, cartesian_speed)

    arm.labelRun("forward")
    roll2 = -1.0
    pitch2 = 0.8
    yaw2 = -0.5
    x20 = round(x20 / 1000 + 0.1 , 3)
    y20 = round(y20 / 1000, 3)
    z20 = round(z20 / 1000, 3)
    x2 = z20 + width02/1000 -0.13
    y2 = -x20
    z2 = -y20+0.055+0.03
    arm.MoveL(np.array([roll2,pitch2,yaw2,x2,y2,z2]), gripper_pos_grasp, cartesian_speed)
    arm.MoveL(np.array([roll2,pitch2,yaw2,x2,y2,z2]), gripper_pos_release, cartesian_speed)
    arm.MoveJ(np.array([roll2,pitch2,yaw2,x2,y2,z2]), gripper_pos_release, cartesian_speed)
    arm.MoveL(np.array([roll2,pitch2,yaw2,x2,y2,z2]), gripper_pos_release, cartesian_speed)


    arm.backToStart()
    arm.loopOff()
